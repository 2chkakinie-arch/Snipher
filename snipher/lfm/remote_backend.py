"""LFM2.5-1.2B-JP フルウェイトを **別の常駐ホストに委譲する** 軽量バックエンド。

Vercel のようなサーバーレスでは 731MB(GGUF)/2.2GB(safetensors) のモデルを
ダウンロードもロードもできない（関数のサイズ・時間制限に物理的に収まらない）。
一方で VPS/Render/Fly などの常駐環境なら同じ Snipher を起動するだけで
LFM2.5-1.2B-JP をロードできる。

そこでこのバックエンドは「Snipher のニューラルコアと同じ顔」をした
**リモート プロキシ** になる:

    GUI(Vercel) ── /api/chat ──> Snipher Core ──> [内蔵蒸留コア or リモート LFM2.5]
                                                      │
                                             POST {remote}/api/chat (SSE)

設定は環境変数 2 つだけ（ユーザー操作・ファイルアップロード不要）:

    SNIPHER_LFM_REMOTE_URL   例 https://snipher-on-vps.onrender.com
    SNIPHER_LFM_REMOTE_TOKEN 任意（Authorization: Bearer で送る）

スニッパ側は OpenAI 互換 (`/v1/chat/completions`) も自動で使うので、
llama.cpp server / vLLM などを向けても動く。
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

log = logging.getLogger(__name__)

_PROBE_TTL = 60.0        # ヘルスチェック結果のキャッシュ（秒）
_CONNECT_TIMEOUT = 3.5
_READ_TIMEOUT = 90.0


class RemoteLfmBackend:
    """`LfmEngine` / `GgufBackend` と同じインターフェースのリモート版。"""

    kind = "remote"

    def __init__(self, base_url: str, *, token: str = "", style: str = "auto",
                 model: str = "LFM2.5-1.2B-JP-202606"):
        self.base_url = (base_url or "").rstrip("/")
        self.token = token or ""
        self.style = style if style in ("auto", "snipher", "openai") else "auto"
        self.model = model
        self.error: str | None = None
        self.load_seconds: float | None = None
        self._lock = threading.RLock()
        self._alive: bool | None = None
        self._alive_at = 0.0
        self._detected: str | None = None      # "snipher" | "openai"
        self._cancel = threading.Event()

    # ------------------------------------------------------------------ #
    # ユーティリティ
    # ------------------------------------------------------------------ #
    def _req(self, path: str, payload: dict | None = None, *, method: str | None = None,
             timeout: float = _READ_TIMEOUT):
        data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(self.base_url + path, data=data, method=method or ("POST" if data else "GET"))
        req.add_header("Content-Type", "application/json")
        req.add_header("Accept", "text/event-stream, application/json")
        if self.token:
            req.add_header("Authorization", f"Bearer {self.token}")
        return urllib.request.urlopen(req, timeout=timeout)

    @property
    def is_ready(self) -> bool:
        return self.probe()

    def probe(self, *, force: bool = False) -> bool:
        with self._lock:
            if not force and self._alive is not None and time.time() - self._alive_at < _PROBE_TTL:
                return self._alive
        ok = False
        err = None
        for path in ("/health", "/api/status", "/docs"):
            try:
                with self._req(path, timeout=_CONNECT_TIMEOUT + 1.5) as resp:
                    if 200 <= resp.status < 400:
                        resp.read(4096)
                        ok = True
                        break
            except Exception as exc:  # noqa: BLE001
                err = str(exc)
        with self._lock:
            self._alive = ok
            self._alive_at = time.time()
            self.error = None if ok else f"リモート LFM2.5 に接続できません ({err or 'no response'})"
        return ok

    def engine_name(self) -> str:
        host = urllib.parse.urlsplit(self.base_url).netloc or self.base_url
        return f"LFM2.5-1.2B-JP @ {host}（リモート委譲）"

    def status(self) -> dict:
        return {
            "kind": "remote",
            "state": "ready" if self._alive else ("unknown" if self._alive is None else "unavailable"),
            "engine": self.engine_name(),
            "base_url": self.base_url,
            "style": self._detected or self.style,
            "error": self.error,
            "is_local": False,
            "learned_chars": [],
            "backend": "remote-http",
            "load_seconds": self.load_seconds,
            "download_required": False,
            "auth": bool(self.token),
        }

    def unload(self) -> None:  # pragma: no cover
        self._cancel.set()
        with self._lock:
            self._alive = None

    def cancel(self) -> None:
        self._cancel.set()

    def scan_unknown(self, text: str) -> dict | None:
        """リモート側の語彙は触れないので未知文字学習は行わない。"""
        return None

    # ------------------------------------------------------------------ #
    # 生成
    # ------------------------------------------------------------------ #
    def _candidate_paths(self) -> list[tuple[str, str]]:
        order = [self._detected] if self._detected in ("snipher", "openai") else ["snipher", "openai"]
        if self.style in ("snipher", "openai") and self.style not in order:
            order.insert(0, self.style)
        return [(s, "/api/chat" if s == "snipher" else "/v1/chat/completions") for s in order]

    def _steer_forwarder(self, stop: threading.Event) -> None:
        """生成中に届いたステアリング波をリモートの /api/steer へ転送する裏スレッド。

        リモートが Snipher なら、向こう側の SteeringBus が同じ確率波を
        ロジットに干渉させる（出力は止まらない）。ベストエフォート。
        """
        try:
            from .steering import get_steering_bus
            bus = get_steering_bus()
        except Exception:  # noqa: BLE001
            return
        if bus is None:
            return
        last_id = bus.last_id()
        while not stop.wait(0.2):
            try:
                for sig in bus.snapshot_new(last_id):
                    last_id = max(last_id, sig.id)
                    if sig.kind != "prompt":
                        continue
                    try:
                        with self._req("/api/steer", {"text": sig.text,
                                                      "strength": sig.strength},
                                       timeout=2.0) as resp:
                            resp.read(256)
                    except Exception:  # noqa: BLE001
                        pass
            except Exception:  # noqa: BLE001
                pass

    def stream_chat(self, messages: list[dict], *, max_new_tokens: int | None = None,
                    temperature: float | None = None, top_k: int | None = None,
                    repetition_penalty: float | None = None, use_template: bool = True,
                    system_prompt: str | None = None, **_ignored):
        """SSE を中継して Snipher イベント（start/delta/done/error）に変換する。"""
        if not self.base_url:
            yield {"type": "error", "message": "SNIPHER_LFM_REMOTE_URL が未設定です"}
            return
        if not self.probe():
            yield {"type": "error", "message": self.error or "リモートに接続できません"}
            return
        # リアルタイム・ステアリング: 生成中に届いた介入プロンプトをリモートへ転送
        stop_steer = threading.Event()
        fwd = threading.Thread(target=self._steer_forwarder, args=(stop_steer,),
                               daemon=True, name="snipher-steer-fwd")
        fwd.start()
        try:
            payload = {
                "messages": messages,
                "mode": "lfm",
                "max_new_tokens": int(max_new_tokens or 128),
                "temperature": float(temperature if temperature is not None else 0.3),
                "top_k": int(top_k or 50),
                "repetition_penalty": float(repetition_penalty or 1.05),
            }
            if system_prompt:
                payload["system_prompt"] = system_prompt
            last_err = None
            for style, path in self._candidate_paths():
                body = dict(payload)
                if style == "openai":
                    body = {"model": self.model, "messages": messages,
                            "max_tokens": payload["max_new_tokens"], "temperature": payload["temperature"],
                            "stream": True}
                self._cancel.clear()
                t0 = time.time()
                collected: list[str] = []
                done_stats: dict = {}
                yielded_start = False
                try:
                    with self._req(path, body) as resp:
                        ctype = (resp.headers.get("Content-Type") or "").lower()
                        with self._lock:
                            self._detected = style
                            if self.load_seconds is None:
                                self.load_seconds = round(time.time() - t0, 3)
                        if "text/event-stream" in ctype:
                            for raw in resp:
                                if self._cancel.is_set():
                                    break
                                line = raw.decode("utf-8", "replace").strip()
                                if not line.startswith("data:"):
                                    continue
                                chunk = line[5:].strip()
                                if not chunk or chunk == "[DONE]":
                                    continue
                                try:
                                    ev = json.loads(chunk)
                                except json.JSONDecodeError:
                                    continue
                                ev = _normalize(ev, style)
                                et = ev.get("type")
                                if et == "delta":
                                    if not yielded_start:
                                        yielded_start = True
                                        yield {"type": "start", "engine": self.engine_name(),
                                               "template_mode": f"remote-{style}"}
                                    collected.append(ev.get("text", ""))
                                    yield ev
                                elif et == "done":
                                    done_stats = dict(ev.get("stats") or {})
                                    done_stats.setdefault("text", ev.get("text"))
                                elif et == "error":
                                    raise RuntimeError(ev.get("message") or "リモート側でエラー")
                                else:
                                    if et == "start":
                                        yielded_start = True      # 二重 start を防ぐ
                                    yield ev
                        else:                                     # 非ストリーミング JSON
                            obj = json.loads(resp.read().decode("utf-8", "replace") or "{}")
                            text = _extract_text(obj, style)
                            yield {"type": "start", "engine": self.engine_name(),
                                   "template_mode": f"remote-{style}"}
                            if text:
                                collected.append(text)
                                yield {"type": "delta", "text": text}
                            done_stats = {"new_tokens": len(text)}
                except Exception as exc:  # noqa: BLE001
                    last_err = f"{type(exc).__name__}: {exc}"
                    log.warning("リモート LFM2.5 生成失敗 (%s %s): %s", style, path, exc)
                    if collected:                # 途中で切れてもここまでを返す
                        dt = max(1e-6, time.time() - t0)
                        done_stats.setdefault("tokens_per_second", round(len(collected) / dt, 1))
                        done_stats["remote_truncated"] = last_err
                        break
                    continue                     # 次のスタイルを試す
                dt = max(1e-6, time.time() - t0)
                text = "".join(collected).strip()
                stats = {"engine": self.engine_name(), "backend": "remote", "remote_style": style,
                         "new_tokens": done_stats.get("new_tokens") or len(text),
                         "tokens_per_second": done_stats.get("tokens_per_second")
                         or round(len(text) / dt, 1),
                         "seconds": round(dt, 3), "remote_url": self.base_url}
                stats.update({k: v for k, v in done_stats.items() if k in
                              ("template_mode", "remote_truncated", "prompt_tokens")})
                yield {"type": "done", "text": text, "stats": stats}
                return
            yield {"type": "error", "message": f"リモート LFM2.5 を使えませんでした ({last_err})"}
        finally:
            stop_steer.set()

    def complete(self, fragment: str) -> dict:
        """断片文の補完をリモートに依頼する（内蔵コアと同じ戻り値形式）。"""
        if not self.probe():
            return {"text": fragment, "added": "", "changed": False, "confidence": 0.0}
        out = ""
        try:
            for ev in self.stream_chat(
                [{"role": "user",
                  "content": "次の日本語の断片文を、助動詞や文末を補って自然な一文にしてください。"
                             f"出力は補完した一文のみ。\n断片: {fragment}"}],
                max_new_tokens=48, temperature=0.1):
                if ev.get("type") == "done":
                    out = (ev.get("text") or "").strip()
        except Exception as exc:  # noqa: BLE001
            log.warning("リモート補完に失敗: %s", exc)
            return {"text": fragment, "added": "", "changed": False, "confidence": 0.0}
        out = out.split("\n")[0].strip()
        if not out or len(out) < len(fragment) * 0.5:
            return {"text": fragment, "added": "", "changed": False, "confidence": 0.0}
        added = out[len(fragment):] if out.startswith(fragment) else out
        return {"text": out, "added": added.strip(), "changed": True, "confidence": 0.85}


def _normalize(ev: dict, style: str) -> dict:
    """リモート側のイベントを Snipher の形式に寄せる。"""
    t = str(ev.get("type") or "")
    if t in ("start", "delta", "done", "error", "meta", "assist"):
        return ev
    # OpenAI 互換 SSE
    if style == "openai" or "choices" in ev:
        try:
            ch = (ev.get("choices") or [{}])[0]
        except (IndexError, AttributeError):
            ch = {}
        piece = ((ch.get("delta") or {}).get("content")
                 or (ch.get("message") or {}).get("content") or ch.get("text") or "")
        if piece:
            return {"type": "delta", "text": str(piece)}
        if ev.get("error"):
            return {"type": "error", "message": str(ev["error"])}
        return {"type": "skip"}
    if "content" in ev:
        return {"type": "delta", "text": str(ev.get("content") or "")}
    return {"type": "skip"}


def _extract_text(obj: dict, style: str) -> str:
    if style == "openai" or "choices" in (obj or {}):
        try:
            return str(obj["choices"][0]["message"]["content"])
        except Exception:  # noqa: BLE001
            return ""
    return str((obj or {}).get("text") or (obj or {}).get("reply") or "")


def backend_from_env(cfg) -> RemoteLfmBackend | None:
    """環境変数からバックエンドを作る（未設定なら None）。"""
    url = getattr(cfg, "remote_url", "") or ""
    if not url:
        return None
    return RemoteLfmBackend(url, token=getattr(cfg, "remote_token", "") or "")
