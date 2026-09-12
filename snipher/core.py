"""Snipher Core — LFM2.5-1.2B-JP を内部構造として動かす統括エンジン。

「2 つの AI を並べるハイブリッド」ではなく、**1 つの Snipher** の中で:

    ┌────────────────────────────── Snipher Core ──────────────────────────────┐
    │  高速コア(数ミリ秒)                                                       │
    │    ├─ 意図判定(対話テーブル)      … 挨拶・感謝などの確実な応答は即座に返す  │
    │    ├─ 確率的下書き + 確信度       … 各スロットの softmax 確率から自信を算出 │
    │    └─ 助動詞の補い(polisher)      … 文末・助動詞・文体の欠落をルールで修復  │
    │                                                                          │
    │  ニューラルコア = LFM2.5-1.2B-JP（内部構造）                               │
    │    ├─ 確率的に不安な部分の文章生成 … 下書きの確信度が低い/自由応答のとき、   │
    │    │                               会話履歴ごと LFM2.5 が本文を生成する    │
    │    ├─ 助動詞の補い(神経系)        … 断片文を LFM2.5 に自然に補完させる      │
    │    ├─ 未知文字の学習              … 学習済みパラメータ(埋め込み)の合成で     │
    │    │                               未知文字を 1 トークン化(torch バックエンド)│
    │    └─ テンプレートフォールバック  … native → 内蔵 ChatML → 素の生成         │
    │                                                                          │
    │  モデルは起動時に全自動取得(レジューム対応・マルチソース)。                  │
    │  バックエンドは llama.cpp(GGUF・最速) / torch(INT8) を自動選択。            │
    └──────────────────────────────────────────────────────────────────────────┘

ニューラルコアが使えない環境（取得失敗・依存なし・低リソース）では、
高速コアだけで常に応答を返す（品質の下限を守る）。
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

from .lfm.assist import AssistConfig, HybridAssist
from .lfm.config import DEFAULT_GGUF_QUANT, LfmConfig
from .polisher import Polisher

log = logging.getLogger(__name__)

# 確実な定形応答（高速コアが即答する意図）。それ以外＝確率的に不安 → ニューラルコア。
CANNED_INTENTS = {
    "greeting", "how_are_you", "identity", "capability", "thanks", "apology",
    "farewell", "agreement", "disagreement", "praise",
}

ROUTE_INSTANT = "instant"   # ⚡ 高速コアが即答
ROUTE_NEURAL = "neural"     # ✨ LFM2.5 が生成（内部構造）
ROUTE_FALLBACK = "fallback" # 高速コアのみ（ニューラル未就绪）


class SnipherCore:
    """高速コア + LFM2.5-1.2B-JP ニューラルコアの統括。"""

    def __init__(self, cfg: LfmConfig | None = None, *, torch_provider=None):
        self.cfg = cfg or LfmConfig()
        self._torch_provider = torch_provider  # api 側のシングルトンと共有するためのコールバック
        self._torch_engine = None
        self.assist = HybridAssist(cfg=AssistConfig.from_env())
        self.polisher: Polisher = self.assist.polisher

        self.backend_kind: str | None = None   # "gguf" | "torch"
        self._gguf = None
        self._acquirer = None
        self.boot_state = "idle"               # idle|booting|fetching|loading|ready|failed|deps_missing
        self.boot_error: str | None = None
        self._boot_started = False
        self._boot_gen = 0                     # reload 時に古い boot ループを止める
        self._pending_backend: str | None = None
        self._boot_lock = threading.RLock()
        self._cancel = threading.Event()
        self._learn_threads: set[threading.Thread] = set()
        self._env_sig = self._env_signature()

    @staticmethod
    def _env_signature() -> tuple:
        import os

        return (
            os.environ.get("SNIPHER_LFM_MODEL", ""),
            os.environ.get("SNIPHER_LFM_GGUF", ""),
            os.environ.get("SNIPHER_LFM_BACKEND", ""),
            os.environ.get("SNIPHER_LFM_STORE_DIR", ""),
        )

    def _check_env(self) -> None:
        """環境変数が変わっていたら設定を作り直す（テスト/ホット再設定用）。"""
        sig = self._env_signature()
        if sig != self._env_sig:
            self._env_sig = sig
            self.cfg = LfmConfig()
            self.assist = HybridAssist(cfg=AssistConfig.from_env())
            self.polisher = self.assist.polisher
            self._acquirer = None
            self._torch_engine = None
            self._boot_started = False
            self.boot_state = "idle"
            self._boot_gen += 1

    # ------------------------------------------------------------------ #
    # バックエンド解決
    # ------------------------------------------------------------------ #
    def torch_engine(self):
        """torch バックエンド（LfmEngine）。依存が無ければ None。"""
        if self._torch_provider is not None:
            return self._torch_provider()
        if self._torch_engine is None:
            from .lfm import LLM_DEPS_AVAILABLE

            if not LLM_DEPS_AVAILABLE:
                return None
            from .lfm.engine import LfmEngine

            self._torch_engine = LfmEngine(self.cfg)
        return self._torch_engine

    def gguf_backend(self):
        if self._gguf is None:
            from .lfm.gguf_backend import GgufBackend, runtime_kind

            if runtime_kind(self.cfg) is None:
                return None
            self._gguf = GgufBackend(self.cfg)
        return self._gguf

    def active_backend(self):
        """現在アクティブ（ready）なニューラルバックエンドを返す。"""
        if self.backend_kind == "gguf" and self._gguf is not None and self._gguf.is_ready:
            return self._gguf
        eng = self.torch_engine()
        if self.backend_kind == "torch" and eng is not None and eng.is_ready:
            return eng
        # boot 前でも既に ready なものがあるなら使う
        if self._gguf is not None and self._gguf.is_ready:
            return self._gguf
        if eng is not None and eng.is_ready:
            return eng
        return None

    def neural_available(self) -> bool:
        return self.active_backend() is not None

    # ------------------------------------------------------------------ #
    # 自動取得
    # ------------------------------------------------------------------ #
    def acquirer(self):
        if self._acquirer is None:
            from .lfm.acquire import build_acquirer

            self._acquirer = build_acquirer(self.cfg)
        return self._acquirer

    def acquire_status(self) -> dict:
        if self._acquirer is None:
            return {"phase": "idle"}
        return self._acquirer.status()

    def fetch_now(self, backend: str | None = None) -> dict:
        """UI/CLI からの明示的な再試行（同期実行せずジョブとして起動）。"""
        if self._boot_started and self.boot_state in ("fetching", "loading"):
            return {"ok": False, "reason": "取得/ロードが進行中です"}
        self._cancel.clear()
        self._boot_started = False
        self._pending_backend = backend
        self.ensure_started()
        return {"ok": True}

    # ------------------------------------------------------------------ #
    # 起動シーケンス（全自動: 取得 → ロード → ready）
    # ------------------------------------------------------------------ #
    def ensure_started(self) -> None:
        self._check_env()
        if self._boot_started:
            return
        with self._boot_lock:
            if self._boot_started:
                return
            self._boot_started = True
            self._boot_gen += 1
            gen = self._boot_gen
        threading.Thread(target=self._boot_loop, args=(gen,), name="snipher-core-boot", daemon=True).start()

    def _choose_backend(self) -> str | None:
        """auto: ローカル指定優先 → gguf(llama.cpp・最速) → torch。環境変数で強制可能。"""
        want = (self.cfg.backend or "auto").lower()
        from .lfm import LLM_DEPS_AVAILABLE
        from .lfm.gguf_backend import runtime_kind

        has_llama = runtime_kind(self.cfg) is not None
        has_torch = LLM_DEPS_AVAILABLE
        # 明示的なローカル指定があればそれが最優先
        if self.cfg.gguf_path and Path(self.cfg.gguf_path).expanduser().exists():
            want = "gguf"
        else:
            src = Path(self.cfg.model_source)
            if src.exists() and src.is_dir():
                want = "torch"
            elif src.suffix.lower() == ".gguf":
                want = "gguf"
        if want == "auto":
            if has_llama:
                return "gguf"
            if has_torch:
                return "torch"
            return None
        if want == "off":
            return None
        if want == "gguf":
            return "gguf" if has_llama else None
        if want == "torch":
            return "torch" if has_torch else None
        return None

    def _local_model(self, backend: str) -> Path | None:
        """ローカルにあるモデル（環境変数 or キャッシュ）を探す。"""
        if backend == "gguf":
            if self.cfg.gguf_path:
                p = Path(self.cfg.gguf_path).expanduser()
                if p.exists():
                    return p
            src = Path(self.cfg.model_source)
            if src.suffix == ".gguf" and src.exists():
                return src
            cached = Path(self.cfg.cache_dir) / f"LFM2.5-1.2B-JP-202606-{DEFAULT_GGUF_QUANT}.gguf"
            if cached.exists() and cached.stat().st_size > 10 * 1024 * 1024:
                return cached
            # キャッシュ内の任意の .gguf
            cdir = Path(self.cfg.cache_dir)
            if cdir.exists():
                for f in sorted(cdir.glob("*.gguf"), key=lambda p: -p.stat().st_size):
                    if f.stat().st_size > 10 * 1024 * 1024:
                        return f
            return None
        # torch
        src = Path(self.cfg.model_source)
        if src.exists() and src.is_dir():
            return src
        if src.exists() and src.is_file():
            return src.parent
        cached = Path(self.cfg.cache_dir) / "LFM2.5-1.2B-JP-202606"
        if (cached / "model.safetensors").exists():
            return cached
        upload = Path(self.cfg.cache_dir) / "upload"
        if (upload / "model.safetensors").exists():
            return upload
        return None

    def _boot_loop(self, gen: int) -> None:
        pending = getattr(self, "_pending_backend", None)
        while not self._cancel.is_set() and gen == self._boot_gen:
            self.boot_error = None
            backend = pending or self._choose_backend()
            pending = None
            if backend is None:
                self.boot_state = "deps_missing"
                self.boot_error = (
                    "ニューラルランタイムがありません。pip install -r requirements-llm.txt (torch) "
                    "または pip install llama-cpp-python (GGUF・最速)"
                )
                log.info(self.boot_error)
                return
            self.backend_kind = backend
            try:
                self.boot_state = "booting"
                local = self._local_model(backend)
                if local is None and self.cfg.auto_fetch:
                    self.boot_state = "fetching"
                    local = self._fetch_model(backend)
                if local is None:
                    # 別バックエンドで local があるなら切替（例: gguf 未取得 but torch キャッシュあり）
                    other = "torch" if backend == "gguf" else "gguf"
                    if self._choose_backend_available(other):
                        alt = self._local_model(other)
                        if alt is not None:
                            backend, local = other, alt
                            self.backend_kind = backend
                if local is None:
                    raise RuntimeError(
                        "モデルを自動取得できませんでした（ネットワーク未接続またはソース期限切れ）。"
                        "再接続後に『再試行』を押すか、SNIPHER_LFM_MODEL でローカル指定してください。"
                    )
                self.boot_state = "loading"
                self._load_backend(backend, local)
                self.boot_state = "ready"
                log.info("Snipher Core: ニューラルコア準備完了 (%s / %s)", backend, local)
                return
            except Exception as exc:  # noqa: BLE001
                log.warning("Snipher Core boot 失敗 (%s): %s", backend, exc)
                self.boot_state = "failed"
                self.boot_error = f"{exc}"
                # gguf がダメなら torch を試す（逆も）
                other = "torch" if backend == "gguf" else "gguf"
                if self._choose_backend_available(other):
                    pending = other
                    continue
                # 依存が無い環境ではリトライしても同じ
                if backend == "torch" and not self.cfg.auto_fetch:
                    return
            # クールダウン後に自動再試行（ネットワーク復帰に追従）
            deadline = time.time() + self.cfg.fetch_retry_seconds
            while time.time() < deadline and gen == self._boot_gen and not self._cancel.is_set():
                time.sleep(0.5)
            if gen != self._boot_gen or self._cancel.is_set():
                return

    def _choose_backend_available(self, backend: str) -> bool:
        if backend == "gguf":
            from .lfm.gguf_backend import runtime_kind

            return runtime_kind(self.cfg) is not None
        from .lfm import LLM_DEPS_AVAILABLE

        return LLM_DEPS_AVAILABLE

    def _fetch_model(self, backend: str) -> Path | None:
        acq = self.acquirer()
        if backend == "gguf":
            plans = acq.plan_gguf()
        else:
            plans = acq.plan_torch()
        ok, paths, err = acq.ensure_plan(plans, backend)
        if not ok or not paths:
            return None
        if backend == "gguf":
            return Path(paths[0])
        # torch: キャッシュディレクトリ（config+tokenizer+safetensors が揃った場所）
        return Path(self.cfg.cache_dir)

    def _load_backend(self, backend: str, path: Path) -> None:
        if backend == "gguf":
            gguf = self.gguf_backend()
            if gguf is None:
                raise RuntimeError("llama.cpp ランタイムがありません")
            gguf.load(path)
            return
        eng = self.torch_engine()
        if eng is None:
            raise RuntimeError("torch/transformers がありません")
        # アップロード済みディレクトリがあればそちらを優先（ホットスワップ互換）
        src = str(path)
        if eng.cfg.model_source != src:
            eng.cfg.model_source = src
        eng.reload(src)
        # reload は非同期。ready/failed になるまで待つ（最大 10 分: 2.2GB の量子化を含む）
        deadline = time.time() + 600
        while time.time() < deadline:
            if eng.state == "ready":
                return
            if eng.state == "failed":
                raise RuntimeError(eng.error or "torch エンジンのロードに失敗")
            if self._cancel.is_set():
                return
            time.sleep(0.25)
        raise RuntimeError("torch エンジンのロードがタイムアウトしました")

    # ------------------------------------------------------------------ #
    # 状態
    # ------------------------------------------------------------------ #
    def status(self) -> dict:
        backend = self.active_backend()
        if backend is not None:
            st = dict(backend.status())
        else:
            eng = self.torch_engine()
            if self.backend_kind == "torch" and eng is not None:
                st = dict(eng.status())
            elif self._gguf is not None:
                st = dict(self._gguf.status())
            elif eng is not None:
                st = dict(eng.status())
            else:
                st = {"state": "unavailable", "error": self.boot_error}
        st["boot_state"] = self.boot_state
        st["boot_error"] = self.boot_error
        st["backend_kind"] = self.backend_kind
        st["neural_ready"] = backend is not None
        st["acquire"] = self.acquire_status()
        st["engine_label"] = backend.engine_name() if backend is not None else None
        return st

    # ------------------------------------------------------------------ #
    # 未知文字の自動学習（torch バックエンド・LFM2.5 のパラメータを適用）
    # ------------------------------------------------------------------ #
    def auto_learn_async(self, text: str) -> list[dict] | None:
        """入力文の未知文字を検出し、即座にバックグラウンドで自動学習する。

        戻り値は検出された未知文字（auto フラグ付き）。学習自体は応答を
        ブロックしない。GGUF バックエンドは byte-fallback のため学習不要。
        """
        eng = self.torch_engine()
        if eng is None or not eng.is_ready or self.backend_kind != "torch":
            return None
        try:
            scan = eng.scan_unknown(text)
        except Exception:  # noqa: BLE001
            return None
        if not scan or not scan.get("unknown"):
            return None
        learned = {c["char"] for c in eng.status()["learned_chars"]}
        todo = []
        for c in scan["unknown"]:
            if c["char"] in learned:
                c["learned"] = True
            else:
                c["auto"] = True
                todo.append(c["char"])
        if todo:
            def _run():
                try:
                    eng.learn_instant(todo)
                    log.info("未知文字を自動学習しました: %s", "".join(todo))
                except Exception:  # noqa: BLE001
                    log.warning("未知文字の自動学習に失敗", exc_info=True)

            th = threading.Thread(target=_run, name="auto-learn", daemon=True)
            self._learn_threads.add(th)
            th.add_done_callback(self._learn_threads.discard)
            th.start()
        return scan["unknown"]

    # ------------------------------------------------------------------ #
    # 会話（内部パイプライン）
    # ------------------------------------------------------------------ #
    def route_of(self, draft: dict, mode: str) -> str:
        """auto モードの経路判定: 確実な定形 → instant / それ以外 → neural。"""
        if mode == "fast":
            return ROUTE_INSTANT
        if not self.neural_available():
            return ROUTE_FALLBACK
        if mode in ("lfm", "neural"):
            return ROUTE_NEURAL
        intent = str(draft.get("intent") or "")
        conf = float(draft.get("confidence", 1.0))
        if intent not in CANNED_INTENTS or conf < self.assist.cfg.threshold:
            return ROUTE_NEURAL
        return ROUTE_INSTANT

    def stream_reply(self, messages: list[dict], *, mode: str = "auto",
                     max_new_tokens: int | None = None, temperature: float | None = None,
                     top_k: int | None = None, repetition_penalty: float | None = None,
                     use_template: bool = True, system_prompt: str | None = None):
        """SSE 用イベントジェネレータ。Snipher Core の内部パイプライン本体。"""
        last_user = next(
            (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), ""
        )

        # 1) 未知文字の自動検出・自動学習（LFM2.5 のパラメータを適用）
        try:
            unknown = self.auto_learn_async(last_user)
            if unknown:
                yield {"type": "meta", "unknown_chars": unknown}
        except Exception:  # noqa: BLE001
            pass

        # 2) 高速コアの下書き（数ミリ秒）
        t0 = time.time()
        draft = self.assist.draft(last_user)
        route = self.route_of(draft, mode)

        if route == ROUTE_NEURAL:
            yield from self._neural_reply(messages, draft, mode,
                                          max_new_tokens=max_new_tokens,
                                          temperature=temperature, top_k=top_k,
                                          repetition_penalty=repetition_penalty,
                                          use_template=use_template,
                                          system_prompt=system_prompt)
            return

        # 3) instant / fallback: 高速コアの応答をそのまま返す（速度維持）
        if route == ROUTE_FALLBACK and float(draft.get("confidence", 1.0)) < self.assist.cfg.threshold:
            text = draft.get("base_text") or draft["text"]  # 不安な生成文は安全な骨子へ
            degraded = text != draft["text"]
        else:
            text = draft["text"]
            degraded = False
        stats = {
            "engine": "Snipher-mini+" if route == ROUTE_FALLBACK else "Snipher Core (高速経路)",
            "template_mode": "rule-based",
            "new_tokens": None,
            "tokens_per_second": None,
            "assist": "rule",
            "draft": draft["text"],
            "draft_confidence": draft.get("confidence"),
            "draft_seconds": round(time.time() - t0, 5),
            "intent": draft.get("intent"),
            "fixes": draft.get("fixes", []),
            "route": route,
        }
        if route == ROUTE_FALLBACK:
            stats["engine"] = "Snipher-mini+"
            stats["fallback_reason"] = self._fallback_reason()
            stats["degraded_to_base"] = degraded
            stats["acquire"] = self.acquire_status().get("phase")
        yield {"type": "assist", "mode": "rule", "confidence": draft.get("confidence"),
               "route": route}
        yield {"type": "start", "engine": stats["engine"], "template_mode": "rule-based"}
        for piece in _chunk_for_stream(text):
            yield {"type": "delta", "text": piece}
        yield {"type": "done", "text": text, "stats": stats}

    def _fallback_reason(self) -> str:
        if self.boot_state == "fetching":
            acq = self.acquire_status()
            pct = acq.get("percent")
            p = f" {pct}%" if pct is not None else ""
            return f"モデルを自動取得中{p}（{acq.get('current_file') or '準備中'}）"
        if self.boot_state == "loading":
            return "モデルをロード中"
        if self.boot_state == "failed":
            return self.boot_error or "ニューラルコアの準備に失敗"
        if self.boot_state == "deps_missing":
            return self.boot_error or "ニューラルランタイム未インストール"
        return "ニューラルコア未起動"

    def _neural_reply(self, messages: list[dict], draft: dict, mode: str, **opts):
        """✨ LFM2.5-1.2B-JP が内部構造として本文を生成する経路。"""
        backend = self.active_backend()
        if backend is None:  # 判定直後に落ちた場合の安全網
            text = draft.get("base_text") or draft["text"]
            yield {"type": "start", "engine": "Snipher-mini+", "template_mode": "rule-based"}
            for piece in _chunk_for_stream(text):
                yield {"type": "delta", "text": piece}
            yield {"type": "done", "text": text, "stats": {
                "engine": "Snipher-mini+", "assist": "rule", "template_mode": "rule-based",
                "draft": draft["text"], "draft_confidence": draft.get("confidence"),
                "intent": draft.get("intent"), "fixes": draft.get("fixes", []),
                "route": ROUTE_FALLBACK, "fallback_reason": self._fallback_reason(),
            }}
            return
        if mode not in ("lfm", "neural"):
            # auto 経路のときだけ「なぜニューラルコアが動くのか」を UI に可視化
            yield {"type": "assist", "mode": "lfm", "draft": draft["text"],
                   "confidence": draft.get("confidence"), "reason": "uncertain_slot",
                   "route": ROUTE_NEURAL}

        # 内部プロンプト: Snipher の人格 + 高速コアの下書きをヒントとして同梱
        sp = opts.get("system_prompt")
        if not sp:
            from .lfm.config import DEFAULT_SYSTEM_PROMPT

            sp = DEFAULT_SYSTEM_PROMPT
            hint = (draft.get("text") or "").strip()
            if hint and draft.get("intent") in ("question", "fallback"):
                sp += f"\n（内部ヒント: 高速コアの下書き「{hint[:120]}」を参考にしつつ、自然な返答を作ってください）"

        collected: list[str] = []
        stats: dict = {}
        failed = False
        try:
            for ev in backend.stream_chat(
                messages,
                max_new_tokens=opts.get("max_new_tokens"),
                temperature=opts.get("temperature"),
                top_k=opts.get("top_k"),
                repetition_penalty=opts.get("repetition_penalty"),
                use_template=opts.get("use_template", True),
                system_prompt=sp,
            ):
                if ev.get("type") == "delta":
                    collected.append(ev.get("text", ""))
                    yield ev
                elif ev.get("type") == "done":
                    stats = dict(ev.get("stats", {}))
                elif ev.get("type") == "error":
                    failed = True
                    log.warning("ニューラル生成エラー: %s", ev.get("message"))
                else:
                    yield ev
        except Exception as exc:  # noqa: BLE001
            failed = True
            log.warning("ニューラル生成の例外: %s", exc)

        text = "".join(collected).strip()
        if text:
            # 助動詞の補い: ニューラル出力にも高速コアの polisher を通す（内部処理）
            polished = self.polisher.polish(text, register="polite")
            fixes = polished["fixes"]
            if polished["text"] != text:
                # 差分だけを追加 delta として送るのは複雑なので、done で確定テキストを返す
                text = polished["text"]
        else:
            # 神経系が空/失敗 → 高速コアの下書きに安全フォールバック
            conf = float(draft.get("confidence", 1.0))
            text = draft["text"] if conf >= self.assist.cfg.threshold else (draft.get("base_text") or draft["text"])
            fixes = draft.get("fixes", [])
            for piece in _chunk_for_stream(text):
                yield {"type": "delta", "text": piece}

        stats.update({
            "assist": "lfm",
            "draft": draft["text"],
            "draft_confidence": draft.get("confidence"),
            "intent": draft.get("intent"),
            "fixes": fixes,
            "route": ROUTE_NEURAL,
            "engine": stats.get("engine") or backend.engine_name(),
        })
        if failed and not collected:
            stats["neural_fallback"] = True
        yield {"type": "done", "text": text, "stats": stats}

    # ------------------------------------------------------------------ #
    # 助動詞の補い（断片文の補完 API 用）
    # ------------------------------------------------------------------ #
    def complete_fragment(self, text: str, *, register: str = "polite",
                          use_neural: bool = True) -> dict:
        """文末・助動詞が欠けた断片文を補完する。

        1. ルール(polisher)で確実に修復（マイクロ秒）
        2. まだ不完全そうなら、LFM2.5 に自然な補完をさせる（内部構造）
        """
        fixed = self.polisher.polish(text, register=register)
        result = {"text": fixed["text"], "fixes": fixed["fixes"], "engine": "rule"}
        backend = self.active_backend() if use_neural else None
        if backend is not None and _looks_incomplete(fixed["text"]):
            msgs = [
                {"role": "user",
                 "content": (
                     "次の日本語の断片文を、助動詞や文末を補って自然な一文にしてください。"
                     "出力は補完した一文のみ。\n"
                     f"断片: {fixed['text']}"
                 )},
            ]
            out: list[str] = []
            try:
                for ev in backend.stream_chat(msgs, max_new_tokens=48, temperature=0.1,
                                              top_k=40, repetition_penalty=1.05):
                    if ev.get("type") == "delta":
                        out.append(ev.get("text", ""))
            except Exception:  # noqa: BLE001
                out = []
            completed = "".join(out).strip().split("\n")[0]
            if completed and len(completed) >= len(fixed["text"]) * 0.5:
                result = {"text": completed, "fixes": fixed["fixes"] + ["neural_completion"],
                          "engine": backend.engine_name()}
        return result

    # ------------------------------------------------------------------ #
    # 管理系
    # ------------------------------------------------------------------ #
    def reload(self, source: str | None = None) -> dict:
        """モデルソースを差し替えて再ロード（アップロード後のホットスワップ等）。"""
        self._cancel.set()
        time.sleep(0.05)
        self._cancel.clear()
        self._boot_started = False
        self._boot_gen += 1
        self.boot_state = "idle"
        if self._gguf is not None:
            self._gguf.unload()
        if source:
            p = Path(source)
            if source.lower().endswith(".gguf"):
                self.cfg.gguf_path = source
            elif p.is_dir():
                self.cfg.model_source = source
                eng = self.torch_engine()
                if eng is not None:
                    eng.cfg.model_source = source
            else:
                self.cfg.model_source = source
        self._pending_backend = None
        self.ensure_started()
        return {"ok": True, "state": "booting"}

    def reset_learned(self) -> dict:
        eng = self.torch_engine()
        if eng is None:
            return {"removed": 0}
        return eng.reset_learned()

    def shutdown(self) -> None:
        self._cancel.set()
        if self._gguf is not None:
            self._gguf.unload()


# ---------------------------------------------------------------------- #
_INCOMPLETE_ENDS = (
    # 助詞で終わっている → 述語(助動詞)が欠落している
    "が", "は", "を", "に", "で", "と", "も", "へ", "や", "より", "まで", "から",
    "ので", "けど", "し", "たら", "ば", "のに", "て", "で", "ながら",
)
_COMPLETE_ENDS = (
    "ます", "です", "ました", "ません", "だ", "た", "ない", "る", "い", "か",
    "ね", "よ", "さ", "な", "わ", "ぞ", "ぜ", "かしら", "でしょう", "だろう",
)


def _looks_incomplete(text: str) -> bool:
    """文末が助動詞で閉じていなければ不完全とみなす（助動詞の補いの対象）。

    「私は猫が好き」「歩いていたら急に雨が」のような、述語・助動詞が
    欠落した断片文を検出する。句読点は無視して判定する。
    """
    t = text.strip().rstrip("。.！!？? ").strip()
    if not t:
        return False
    if t.endswith(_COMPLETE_ENDS):
        return False
    if t.endswith(_INCOMPLETE_ENDS):
        return True
    # 名詞・形容動詞の語幹で終わっている場合（好き/元気/きれい…）も不完全
    return not t.endswith(_COMPLETE_ENDS)


def _chunk_for_stream(text: str, pieces: int = 3) -> list[str]:
    """軽量経路のテキストを擬似ストリーミング用に文単位で分割する。"""
    import re as _re

    parts = [p for p in _re.split(r"(?<=。)|(?<=？)|(?<=！)", text) if p]
    if len(parts) <= pieces:
        return parts or [text]
    merged: list[str] = []
    per = max(1, -(-len(parts) // pieces))
    for i in range(0, len(parts), per):
        merged.append("".join(parts[i : i + per]))
    return merged
