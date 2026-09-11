"""LFM2.5-1.2B-JP 推論エンジン。

- CPU でも高速な日常会話を実現するため、動的 INT8 量子化 + 短いコンテキスト
  + ストリーミング生成（TextIteratorStreamer）を使う。
- チャットテンプレートが無い tokenizer / テンプレート無効モードでも
  TemplateManager が段階フォールバックで必ず生成できる。
- 未知文字は UnknownCharLearner が予約トークン枠で学習する
  （即時: 事前学習済み断片埋め込みの平均 / 深学習: 埋め込みのみ勾配更新）。
"""

from __future__ import annotations

import gc
import logging
import os
import threading
import time
from pathlib import Path

from .config import LfmConfig, DEFAULT_SYSTEM_PROMPT
from .learner import LearnedVocabStore, UnknownCharLearner
from .template import TemplateManager

log = logging.getLogger(__name__)

STATE_IDLE = "idle"
STATE_LOADING = "loading"
STATE_READY = "ready"
STATE_LEARNING = "learning"
STATE_FAILED = "failed"

_ENGINE: "LfmEngine | None" = None
_ENGINE_LOCK = threading.Lock()


def get_engine() -> "LfmEngine":
    global _ENGINE
    with _ENGINE_LOCK:
        if _ENGINE is None:
            _ENGINE = LfmEngine(LfmConfig())
        return _ENGINE


class BusyError(RuntimeError):
    pass


class NotReadyError(RuntimeError):
    pass


class _SuppressReserved:
    """未割当の予約トークン（学習スロットの空き）を出力から抑制する。"""

    def __init__(self, base_vocab: int, total_vocab: int, allowed: set[int]):
        self.bad = [i for i in range(base_vocab, total_vocab) if i not in allowed]

    def __call__(self, input_ids, scores):
        if self.bad:
            scores[:, self.bad] = float("-inf")
        return scores


class LfmEngine:
    def __init__(self, cfg: LfmConfig):
        self.cfg = cfg
        self.state = STATE_IDLE
        self.error: str | None = None
        self.model = None
        self.tokenizer = None
        self.tmpl: TemplateManager | None = None
        self.base_vocab: int = 0
        self.total_vocab: int = 0
        self.quantized: bool = False
        self.load_seconds: float | None = None

        store = LearnedVocabStore(Path(cfg.store_dir))
        try:
            store.load()
        except Exception as exc:
            log.warning("学習済み語彙の読込に失敗: %s", exc)
        self.store = store
        self.learner = UnknownCharLearner(store, reserved=cfg.reserved_tokens)

        self.lifecycle_lock = threading.RLock()  # load/unload/learn_deep で再入する
        self.gen_lock = threading.Lock()
        self._started = False

        try:
            import torch

            torch.set_num_threads(max(1, int(os.environ.get("SNIPHER_TORCH_THREADS", "0")) or min(4, os.cpu_count() or 1)))
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # ライフサイクル
    # ------------------------------------------------------------------ #
    def ensure_started(self) -> None:
        """初回 API アクセス時にバックグラウンドでロードする。"""
        if self._started:
            return
        with self.lifecycle_lock:
            if self._started:
                return
            self._started = True
            t = threading.Thread(target=self._safe_load, name="lfm-loader", daemon=True)
            t.start()

    def _safe_load(self) -> None:
        try:
            self.load()
        except Exception as exc:
            log.exception("モデルのロードに失敗")
            self.state = STATE_FAILED
            self.error = f"{exc}"

    @staticmethod
    def _probe_hf(source: str) -> None:
        """リモートモデルIDのときだけ、短いタイムアウトで接続を事前確認する。

        huggingface_hub 自体は数回のリトライ・バックオフを持つため、
        オフライン環境ではここで素早く明確なエラーにする(既定 5 秒)。
        """
        if os.environ.get("SNIPHER_LFM_SKIP_NET_CHECK", "0").strip() in ("1", "true", "yes"):
            return
        import urllib.request

        url = f"https://huggingface.co/{source}/resolve/main/config.json"
        try:
            req = urllib.request.Request(url, method="HEAD")
            urllib.request.urlopen(req, timeout=5)  # noqa: S310
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"HuggingFace に接続できません(オフライン?): {exc}. "
                "UI の『モデル管理』からローカルのモデルを取り込むか、"
                "SNIPHER_LFM_MODEL=ローカルディレクトリ を指定してください。"
            ) from exc

    def load(self, quantize: bool | None = None) -> None:
        """モデルをロードし（必要なら）INT8 量子化する。"""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        quantize = self.cfg.quantize_int8 if quantize is None else quantize
        with self.lifecycle_lock:
            self.state = STATE_LOADING if self.state != STATE_LEARNING else self.state
            self.error = None
            t0 = time.time()
            src = self.cfg.model_source
            log.info("LFM モデルをロード中: %s (quantize=%s)", src, quantize)
            if not self.cfg.is_local:
                self._probe_hf(src)
            self.tokenizer = AutoTokenizer.from_pretrained(src)
            model = AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16)
            model.eval()

            self.base_vocab = int(model.get_input_embeddings().weight.shape[0])
            self.total_vocab = self.base_vocab + self.cfg.reserved_tokens
            try:
                model.resize_token_embeddings(self.total_vocab, mean_resizing=False)
            except TypeError:  # 古い transformers には mean_resizing が無い
                model.resize_token_embeddings(self.total_vocab)
            # 新規スロットをゼロで初期化（未割当スロットは生成時に抑制する）
            with torch.no_grad():
                emb = model.get_input_embeddings().weight
                emb[self.base_vocab :] = 0
                head_w = getattr(getattr(model, "lm_head", None), "weight", None)
                if head_w is not None and not callable(head_w):
                    if head_w.data_ptr() != emb.data_ptr() and head_w.shape[0] >= self.total_vocab:
                        head_w[self.base_vocab :] = 0

            self.learner.base_vocab = self.base_vocab
            self.learner.apply_persisted(model)

            if quantize:
                from torch.ao.quantization import quantize_dynamic

                model = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
                # 動的量子化 Linear は fp32 入力を要求するため、
                # 量子化しないモジュール（Embedding / norm / conv）を fp32 に寄せる
                for mod in model.modules():
                    if isinstance(
                        mod,
                        (torch.nn.Embedding, torch.nn.LayerNorm, torch.nn.Conv1d),
                    ):
                        mod.to(torch.float32)
                try:
                    from torch.nn import RMSNorm

                    for mod in model.modules():
                        if isinstance(mod, RMSNorm):
                            mod.to(torch.float32)
                except ImportError:
                    pass
                self.quantized = True
            else:
                self.quantized = False

            self.model = model
            self.tmpl = TemplateManager(self.tokenizer)
            self.load_seconds = round(time.time() - t0, 2)
            self.state = STATE_READY
            self._started = True  # 手動ロードでも再ロードを起こさない
            log.info("LFM モデル準備完了 (%.1fs, vocab=%d+%d)", self.load_seconds, self.base_vocab, self.cfg.reserved_tokens)

    def unload(self) -> None:
        with self.lifecycle_lock:
            self.model = None
            gc.collect()

    @property
    def is_ready(self) -> bool:
        return self.state == STATE_READY and self.model is not None

    # ------------------------------------------------------------------ #
    # 状態
    # ------------------------------------------------------------------ #
    def status(self) -> dict:
        try:
            import torch
            import transformers

            versions = {"torch": torch.__version__, "transformers": transformers.__version__}
        except Exception:
            versions = None

        info = {
            "state": self.state,
            "error": self.error,
            "model_source": self.cfg.model_source,
            "is_local": self.cfg.is_local,
            "quantized": self.quantized,
            "base_vocab": self.base_vocab,
            "reserved_tokens": self.cfg.reserved_tokens,
            "learned_chars": [
                {"char": c, "token_id": m["token_id"], "source": m.get("source", "?")}
                for c, m in sorted(self.store.chars.items(), key=lambda kv: kv[1]["token_id"])
            ],
            "template": self.tmpl.info() if self.tmpl else None,
            "versions": versions,
            "autostart": self.cfg.autostart,
        }
        return info

    # ------------------------------------------------------------------ #
    # 未知文字
    # ------------------------------------------------------------------ #
    def scan_unknown(self, text: str) -> dict | None:
        if self.tokenizer is None:
            return None
        from .vocab import scan_text

        return scan_text(self.tokenizer, text)

    # ------------------------------------------------------------------ #
    # 学習
    # ------------------------------------------------------------------ #
    def learn_instant(self, chars: list[str], examples: list[str] | None = None) -> dict:
        """即時学習（モデルを無停止で書き換える）。"""
        if not self.is_ready:
            raise NotReadyError("モデルが準備できていません")
        if not chars:
            raise ValueError("学習する文字が指定されていません")
        results = []
        with self.gen_lock:
            for ch in chars:
                pieces = None
                try:
                    from .vocab import classify_char

                    rep = classify_char(self.tokenizer, ch)
                    if rep and rep.is_unknown:
                        pieces = rep.pieces
                except Exception:
                    pieces = None
                r = self.learner.learn_instant(self.tokenizer, self.model, ch, pieces=pieces)
                r["status"] = "already" if r.get("already") else "learned"
                results.append(r)
        return {"mode": "instant", "results": results}

    def learn_deep(self, chars: list[str], examples: list[str] | None = None,
                   steps: int | None = None, on_progress=None) -> dict:
        """深学習: チャット用モデルを外して bf16 マスターで埋め込みのみ更新 → 再ロード。"""
        steps = steps or self.cfg.learn_steps
        with self.gen_lock:  # 生成中のモデル差し替えを防ぐ
            with self.lifecycle_lock:
                prev = self.state
                self.state = STATE_LEARNING
                try:
                    self.unload()
                    self.load(quantize=False)  # bf16 マスター（autograd 可能）
                    result = self.learner.train_deep(
                        self.tokenizer, self.model, chars, examples,
                        steps=steps, lr=self.cfg.learn_lr, on_progress=on_progress,
                    )
                    self.unload()
                    self.load(quantize=self.cfg.quantize_int8)  # 学習済み行を含めて再量子化
                    result["mode"] = "deep"
                    self.state = STATE_READY
                    return result
                except Exception as exc:
                    log.exception("深学習に失敗")
                    self.error = f"深学習エラー: {exc}"
                    self.state = STATE_FAILED if prev != STATE_READY else STATE_READY
                    # 復旧を試みる
                    try:
                        self.unload()
                        self.load(quantize=self.cfg.quantize_int8)
                    except Exception:
                        pass
                    raise

    def reset_learned(self) -> dict:
        n = len(self.store.chars)
        with self.gen_lock:
            self.store.clear()
            if self.is_ready:
                self.unload()
                self.load()
        return {"removed": n}

    # ------------------------------------------------------------------ #
    # 会話生成（ストリーミング）
    # ------------------------------------------------------------------ #
    def stream_chat(self, messages: list[dict], *, max_new_tokens: int | None = None,
                    min_new_tokens: int | None = None,
                    temperature: float | None = None, top_k: int | None = None,
                    repetition_penalty: float | None = None,
                    use_template: bool = True, system_prompt: str | None = None):
        """SSE 用のジェネレータ。dict イベントを yield する。

        ここでは blocking で排他する（CPU 2 コアなので同時生成は直列で十分）。
        """
        import torch
        from transformers import TextIteratorStreamer

        if not self.is_ready:
            raise NotReadyError("モデルが準備できていません")
        if not self.gen_lock.acquire(timeout=0.2):
            raise BusyError("別の生成が進行中です")

        try:
            msgs = self._normalize_messages(messages, use_template, system_prompt)
            max_new = int(max_new_tokens or self.cfg.max_new_tokens)
            temp = float(temperature if temperature is not None else self.cfg.temperature)
            top_k = int(top_k or self.cfg.top_k)
            rep = float(repetition_penalty or self.cfg.repetition_penalty)

            # プロンプト予算: 超過時は古い履歴から落とす
            prompt, mode, used = self._build_prompt(msgs, use_template)
            enc = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
            while len(enc["input_ids"][0]) > self.cfg.prompt_budget and len(used) > 1:
                drop = 1 if used[0].get("role") == "system" else 0
                used.pop(drop if drop < len(used) - 1 else 1)
                prompt, mode, used = self._build_prompt(used, use_template)
                enc = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
            input_ids = enc["input_ids"]
            attention_mask = enc["attention_mask"]
            # 学習済み未知文字を予約トークンへ写像（𠮷 → 1意味トークン）
            if self.store.chars:
                mapped = self.learner.map_ids(self.tokenizer, input_ids[0].tolist())
                if mapped != input_ids[0].tolist():
                    input_ids = torch.tensor([mapped], dtype=torch.long)
                    attention_mask = torch.ones_like(input_ids)  # 長さが変わるため再構築
            if mode == "raw" and getattr(self.tokenizer, "bos_token_id", None) is not None:
                bos = self.tokenizer.bos_token_id
                if input_ids[0, 0].item() != bos:
                    import torch as _t

                    input_ids = _t.cat([_t.tensor([[bos]]), input_ids], dim=1)
                    attention_mask = _t.cat([_t.ones((1, 1), dtype=attention_mask.dtype), attention_mask], dim=1)

            in_len = int(input_ids.shape[1])
            yield {
                "type": "start",
                "engine": self.engine_name(),
                "template_mode": mode,
                "prompt_tokens": in_len,
                "max_new_tokens": max_new,
            }

            streamer = TextIteratorStreamer(
                self.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=600
            )
            allowed = self.store.used_ids()
            suppress = _SuppressReserved(self.base_vocab, self.total_vocab, allowed)

            from transformers import LogitsProcessorList

            holder: dict = {}

            def _run():
                try:
                    with torch.inference_mode():
                        holder["out"] = self.model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=max_new,
                            min_new_tokens=int(min_new_tokens) if min_new_tokens else None,
                            do_sample=temp > 1e-4,
                            temperature=max(temp, 1e-4) if temp > 1e-4 else None,
                            top_k=top_k if temp > 1e-4 else None,
                            repetition_penalty=rep,
                            logits_processor=LogitsProcessorList([suppress]),
                            streamer=streamer,
                            pad_token_id=self.tokenizer.pad_token_id
                            or self.tokenizer.eos_token_id,
                        )
                except Exception as exc:  # noqa: BLE001
                    holder["err"] = exc
                    try:
                        streamer.end()
                    except Exception:
                        pass

            th = threading.Thread(target=_run, name="lfm-generate", daemon=True)
            t0 = time.time()
            th.start()

            t_first: float | None = None
            text_out: list[str] = []
            try:
                for piece in streamer:
                    if piece:
                        if t_first is None:
                            t_first = time.time()
                        text_out.append(piece)
                        yield {"type": "delta", "text": piece}
            except Exception as exc:  # queue.Empty など
                yield {"type": "error", "message": f"生成ストリーム中断: {exc}"}
            th.join(timeout=600)

            t_end = time.time()
            if "err" in holder:
                yield {"type": "error", "message": f"生成エラー: {holder['err']}"}
                return
            new_tokens = 0
            if "out" in holder:
                new_tokens = max(0, int(holder["out"].shape[1]) - in_len)
            prefill_s = (t_first - t0) if t_first else None
            decode_s = (t_end - t_first) if (t_first and new_tokens) else None
            tps = (new_tokens / decode_s) if (decode_s and new_tokens) else None
            yield {
                "type": "done",
                "text": "".join(text_out),
                "stats": {
                    "new_tokens": new_tokens,
                    "prefill_seconds": round(prefill_s, 3) if prefill_s else None,
                    "decode_seconds": round(decode_s, 3) if decode_s else None,
                    "tokens_per_second": round(tps, 2) if tps else None,
                    "template_mode": mode,
                    "engine": self.engine_name(),
                },
            }
        finally:
            self.gen_lock.release()

    # ------------------------------------------------------------------ #
    def engine_name(self) -> str:
        base = "LFM2.5-1.2B-JP"
        if self.quantized:
            return f"{base} (int8)"
        return base

    def _normalize_messages(self, messages: list[dict], use_template: bool, system_prompt: str | None) -> list[dict]:
        msgs = [{"role": str(m.get("role", "user")), "content": str(m.get("content", ""))} for m in messages]
        if use_template and system_prompt:
            msgs = [{"role": "system", "content": system_prompt}] + [m for m in msgs if m["role"] != "system"]
        elif use_template and not any(m["role"] == "system" for m in msgs):
            msgs = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}] + msgs
        return msgs

    def _build_prompt(self, msgs: list[dict], use_template: bool):
        prompt, mode = self.tmpl.apply(msgs, use_template=use_template)
        return prompt, mode, msgs

    # ------------------------------------------------------------------ #
    # モデルのホットスワップ(オフライン取り込み用)
    # ------------------------------------------------------------------ #
    def reload(self, source: str | None = None) -> None:
        """モデルソースを変更し、バックグラウンドで再ロードする。

        HuggingFace に接続できない環境では、ブラウザからアップロードした
        ローカルのモデルディレクトリをここで差し替える。
        """
        with self.lifecycle_lock:
            if source:
                self.cfg.model_source = source
            self._started = False
            self.state = STATE_IDLE
            self.error = None
        self.ensure_started()
