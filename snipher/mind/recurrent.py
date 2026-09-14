"""再帰的思考（Recurrent / Draft-Verification）モデル — v8 の頭脳。

ユーザーの設計図そのままの 3 段構成を、モデル自身のパラメータだけで回します。

    ┌──────────────────────────────────────────────────────────────┐
    │ モデルA（ルーティング・思考）                                  │
    │   入力から <think>…</think> の思考（結論の箇条書き）を生成      │
    │        ↓ 思考が「確率の波」として下流の文脈に干渉              │
    │ モデルB（知識・文章化）                                        │
    │   箇条書きをもとに日本語の長文を *テンプレート無しで* 生成      │
    │        ↓ 下書きを検証                                          │
    │ モデルC（校正・フォーマット）                                   │
    │   不要な文言・反復・文体混在・壊れた文末を検査 → 合格/却下       │
    │   却下なら（反復トークンの禁止・低温化という波を乗せて）再下書き │
    └──────────────────────────────────────────────────────────────┘

* 反復は ``sample.py`` の n-gram 禁止（物理的なループ封印）で抑止。
* 文はすべてモデルの確率サンプリングで生まれる。テンプレートも
  プログラムによる組み立て文も一切使わない（``generate_novel`` を参照）。
* A/B/C は別々の重み（``v8_a/b/c.npz``）でも、同一の重みを役割プロンプトで
  使い分ける単一モデルでも動く。無ければ ``None`` で graceful に縮退する。

イベント契約（SSE）は既存の ``thought / delta / done`` に加えて
``draft`` / ``verify`` を流す。UI は同じ契約で描画できます。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..neural.moe_core import MoECore
from ..neural.sample import SamplingControls
from ..neural.tokenizer import ASST, BOS, EOS, PAD, SYS, UNK, USER

# 思考区間のマーカー。既存の <user>/<asst> と同じ文字レベル規約（特殊トークンで
# はなくリテラル文字列）。学習データ（cot.py）も同じ文字列で統一する。
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"

_VOCAB_DIR = Path(__file__).resolve().parents[1] / "data" / "neural"

_DANGLING = ("を", "が", "は", "に", "で", "と", "も", "へ", "の", "や", "から", "まで")
_TEMPLATE_MARKERS = ("の知識で答えます", "としてお答えします", "以下の通りです",
                     "ご質問ありがとうございます", "お役に立てれば幸いです")


@dataclass
class ProbabilityWave:
    """段と段の間で受け渡す「確率の波」。生成中にロジットへ干渉する。"""
    source: str                          # "think" / "verify"
    text: str = ""                       # 可読な注釈（UI 表示用）
    ban: list[int] = field(default_factory=list)      # 禁止トークン ID
    boost: dict[int, float] = field(default_factory=dict)  # 増幅トークン
    temperature: float | None = None     # 次の段の温度
    no_repeat_ngram: int | None = None   # 次の段の反復禁止強度


def _find_repeat(ids: list[int], n: int = 4) -> tuple[list[int], list[int]]:
    """直近ウィンドウで n-gram の反復を探し、(反復している n-gram, その全トークン) を返す。"""
    if len(ids) < 2 * n:
        return [], []
    counts: dict[tuple[int, ...], int] = {}
    for i in range(len(ids) - n + 1):
        g = tuple(ids[i:i + n])
        counts[g] = counts.get(g, 0) + 1
    bad = [list(g) for g, c in counts.items() if c >= 2]
    toks: list[int] = []
    for g in bad:
        toks.extend(g)
    return bad, sorted(set(toks))


class RecurrentMind:
    """再帰的思考ループ。A/B/C の 3 コアを束ねる。"""

    def __init__(self, cores: dict[str, MoECore | None] | None = None, *,
                 max_rounds: int = 3, max_think_chars: int = 120,
                 max_draft_chars: int = 400, default_paths: dict[str, str] | None = None):
        self.max_rounds = max_rounds
        self.max_think_chars = max_think_chars
        self.max_draft_chars = max_draft_chars
        paths = default_paths or {
            "A": str(_VOCAB_DIR / "v8_a.npz"),
            "B": str(_VOCAB_DIR / "v8.npz"),
            "C": str(_VOCAB_DIR / "v8_c.npz"),
        }
        if cores is None:
            loaded: dict[str, MoECore | None] = {}
            for role, p in paths.items():
                try:
                    loaded[role] = MoECore(path=p) if Path(p).exists() else None
                except Exception:  # noqa: BLE001 - 1 個壊れていても他で回す
                    loaded[role] = None
            cores = loaded
        # 足りない役割は他で埋める（単一モデルでも 3 段が回る）
        self.cores: dict[str, MoECore | None] = {}
        for role in ("A", "B", "C"):
            self.cores[role] = cores.get(role) or cores.get("B") or cores.get("A") or cores.get("C")

    # ------------------------------------------------------------------ #
    @property
    def available(self) -> bool:
        return bool(self.cores.get("B") and self.cores["B"].is_ready)

    def status(self) -> dict:
        out = {"kind": "recurrent-v8", "state": "ready" if self.available else "unavailable",
               "max_rounds": self.max_rounds}
        for role in ("A", "B", "C"):
            c = self.cores.get(role)
            out[f"core_{role}"] = c.status() if c and c.is_ready else None
        return out

    # ------------------------------------------------------------------ #
    # モデルA: 思考（<think>…</think> の箇条書き）
    # ------------------------------------------------------------------ #
    def think(self, user_text: str, *, seed: int | None = None) -> dict:
        core = self.cores["A"]
        if core is None or not core.is_ready:
            return {"text": "", "ids": [], "closed": False}
        tok = core.tok
        head = [BOS] + tok.encode(f"<user>{user_text}\n<asst>{THINK_OPEN}")
        ctrl = SamplingControls(temperature=0.6, top_k=30, top_p=0.95,
                                no_repeat_ngram=4, repetition_penalty=1.1)
        # EOS は禁止（break）ではなくマスク（ban）して、思考を最後まで書かせる
        ctrl.ban = set(ctrl.ban) | {EOS}
        ids = core.generate_ids(head, max_new=self.max_think_chars, controls=ctrl,
                                seed=seed, forbid=(PAD, UNK, USER, SYS),
                                stop_text=THINK_CLOSE, stop_on_sentence=False)
        text = tok.decode(ids)
        closed = THINK_CLOSE in text
        if closed:
            text = text.split(THINK_CLOSE, 1)[0]
        return {"text": text.strip(), "ids": ids, "closed": closed}

    # ------------------------------------------------------------------ #
    # モデルB: 文章化（テンプレート無し・純粋な確率サンプリング）
    # ------------------------------------------------------------------ #
    def draft(self, user_text: str, outline: str | list[int] | None = None, *,
              wave: ProbabilityWave | None = None, max_chars: int | None = None,
              seed: int | None = None) -> dict:
        core = self.cores["B"]
        if core is None or not core.is_ready:
            return {"text": "", "ids": []}
        tok = core.tok
        head = [BOS] + tok.encode(f"<user>{user_text}\n<asst>")
        # 思考は「文字列」で受け渡す（A と B の語彙が違っても壊れない）。
        if outline is not None:
            outline_text = outline if isinstance(outline, str) else tok.decode(list(outline))
            if outline_text:
                head += tok.encode(THINK_OPEN) + tok.encode(outline_text) + tok.encode(THINK_CLOSE + "\n")
        w = wave or ProbabilityWave(source="think")
        ctrl = SamplingControls(
            temperature=w.temperature if w.temperature is not None else 0.8,
            top_k=40, top_p=0.95, repetition_penalty=1.08,
            no_repeat_ngram=w.no_repeat_ngram if w.no_repeat_ngram is not None else 4,
            boost=w.boost or None, ban=w.ban or None)
        ids = core.generate_ids(head, max_new=max_chars or self.max_draft_chars, controls=ctrl,
                                seed=seed, forbid=(PAD, UNK, USER, SYS, ASST))
        text = tok.decode(ids).strip()
        # 生成がはみ出して思考マーカーを出した場合は取り除く（校正段の仕事）
        text = text.replace(THINK_OPEN, "").replace(THINK_CLOSE, "")
        return {"text": text, "ids": ids}

    # ------------------------------------------------------------------ #
    # モデルC: 校正・検証（モデル自身の perplexity + 日本語の構造検査）
    # ------------------------------------------------------------------ #
    def verify(self, text: str) -> dict:
        core = self.cores["C"]
        reasons: list[str] = []
        if not text:
            return {"accepted": False, "reasons": ["空"], "score": None, "wave": None}

        # 1) モデル自身の見立て（パラメータによる採点。テンプレートではない）
        score = None
        if core is not None and core.is_ready:
            score = core.score(text)

        # 2) 日本語の構造検査（校正段の規則。生成の母体ではない）
        from ..composer import validate
        ok, reason = validate(text, min_len=1)
        if not ok:
            reasons.append(f"構造: {reason}")

        # 3) 反復の検査
        ids = core.tok.encode(text) if core and core.is_ready else list(text)
        bad, toks = _find_repeat(ids, n=4)
        if bad:
            reasons.append(f"反復: {len(bad)} 個の 4-gram が周回")

        # 4) 文体混在の検査
        if ("です" in text or "ます" in text) and ("だ。" in text or "た。" in text):
            reasons.append("文体: です・ます と だ・た が混在")

        # 5) 定型表現の検査（「〜の知識で答えます」等のループを物理的に禁止）
        for m in _TEMPLATE_MARKERS:
            if m in text:
                reasons.append(f"定型表現: {m}")

        accepted = not reasons
        wave = None
        if not accepted:
            wave = ProbabilityWave(
                source="verify", text=" / ".join(reasons),
                ban=list(toks), temperature=0.65,
                no_repeat_ngram=max(4, (len(bad[0]) if bad else 4)))
        return {"accepted": accepted, "reasons": reasons, "score": score, "wave": wave}

    # ------------------------------------------------------------------ #
    # 全体: 思考 → 下書き → 検証（→ 却下なら再下書き）
    # ------------------------------------------------------------------ #
    def respond(self, user_text: str, *, max_chars: int | None = None, seed: int | None = None) -> dict:
        if not self.available:
            return {"text": "", "thought": "", "rounds": 0, "accepted": False,
                    "reasons": ["v8 コア未ロード"], "route": "recurrent", "waves": []}
        thought = self.think(user_text, seed=seed)
        wave_a = ProbabilityWave(source="think", text="思考を文脈へ干渉")
        draft = self.draft(user_text, thought["text"], wave=wave_a,
                           max_chars=max_chars, seed=seed)
        verdict = self.verify(draft["text"])
        rounds = 1
        waves = [{"source": "think", "text": thought["text"][:80]}]
        while not verdict["accepted"] and rounds < self.max_rounds and self.available:
            draft = self.draft(user_text, thought["text"], wave=verdict["wave"],
                               max_chars=max_chars, seed=None if seed is None else seed + rounds)
            verdict = self.verify(draft["text"])
            rounds += 1
            if verdict["wave"] is not None:
                waves.append({"source": "verify", "text": verdict["wave"].text})
        return {"text": draft["text"], "thought": thought["text"], "rounds": rounds,
                "accepted": verdict["accepted"], "reasons": verdict["reasons"],
                "score": verdict["score"], "route": "recurrent", "waves": waves}

    def respond_stream(self, user_text: str, *, max_chars: int | None = None,
                       seed: int | None = None):
        """SSE 契約（start / thought / draft / verify / delta / done）。"""
        if not self.available:
            yield {"type": "start", "engine": "Snipher recurrent-v8 (unavailable)"}
            yield {"type": "done", "text": "", "stats": {"route": "recurrent", "ready": False}}
            return
        yield {"type": "start", "engine": "Snipher recurrent-v8"}
        thought = self.think(user_text, seed=seed)
        yield {"type": "thought", "state": "start"}
        yield {"type": "thought", "state": "done", "text": thought["text"]}
        wave_a = ProbabilityWave(source="think", text="思考を文脈へ干渉")
        draft = self.draft(user_text, thought["text"], wave=wave_a, max_chars=max_chars, seed=seed)
        yield {"type": "draft", "round": 1, "text": draft["text"]}
        verdict = self.verify(draft["text"])
        rounds = 1
        while not verdict["accepted"] and rounds < self.max_rounds and self.available:
            draft = self.draft(user_text, thought["text"], wave=verdict["wave"],
                               max_chars=max_chars, seed=None if seed is None else seed + rounds)
            rounds += 1
            yield {"type": "draft", "round": rounds, "text": draft["text"]}
            verdict = self.verify(draft["text"])
        yield {"type": "verify", "accepted": verdict["accepted"],
               "reasons": verdict["reasons"], "rounds": rounds}
        yield {"type": "delta", "text": draft["text"]}
        yield {"type": "done", "text": draft["text"],
               "stats": {"route": "recurrent", "rounds": rounds,
                         "accepted": verdict["accepted"],
                         "thought_chars": len(thought["text"]),
                         "score": verdict["score"],
                         "waves": {"think": True, "verify": rounds > 1}}}

    # ------------------------------------------------------------------ #
    # テンプレート無しの長文生成（小説・説明文） — 完全にパラメータだけで書く
    # ------------------------------------------------------------------ #
    def generate_novel(self, prompt: str, *, max_chars: int = 2000,
                       temperature: float = 0.9, seed: int | None = None,
                       with_think: bool = True) -> str:
        """2000 文字級の文を、テンプレートもプログラム組み立ても一切使わず
        モデルの確率サンプリングだけで生成する。"""
        core = self.cores["B"]
        if core is None or not core.is_ready:
            return ""
        tok = core.tok
        ids: list[int] = [BOS] + tok.encode(f"<user>{prompt}\n<asst>")
        if with_think and self.cores.get("A") and self.cores["A"].is_ready:
            th = self.think(prompt, seed=seed)
            if th["text"]:
                ids += (tok.encode(THINK_OPEN) + tok.encode(th["text"])
                        + tok.encode(THINK_CLOSE + "\n"))
        ctrl = SamplingControls(temperature=temperature, top_k=50, top_p=0.96,
                                repetition_penalty=1.1, no_repeat_ngram=5,
                                no_repeat_window=4096)
        # 終端・役割・未知トークン、および思考マーカーの断片（< > /）は logit 段階で
        # マスク（ban）して「書き続けさせる」。反復は no_repeat_ngram=5 が物理的に
        # 封じるので、同じ語列の周回にはならない。
        ctrl.ban = set(ctrl.ban) | {PAD, UNK, USER, SYS, ASST, EOS}
        for ch in ("<", ">", "/"):
            if ch in tok.stoi:
                ctrl.ban.add(tok.stoi[ch])
        chunks: list[int] = []
        total_chars = 0
        stall = 0
        while total_chars < max_chars and stall < 8:
            chunk = core.generate_ids(ids + chunks, max_new=200, controls=ctrl,
                                      seed=None if seed is None else seed + len(chunks),
                                      forbid=(PAD, UNK, USER, SYS, ASST),
                                      ctx=192, stop_on_sentence=False,
                                      history=chunks)
            if not chunk:
                # 反復封印で候補が尽きた等 → 温度を上げて再挑戦（無限ループはしない）
                stall += 1
                ctrl.temperature = min(1.5, ctrl.temperature + 0.1)
                continue
            stall = 0
            chunks.extend(chunk)
            total_chars = len(tok.decode(chunks))
        text = tok.decode(chunks)
        # 制御マーカーの残骸を除去（校正段の後始末。文の組み立てには使わない）。
        # 完全マーカーに加え、崩れた断片（< / > が残る）も落として漏れを防ぐ。
        for m in (THINK_OPEN, THINK_CLOSE, "<user>", "<asst>", "<sys>"):
            text = text.replace(m, "")
        text = text.replace("<", "").replace(">", "")
        return text.strip()
