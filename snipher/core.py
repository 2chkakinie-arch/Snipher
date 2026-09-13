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
    │  ニューラルコアは環境に応じて 3 段構えで自動選択（ユーザー設定ゼロ）:        │
    │    T1  ローカル フルウェイト … llama.cpp(GGUF ~731MB) / torch(INT8) を自動取得│
    │    T2  リモート委譲          … SNIPHER_LFM_REMOTE_URL の常駐ホストに生成を依頼│
    │    T3  内蔵蒸留コア          … LFM2.5 と同じアーキテクチャを蒸留した           │
    │                                NumPy スナップショット（同梱・ミリ秒ロード）     │
    │  T3 は重み同梱なので Vercel 等のサーバーレスでもダウンロード不要で動く。        │
    │  知識は `snipher/knowledge.py`（BM25 検索）で補い、どんな話題でも事実を引ける。 │
    │                                                                          │
    │  モデルは起動時に全自動取得(レジューム対応・マルチソース)。                  │
    │  バックエンドは llama.cpp(GGUF・最速) / torch(INT8) を自動選択。            │
    └──────────────────────────────────────────────────────────────────────────┘

ニューラルコアが使えない環境（取得失敗・依存なし・低リソース）では、
高速コアだけで常に応答を返す（品質の下限を守る）。
"""

from __future__ import annotations

import logging
import math
import os
import re
import threading
import time
from pathlib import Path

from .knowledge import KnowledgeBase
from .lfm.assist import AssistConfig, HybridAssist
from .lfm.config import DEFAULT_GGUF_QUANT, LfmConfig, is_serverless
from .polisher import Polisher
from .research import ResearchEngine
from .tasks import TaskRouter

log = logging.getLogger(__name__)

# 確実な定形応答（高速コアが即答する意図）。それ以外＝確率的に不安 → ニューラルコア。
CANNED_INTENTS = {
    "greeting", "how_are_you", "identity", "capability", "thanks", "apology",
    "farewell", "agreement", "disagreement", "praise",
}

ROUTE_INSTANT = "instant"     # ⚡ 高速コアが即答（数ミリ秒）
ROUTE_KNOWLEDGE = "knowledge"  # 📚 知識ベース検索＋補足生成
ROUTE_LIGHT = "light"          # ✨ 内蔵ニューラルコア（LFM2.5 蒸留）が生成
ROUTE_NEURAL = "neural"        # ✨ LFM2.5 が生成（フルウェイト / リモート）
ROUTE_FALLBACK = "fallback"    # 高速コアのみ（ニューラル未就绪）


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
        # ---- 軽量ニューラルコア（LFM2.5 の蒸留スナップショット）/ 知識ベース ---- #
        self._light = None
        self._light_state = "unchecked"        # unchecked|ready|absent|off|error
        self._remote = None
        self.kb = KnowledgeBase.shared()
        # ---- 道具層（厳密計算・コード・現在情報） -------------------------- #
        # ResearchEngine はネットワークを行わず、必要判定後にだけ fetch する。
        self.research = ResearchEngine()
        self.tasks = TaskRouter(self.research)
        # ---- 文章生成（composer）と流暢さの審判（n-gram LM） ---- #
        self._composer = None
        self._lm = None
        self._lm_state = "unchecked"           # unchecked|ready|absent|off|error

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
            self._composer = None
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
        """現在アクティブ（ready）な「重い」ニューラルバックエンドを返す。

        優先順位: ローカル llama.cpp(GGUF) → ローカル torch → リモート委譲。
        どれも無ければ None（→ 内蔵蒸留コア／ルール経路が担う）。
        """
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
        # T2: リモートのフルウェイト（設定済みで疎通できる場合のみ）
        remote = self.remote_backend()
        if remote is not None and remote.is_ready:
            return remote
        return None

    def neural_available(self) -> bool:
        """LFM2.5 フルウェイト（ローカル or リモート）が使えるか。

        内蔵蒸留コア（T3）は含めない — 高速経路の判定は「重い生成を挟むか否か」で
        決まるため、既存の経路契約（instant / neural / fallback）を崩さない。
        """
        return self.active_backend() is not None

    # ------------------------------------------------------------------ #
    # T2: リモート委譲 / T3: 内蔵蒸留コア
    # ------------------------------------------------------------------ #
    def remote_backend(self):
        """常駐ホストの LFM2.5 に生成を委譲するバックエンド（未設定なら None）。"""
        if not self.cfg.remote_url:
            return None
        if self._remote is None:
            from .lfm.remote_backend import RemoteLfmBackend

            self._remote = RemoteLfmBackend(self.cfg.remote_url, token=self.cfg.remote_token)
        return self._remote

    def light_core(self):
        """内蔵ニューラルコア（LFM2.5 を蒸留した NumPy スナップショット）。

        重みはパッケージ同梱なのでダウンロード不要・初回のみ数十ミリ秒でロードし、
        以降はメモリ上の行列をそのまま使う（サーバーレスでも即動）。
        """
        want = (self.cfg.light_core or "auto").lower()
        if want == "off":
            self._light_state = "off"
            return None
        if self._light_state == "ready":
            return self._light
        if self._light_state in ("absent", "error") and want != "on":
            return None
        if self._light is not None and self._light_state == "unchecked":
            return self._light
        from .neural.cache import get_core

        core = get_core()
        if core is None:
            self._light_state = "absent" if want != "on" else "error"
            self._light = None
            return None
        self._light = core
        self._light_state = "ready"
        log.info("内蔵ニューラルコアを有効化: %s", core.engine_name())
        return core

    def light_ready(self) -> bool:
        return self.light_core() is not None

    def disable_light(self) -> None:
        """テスト/低速環境用に内蔵ニューラルコアを無効化する。"""
        self._light = None
        self._light_state = "off"
        self.cfg.light_core = "off"

    def any_neural(self) -> bool:
        return self.active_backend() is not None or self.light_ready()

    # ------------------------------------------------------------------ #
    # 文章生成（composer）と流暢さの審判（巨大 n-gram LM）
    # ------------------------------------------------------------------ #
    def lm(self):
        """文字 n-gram 言語モデル（snipher/data/lm.npz）。無ければ None。

        役割は生成ではなく **判定**: 候補文の perplexity を測って、日本語として
        壊れている文を捨てる／自然な文を選ぶ。数十万〜数百万エントリを持ち、
        1 文の判定は 1ms 未満。ダウンロード不要でリポジトリに同梱。
        """
        want = (os.environ.get("SNIPHER_LM", "auto") or "auto").lower()
        if want == "off":
            self._lm_state = "off"
            return None
        if self._lm_state == "ready":
            return self._lm
        if self._lm_state in ("absent", "error") and want != "on":
            return None
        try:
            from . import lm as lm_mod
            model = lm_mod.shared()
        except Exception:  # noqa: BLE001
            # numpy が無い最小環境でも、composer と道具層は動かし続ける。
            model = None
        if model is None or not model.is_ready:
            self._lm_state = "absent" if want != "on" else "error"
            self._lm = None
            return None
        self._lm = model
        self._lm_state = "ready"
        log.info("n-gram 言語モデルを有効化: %s", model.engine_name())
        return model

    def lm_ready(self) -> bool:
        return self.lm() is not None

    def composer(self):
        """発話から日本語の応答を組み立てる composer（知識ベース + 文法 + LM）。"""
        if self._composer is None:
            from .composer import Composer

            self._composer = Composer(kb=self.kb, polisher=self.polisher, lm=self.lm(),
                                      task_router=self.tasks)
        return self._composer

    def judge(self, text: str, core=None, lm=None) -> dict:
        """候補文を (内蔵ニューラルコア + n-gram LM) の両方で採点する。

        どちらか片方でも自信を持てない文は通さない（幾何平均）。
        LM が「壊れた文字」を検出したら、さらに減点する。
        """
        text = str(text or "").strip()
        if not text:
            return {"confidence": 0.0, "perplexity": None, "neural": None, "lm": None,
                    "lm_bad_ratio": None}
        neural_conf: float | None = None
        ppl: float | None = None
        if core is not None:
            try:
                sc = core.score(text)
                neural_conf = float(sc.get("confidence") or 0.0)
                ppl = sc.get("perplexity")
            except Exception:  # noqa: BLE001
                neural_conf = None
        lm_conf: float | None = None
        bad: float | None = None
        if lm is None:
            lm = self.lm()
        if lm is not None:
            try:
                ls = lm.score(text)
                lm_conf = float(ls["confidence"])
                bad = float(ls["bad_ratio"])
                if ppl is None:
                    ppl = ls["perplexity"]
            except Exception:  # noqa: BLE001
                lm_conf = None
        parts = [c for c in (neural_conf, lm_conf) if c is not None]
        if not parts:
            conf = 0.0
        elif len(parts) == 1:
            conf = parts[0] * 0.9            # 片方だけの判定は少し割り引く
        else:
            conf = math.sqrt(max(0.0, parts[0]) * max(0.0, parts[1]))
        if bad is not None and bad > 0.12:
            conf *= max(0.2, 1.0 - bad)
        return {"confidence": round(conf, 4), "perplexity": ppl, "neural": neural_conf,
                "lm": lm_conf, "lm_bad_ratio": bad}

    # ---- 知識ベースの文の 4-gram 転置索引（幻覚の検出に使う・1 回だけ作る） ---- #
    _kb_grams: tuple[dict, list] | None = None

    def _kb_gram_index(self) -> tuple[dict, list] | None:
        if self._kb_grams is not None:
            return self._kb_grams
        if self.kb is None:
            return None
        try:
            index: dict[str, list[int]] = {}
            topics: list[str] = []
            for item in self.kb.items:
                tp = str(item.get("topic") or "")
                sents: list[str] = []
                for key in ("def", "opinion"):
                    v = item.get(key)
                    if isinstance(v, str) and v.strip():
                        sents.append(v.strip())
                for key in ("facts", "why", "how", "tips", "answers", "followups"):
                    for v in item.get(key) or []:
                        if isinstance(v, str) and v.strip():
                            sents.append(v.strip())
                for sent in sents:
                    sid = len(topics)
                    topics.append(tp)
                    for i in range(len(sent) - 3):
                        index.setdefault(sent[i:i + 4], []).append(sid)
            self._kb_grams = (index, topics)
        except Exception:  # noqa: BLE001
            log.debug("知識ベースの 4-gram 索引を作れません", exc_info=True)
            self._kb_grams = None
        return self._kb_grams

    def recited_topic(self, text: str, *, ratio: float = 0.6) -> str:
        """生成文が知識ベースの文の丸書きなら、その話題名を返す（""= 丸書きではない）。

        小さなニューラルコアは、知らない話題を聞かれると **覚えている別の知識** を
        語り出すことがあります。それを「それっぽい返事」として出してしまわないための検査です。
        """
        idx = self._kb_gram_index()
        t = str(text or "").strip()
        if idx is None or len(t) < 8:
            return ""
        grams = {t[i:i + 4] for i in range(len(t) - 3)}
        if not grams:
            return ""
        gram_index, topics = idx
        counts: dict[int, int] = {}
        for g in grams:
            for sid in gram_index.get(g, ()):
                counts[sid] = counts.get(sid, 0) + 1
        best_sid, best = -1, 0.0
        for sid, n in counts.items():
            r = n / len(grams)
            if r > best:
                best, best_sid = r, sid
        return topics[best_sid] if best >= ratio and best_sid >= 0 else ""

    # 会話の受け答えに必ず現れる語（一人称・二人称・依頼）。
    # これが無い生成文は「知識の断片」であって、返事ではありません。
    _DIALOGUE_MARKERS = ("私", "あなた", "返事", "言葉", "話", "続き", "教えて",
                         "聞かせて", "どうぞ", "ください", "ましょう", "一緒に")
    _QUESTION_END = re.compile(
        r"(ますか|ですか|ましょうか|でしょうか|ませんか|たいですか|くれますか|"
        r"か。|か？|か！|？|\?)\s*$")
    # 話題として数えない語（形式名詞・一般的な動詞）
    _NOT_TOPIC_WORDS = ("こと", "もの", "とき", "ある", "いる", "する", "なる", "いう")

    def conversational(self, text: str, user_text: str) -> bool:
        """生成文が「会話の受け答え」になっているか。

        話題の語が無い発話（「何してるの」「話して」）への返事は、語の重なりだけでは
        判定できません。次のいずれかを満たすものだけを通します:

        A. 相手の内容語を一つでも返している（こと・もの等の形式名詞は数えない）
        B. 問いかけで終わっていて、かつ会話の語（話・ください・ましょう…）を含む
        C. 会話の語を 2 つ以上含む（私・あなた・返事・言葉 …）

        「例外のとき量があることが多いです。」のような、文法的でも中身の無い文は
        A/B/C のどれも満たさないので落ちます。
        """
        t = str(text or "").strip()
        u = str(user_text or "").strip()
        if not t:
            return False
        try:
            from .composer import _ECHO_SKIP
            from .knowledge import GENERIC_ALIASES, content_words

            index = self.kb.index if self.kb is not None else None
            skip = set(_ECHO_SKIP) | set(GENERIC_ALIASES) | set(self._NOT_TOPIC_WORDS)
            a = {w for w in content_words(t, index) if w not in skip and len(w) >= 2}
            b = {w for w in content_words(u, index) if w not in skip and len(w) >= 2}
            if a & b:                                     # A
                return True
        except Exception:  # noqa: BLE001
            log.debug("conversational の語比較に失敗", exc_info=True)
        marks = sum(1 for m in self._DIALOGUE_MARKERS if m in t)
        if marks >= 2:                                    # C
            return True
        return bool(self._QUESTION_END.search(t)) and marks >= 1   # B

    def redundant(self, extra: str, text: str) -> bool:
        """付け足す一文が、すでに出した文と同じことを言っていないか。"""
        e, t = str(extra or "").strip(), str(text or "").strip()
        if not e or e in t:
            return True
        for i in range(max(0, len(e) - 7)):       # 8 文字以上の共通部分列
            if e[i:i + 8] and e[i:i + 8] in t:
                return True
        try:
            from .knowledge import content_words

            index = self.kb.index if self.kb is not None else None
            a = set(content_words(e, index))
            b = set(content_words(t, index))
            if a and len(a & b) / len(a) >= 0.5:
                return True
        except Exception:  # noqa: BLE001
            pass
        return False

    def relevant(self, text: str, user_text: str, *, topic: str = "") -> bool:
        """生成文が発話と話題を共有しているか（それっぽいだけの雑談を弾く）。

        判定は 3 段:

        1. 発話の話題語（知識ベースが知っている語）と生成文が語を共有 → 関連あり
        2. 探している話題名が生成文に出ている → 関連あり
        3. 発話に話題語が無い（「何してるの」のような世間話）→ 語の重なりでは
           判定できないので、**別の話題の知識の丸書きでなければ** 通す
        """
        try:
            from .composer import _ECHO_SKIP
            from .knowledge import content_words

            index = self.kb.index if self.kb is not None else None
            a = set(content_words(str(text or ""), index))
            b = {w for w in content_words(str(user_text or ""), index)
                 if w not in _ECHO_SKIP and (index is None or index.topics_of(w))}
            if a & b:
                return True
            if topic:
                tw = set(content_words(str(topic), index)) | {str(topic)}
                if tw & a or str(topic) in str(text or ""):
                    return True
            if not b:
                # 発話に話題の語が無い（世間話）→ 別の話題の知識の丸書きではなく、
                # かつ「会話の受け答え」の形をしているものだけ通す
                recited = self.recited_topic(text)
                if recited and recited != str(topic or ""):
                    return False
                return self.conversational(text, user_text)
            return False
        except Exception:  # noqa: BLE001
            # 判定できないなら採用しない（幻覚の門は「閉じる側」に倒す）
            log.debug("relevant の判定に失敗", exc_info=True)
            return False

    def tiers(self) -> dict:
        """どの知能階層が生きているか（/api/status・/info 用）。"""
        heavy = None
        if self._gguf is not None and self._gguf.is_ready:
            heavy = "gguf"
        elif self.torch_engine() is not None and getattr(self.torch_engine(), "is_ready", False):
            heavy = "torch"
        remote = self.remote_backend()
        return {
            "local_full": heavy,
            "remote": {"configured": bool(self.cfg.remote_url),
                       "alive": bool(remote and remote.is_ready)} if remote or self.cfg.remote_url else None,
            "distilled": self._light_state if self._light_state != "unchecked" else (
                "ready" if self.light_ready() else self._light_state),
            "knowledge_base": self.kb.stats() if self.kb is not None else None,
            "language_model": (lambda m: ({"state": "ready", "params": m.n_params(),
                                          "order": m.order, "bytes": m.bytes_on_disk()}
                                         if m is not None else {"state": self._lm_state,
                                                                "params": 0}))(self.lm()),
            "composer": "ready",
            "serverless": is_serverless(),
        }

    # ------------------------------------------------------------------ #
    # 知識ベース（BM25）
    # ------------------------------------------------------------------ #
    def kb_answer(self, text: str) -> dict | None:
        """発話に合う事実があれば返す（検索は 0.1 ミリ秒オーダー・常時使用可）。"""
        if self.kb is None:
            return None
        try:
            return self.kb.answer(text)
        except Exception:  # noqa: BLE001
            log.debug("知識ベース検索に失敗", exc_info=True)
            return None

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
        if is_serverless() and not self.cfg.auto_fetch and pending is None:
            # Vercel 等: 731MB の取得は無意味。内蔵蒸留コア／リモート委譲が知能を担う。
            self.boot_state = "skipped_serverless"
            self.boot_error = None
            log.info("サーバーレス環境のためフルウェイトの自動取得をスキップします"
                     "（内蔵蒸留コア + 知識ベースで応答）")
            return
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
        st["tiers"] = self.tiers()
        light = self.light_core()
        st["light"] = light.status() if light is not None else {
            "state": self._light_state, "kind": "distilled", "params": 0}
        st["light_backend"] = st["light"].get("engine")
        st["light_ready"] = light is not None
        lm = self.lm()
        st["lm"] = lm.status() if lm is not None else {"state": self._lm_state, "kind": "ngram",
                                                       "params": 0}
        st["lm_ready"] = lm is not None
        st["knowledge"] = self.kb.stats() if self.kb is not None else None
        st["research"] = self.research.status()
        st["neural_ready_any"] = self.any_neural()
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
    def route_of(self, draft: dict, mode: str, user_text: str = "",
                 web: bool | None = None) -> str:
        """auto モードの経路判定。

            厳密タスク（計算・コード・検索） → instant（道具層の確定結果）
            確実な定形                     → instant（数ミリ秒・ニューラル不使用）
            曖昧/自由応答 + 重いコア         → neural（LFM2.5 フルウェイト or リモート）
            曖昧/自由応答 + 蒸留コア         → light / knowledge（内蔵ニューラルコア）
            ニューラル一切なし              → fallback（高速コアのみ）

        ``user_text`` は後方互換で任意。タスク判定は入力そのものを見ないと
        できないため、チャット経路からだけ渡す。
        """
        # 仕事の結果を小さな生成モデルで上書きすると、正解が壊れる。
        # これは mode=lfm の強制指定よりも優先する（計算結果を確率生成で
        # 書き換えないための安全規則）。
        if user_text:
            try:
                if self.tasks.classify(user_text, web=web) is not None:
                    return ROUTE_INSTANT
            except Exception:  # noqa: BLE001
                pass
        if mode == "fast":
            return ROUTE_INSTANT
        if mode == "light":
            return self._light_route(draft, force=True)
        if mode in ("lfm", "neural"):
            # 強制指定でも、重いコアが無ければ内蔵蒸留コアが同じ顔をして応答する
            return ROUTE_NEURAL if self.neural_available() else self._light_route(draft, force=True)
        if not self.neural_available():
            return self._light_route(draft, force=False)
        intent = str(draft.get("intent") or "")
        conf = float(draft.get("confidence", 1.0))
        if intent not in CANNED_INTENTS or conf < self.assist.cfg.threshold:
            return ROUTE_NEURAL
        return ROUTE_INSTANT

    def _light_route(self, draft: dict, *, force: bool) -> str:
        """重いコアが居ないときの代替経路（内蔵蒸留コア + 知識ベース）。"""
        if not self.light_ready():
            return ROUTE_FALLBACK      # 重みが無い環境は従来のルール契約を維持
        intent = str(draft.get("intent") or "")
        conf = float(draft.get("confidence", 1.0))
        if intent in CANNED_INTENTS and conf >= self.assist.cfg.threshold and not force:
            return ROUTE_INSTANT          # 定形応答はニューラルを使わず即答（速度維持）
        return ROUTE_LIGHT

    def stream_reply(self, messages: list[dict], *, mode: str = "auto",
                     max_new_tokens: int | None = None, temperature: float | None = None,
                     top_k: int | None = None, repetition_penalty: float | None = None,
                     use_template: bool = True, system_prompt: str | None = None,
                     web: bool | None = None):
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

        # 2) 指示（プロンプト）の実行 — 下書きより前に行う
        #    「JSON 形式のみで出力」「3 つの箇条書きで要約」「関数を書いて」は
        #    *仕事* であって会話ではありません。ここで指示部と材料部を分けて読み、
        #    実行 → 検証まで済んだ結果をそのまま返します（定型文・読み上げは出さない）。
        t0 = time.time()
        if mode != "neural":
            try:
                events = self._instruction_reply(messages, web=web, t0=t0)
            except Exception:  # noqa: BLE001
                log.debug("指示層が失敗（通常経路へ続行）", exc_info=True)
                events = None
            if events:
                yield from events
                return

        # 3) 高速コアの下書き（数ミリ秒）
        draft = self.assist.draft(last_user)
        route = self.route_of(draft, mode, user_text=last_user, web=web)

        if route == ROUTE_NEURAL:
            yield from self._neural_reply(messages, draft, mode,
                                          max_new_tokens=max_new_tokens,
                                          temperature=temperature, top_k=top_k,
                                          repetition_penalty=repetition_penalty,
                                          use_template=use_template,
                                          system_prompt=system_prompt)
            return

        if route in (ROUTE_LIGHT, ROUTE_KNOWLEDGE):
            yield from self._light_reply(messages, draft, route, mode=mode, web=web)
            return

        # 4) instant / fallback: ニューラルを 1 トークンも使わず、その場で文を組み立てる
        yield from self._fast_reply(messages, draft, route, t0=t0, web=web)

    def _instruction_reply(self, messages: list[dict], *, web: bool | None = None,
                           t0: float | None = None) -> list[dict] | None:
        """📋 指示経路: 指示文を *仕事として実行* し、検証を通した結果だけを返す。

        指示でなければ None（呼び出し側が通常経路へ進みます）。返すイベント列は
        他の経路と同じ形（assist / start / delta / done）なので、UI 側は変更不要です。
        """
        t0 = time.time() if t0 is None else t0
        last_user = last_user_of(messages)
        if not last_user or len(last_user.strip()) < 4:
            return None
        try:
            from .instruction import run as _run_instruction

            res = _run_instruction(last_user, kb=self.kb, web=self._instruction_web(web),
                                   history=messages, lm=self.lm(), core=self.light_core(),
                                   turn=len([m for m in messages if m.get("role") == "user"]))
        except Exception:  # noqa: BLE001
            log.debug("指示層の実行に失敗", exc_info=True)
            return None
        if res is None or not str(res.text or "").strip():
            return None
        text = str(res.text)
        knowledge: dict = {
            "via": "instruction", "task": res.task, "topic": (res.meta or {}).get("topic") or None,
            "chars": (res.meta or {}).get("chars"), "signals": (res.meta or {}).get("signals") or [],
            "checks": [f"{c['name']}={'ok' if c['ok'] else 'ng'}" for c in res.checks[:8]],
            "attempts": res.attempts, "authoritative": bool(res.authoritative),
            "coverage": (res.meta or {}).get("coverage"),
        }
        srcs = [s for s in ((res.meta or {}).get("sources") or []) if isinstance(s, dict)][:4]
        if srcs:
            knowledge["sources"] = [{"title": str(s.get("title") or s.get("url"))[:80],
                                     "url": str(s.get("url") or "")} for s in srcs]
        if res.task == "code":
            knowledge["code"] = {k: (res.meta or {}).get(k)
                                 for k in ("language", "ran", "ok", "notes", "function")
                                 if (res.meta or {}).get(k) is not None}
        lm_info = None
        lm = self.lm()
        if lm is not None:
            try:
                sc = lm.score(text)
                lm_info = {"confidence": sc["confidence"], "perplexity": sc["perplexity"],
                           "bad_ratio": sc["bad_ratio"]}
            except Exception:  # noqa: BLE001
                lm_info = None
        stats = {
            "engine": f"Snipher instruction ({res.task})",
            "template_mode": "instruction", "new_tokens": None, "tokens_per_second": None,
            "assist": "instruction", "draft": text[:120], "draft_confidence": res.confidence,
            "draft_seconds": round(time.time() - t0, 5), "intent": "instruction",
            "fixes": ["instruction"], "route": ROUTE_INSTANT, "plan": res.plan,
            "confidence": round(float(res.confidence), 4), "source": "instruction",
            "task": {"kind": res.task, "verified": bool(res.ok), "checks": res.checks[:8],
                     "directive": (res.directive.as_dict() if res.directive is not None else {})},
            "seconds": round(time.time() - t0, 4), "knowledge": knowledge,
        }
        if lm_info:
            stats["lm"] = lm_info
        _srcs = _extract_sources(knowledge)
        if _srcs:
            stats["sources"] = _srcs
        if not res.ok:
            stats["note"] = "指示の検証で仕様を満たせない欄が残りました（checks を参照）"
        return [
            {"type": "assist", "mode": "instruction", "confidence": res.confidence,
             "route": ROUTE_INSTANT, "plan": res.plan, "reason": "instruction_result",
             "task": res.task},
            {"type": "start", "engine": stats["engine"], "template_mode": "instruction"},
            *[{"type": "delta", "text": piece} for piece in _chunk_for_stream(text)],
            {"type": "done", "text": text, "stats": stats},
        ]

    def _instruction_web(self, web: bool | None = None):
        """指示経路で使う Web 裏取りの窓（無効指定なら None = 検索に出ない）。"""
        if web is False:
            return None
        try:
            composer = self.composer()
            grounding = composer.web_grounding() if composer is not None else None
        except Exception:  # noqa: BLE001
            return None
        if grounding is None:
            return None
        try:
            return grounding if grounding.available() else None
        except Exception:  # noqa: BLE001
            return None

    def _fast_reply(self, messages: list[dict], draft: dict, route: str, *,
                    t0: float | None = None, web: bool | None = None):
        """⚡ 高速経路（instant / fallback）。ニューラルを 1 トークンも使わない。

        v2 では「定型文の引き当て」ではなく **composer がその場で文を組み立てる**:
        発話を解析して計画（知識ベース／話題の受け取り／読み取れない入力／挨拶／社会的応答）
        を選び、文法と n-gram LM の検証を通した文だけを返す。材料が無ければ無いと正直に言う。
        """
        t0 = time.time() if t0 is None else t0
        last_user = last_user_of(messages)
        rule_text = str(draft.get("text") or "").strip()
        base_text = str(draft.get("base_text") or rule_text).strip()
        conf = float(draft.get("confidence", 1.0))
        fallback_engine = "Snipher-mini+"
        fast_engine = "Snipher Core (高速経路)"
        text = rule_text
        degraded = False
        if route == ROUTE_FALLBACK and conf < self.assist.cfg.threshold:
            text = base_text or rule_text
            degraded = text != rule_text
        info = {"engine": fallback_engine if route == ROUTE_FALLBACK else fast_engine,
                "template_mode": "rule-based", "plan": "draft", "confidence": conf,
                "fixes": list(draft.get("fixes") or []), "knowledge": None, "lm": None,
                "degraded": degraded, "source": "rule"}

        composer = None
        try:
            composer = self.composer()
        except Exception:  # noqa: BLE001
            log.debug("composer を初期化できません", exc_info=True)
        reply = None
        if composer is not None:
            try:
                reply = composer.compose(last_user, history=messages, web=web)
            except Exception:  # noqa: BLE001
                log.debug("composer が失敗", exc_info=True)
        if reply is not None and reply.text:
            from .composer import validate as _validate

            authoritative = bool((reply.notes or {}).get("authoritative"))
            ok, _why = (True, "authoritative") if authoritative else _validate(reply.text, max_len=220)
            if ok and len(reply.text) >= 4:
                text = reply.text
                lm_info = None
                lm = self.lm()
                if lm is not None:
                    try:
                        sc = lm.score(text)
                        lm_info = {"confidence": sc["confidence"], "perplexity": sc["perplexity"],
                                   "bad_ratio": sc["bad_ratio"]}
                    except Exception:  # noqa: BLE001
                        lm_info = None
                info.update({
                    "engine": (f"Snipher tool ({(reply.notes or {}).get('task', {}).get('kind', 'task')})"
                               if authoritative else fallback_engine if route == ROUTE_FALLBACK
                               else "Snipher composer (高速経路)"),
                    "template_mode": "tool-grounded" if authoritative else "composer", "plan": reply.plan,
                    "confidence": round(float(reply.confidence), 4),
                    "fixes": info["fixes"] + ["composer"],
                    "knowledge": reply.knowledge, "lm": lm_info,
                    "degraded": False, "source": "task" if authoritative else "composer",
                    "task": (reply.notes or {}).get("task") if authoritative else None,
                    "sentences": len(reply.sentences),
                })

        stats = {
            "engine": info["engine"],
            "template_mode": info["template_mode"],
            "new_tokens": None,
            "tokens_per_second": None,
            "assist": "rule",
            "draft": draft["text"],
            "draft_confidence": draft.get("confidence"),
            "draft_seconds": round(time.time() - t0, 5),
            "intent": draft.get("intent"),
            "fixes": info["fixes"],
            "route": route,
            "plan": info["plan"],
            "confidence": info["confidence"],
            "source": info["source"],
            "task": info.get("task"),
            "seconds": round(time.time() - t0, 4),
        }
        if info["knowledge"]:
            stats["knowledge"] = info["knowledge"]
        if info["lm"]:
            stats["lm"] = info["lm"]
        _srcs = _extract_sources(info.get("knowledge"), info.get("task"))
        if _srcs:
            stats["sources"] = _srcs
        if route == ROUTE_FALLBACK:
            stats["engine"] = fallback_engine
            stats["fallback_reason"] = self._fallback_reason()
            stats["degraded_to_base"] = info["degraded"]
            stats["acquire"] = self.acquire_status().get("phase")
        yield {"type": "assist", "mode": "rule", "confidence": draft.get("confidence"),
               "route": route, "plan": info["plan"],
               "reason": "task_result" if info.get("task") else "rule_result"}
        yield {"type": "start", "engine": stats["engine"], "template_mode": stats["template_mode"]}
        for piece in _chunk_for_stream(text):
            yield {"type": "delta", "text": piece}
        yield {"type": "done", "text": text, "stats": stats}

    def _light_reply(self, messages: list[dict], draft: dict, route: str, *,
                     mode: str = "auto", web: bool | None = None):
        """✨ 内蔵ニューラルコア + 知識ベース + composer の協働経路。

        分担はこうです:

            知識ベース  … 事実を出す（無ければ「無い」と言う）
            composer    … 事実を自然な日本語の応答に組み立て、話題を受け取る
            内蔵コア    … 材料が無いときに本文の生成に挑戦し、文末を補う
            n-gram LM   … 候補文を採点して、壊れた文・的外れな文を捨てる

        生成は 1 文字あたり数ミリ秒（NumPy）。フルウェイトは 1 バイトも要りません。
        """
        import re as _re

        core = self.light_core()
        lm = self.lm()
        composer = None
        try:
            composer = self.composer()
        except Exception:  # noqa: BLE001
            log.debug("composer を初期化できません", exc_info=True)
        last_user = last_user_of(messages)
        t0 = time.time()

        conf = float(draft.get("confidence", 1.0))
        rule_text = str(draft.get("text") or "").strip()
        base_text = str(draft.get("base_text") or rule_text).strip()

        # ---- 1) composer が計画と文を作る（知識ベースを内部で引く） ------------- #
        reply = None
        if composer is not None:
            try:
                reply = composer.compose(last_user, history=messages, web=web)
            except Exception:  # noqa: BLE001
                log.debug("composer が失敗", exc_info=True)
        if reply is None:                                    # composer が動かない環境
            kb = self.kb_answer(last_user)
            from .composer import Reply as _Reply
            body = str((kb or {}).get("text") or (rule_text if conf >= self.assist.cfg.threshold
                                                  else base_text)).strip()
            reply = _Reply(text=body, plan="kb_answer" if kb else "draft",
                           confidence=float((kb or {}).get("confidence", conf)),
                           knowledge=({"topic": kb.get("topic"), "score": kb.get("score"),
                                       "coverage": kb.get("coverage"), "usage": kb.get("usage")}
                                      if kb else None))

        info_kb = reply.knowledge
        authoritative = bool((reply.notes or {}).get("authoritative"))
        material = bool(info_kb and info_kb.get("topic"))
        topic = str((info_kb or {}).get("topic") or "")
        text = str(reply.text or "").strip()
        if not text:
            text = rule_text if conf >= self.assist.cfg.threshold else base_text

        yield {"type": "assist", "mode": "light", "route": route, "plan": reply.plan,
               "confidence": conf,
               "reason": "task_result" if authoritative else ("knowledge_hit" if material else "uncertain_slot"),
               "knowledge": info_kb}

        # ---- 2) 内蔵ニューラルコアの出番 --------------------------------------- #
        gen_text = ""
        used_core = False
        neural_conf: float | None = None      # 昇格判定に使う確信度
        report_conf: float | None = None      # 表示用の確信度
        ppl: float | None = None
        lm_conf: float | None = None
        note: str | None = None
        chosen_from_model = False
        extra = ""
        stats_extra: dict = {}

        needs_follow = not bool(_re.search(r"(か|かな|でしょう)[。！？!?]", text))
        force_light = str(mode or "") in ("lfm", "neural", "light")
        if core is not None and (not authoritative or force_light):
            if force_light:
                # 明示的に light/lfm が選ばれたターンは、材料がなくても生成を試みる
                gen_text = core.reply(last_user, context=messages, max_chars=34,
                                      temperature=0.6, top_k=24) or ""
                used_core = bool(gen_text)
                if gen_text:
                    sc = self.judge(gen_text, core=core, lm=lm)
                    report_conf = sc["confidence"]
                    ppl = sc["perplexity"]
                    lm_conf = sc["lm"]
                    if sc["confidence"] >= 0.62 and self.relevant(gen_text, last_user):
                        text = _re.sub(r"[。！!？?]?$", "。", text.rstrip("。")) + " " + gen_text
                        chosen_from_model = True
            if material:
                # 事実を伝えたあと、会話を続ける一文が **無ければ** だけ作らせる。
                # 知識ベースの followups（人が書いた問い）があるなら、それを優先する
                # （生成文は文法的でも中身が空のことがあるため）。
                if needs_follow and len(text) < 120:
                    gen_text = core.reply(last_user, context=messages, max_chars=28,
                                          temperature=0.5, top_k=20) or ""
                    used_core = bool(gen_text)
                    if gen_text:
                        sc = self.judge(gen_text, core=core, lm=lm)
                        report_conf = sc["confidence"]
                        ppl = sc["perplexity"]
                        lm_conf = sc["lm"]
                        on_topic = self.relevant(gen_text, last_user, topic=topic)
                        if (sc["confidence"] >= 0.75 and (sc["lm"] or 0.0) >= 0.7 and on_topic
                                and not self.redundant(gen_text, text)
                                and len(text) + len(gen_text) < 190):
                            extra = gen_text
            else:
                # 材料が無い → 本文の生成に挑戦させる。
                # 小さなモデルは 1 発だと外すので、温度を変えて複数候補を作り、
                # 「文法チェック → 内蔵コアの自信 → n-gram LM の自然さ → 話題の一致」を
                # 全部通した中から最も良い 1 文だけを採用する（best-of-N）。
                from .composer import validate as _validate

                bar = self.cfg.light_gate
                cands: list[str] = []
                for temp, topk, sd in ((0.60, 24, 11), (0.85, 40, 23), (0.45, 12, 37)):
                    try:
                        one = (core.reply(last_user, context=messages,
                                          max_chars=self.cfg.light_max_chars,
                                          temperature=temp, top_k=topk, seed=sd) or "").strip()
                    except Exception:  # noqa: BLE001
                        one = ""
                    if one and one not in cands:
                        cands.append(one)
                used_core = bool(cands)
                best: tuple[float, str, dict] | None = None
                for one in cands:
                    ok, _why = _validate(one, max_len=200)
                    if not ok:
                        continue
                    sc = self.judge(one, core=core, lm=lm)
                    if sc["confidence"] < bar or float(sc["lm"] or 0.0) < bar:
                        continue
                    if not self.relevant(one, last_user, topic=topic):
                        continue
                    rank = float(sc["confidence"]) * float(sc["lm"] or 0.0)
                    if best is None or rank > best[0]:
                        best = (rank, one, sc)
                if best is not None:
                    gen_text = best[1]
                    sc = best[2]
                    text = gen_text
                    chosen_from_model = True
                    stats_extra["candidates"] = len(cands)
                elif cands:
                    gen_text = cands[0]
                    sc = self.judge(gen_text, core=core, lm=lm)
                    note = "low_confidence_ack"
                    stats_extra["candidates"] = len(cands)
                    stats_extra["rejected"] = "gate"
                else:
                    sc = None
                if sc is not None:
                    neural_conf = sc["confidence"]
                    report_conf = sc["confidence"]
                    ppl = sc["perplexity"]
                    lm_conf = sc["lm"]
                if neural_conf is None:
                    sc = self.judge(text, core=core, lm=lm)
                    neural_conf = report_conf = sc["confidence"]
                    ppl = sc["perplexity"]
                    lm_conf = sc["lm"]

        # ---- 3) 助動詞の補い（内蔵コアの神経系） -------------------------------- #
        added = ""
        if core is not None and _looks_incomplete(text):
            used_core = True
            try:
                comp = core.complete(text)
            except Exception:  # noqa: BLE001
                comp = {}
            if comp.get("changed") and float(comp.get("confidence", 0) or 0) >= 0.25:
                added = comp.get("added") or ""
                text = comp.get("text") or text

        # ---- 3.5) どの経路でも最終文は必ず判定する --------------------------- #
        # 昇格（フルウェイト起動）の判定は「内蔵コアの自信」で行う。知識ベースや
        # ツールが答えられた回でも、文そのものの自然さは測っておくと、
        # 後から重みが揃ったときにどれを差し替えるべきかが分かる。
        if neural_conf is None and core is not None:
            try:
                sc = self.judge(f"{text}{extra}", core=core, lm=lm)
                neural_conf = report_conf = float(sc["confidence"])
                ppl = sc["perplexity"]
                lm_conf = sc["lm"]
            except Exception:  # noqa: BLE001
                pass

        # ---- 4) 磨いて出す ---------------------------------------------------- #
        polished = self.polisher.polish(f"{text}{extra}", register="polite")
        fixes = list(polished["fixes"])
        if added:
            fixes.append("neural_completion")
        if composer is not None:
            fixes.append("composer")
        final = str(polished["text"] or text).strip() or text
        if authoritative:
            task_kind = str((reply.notes or {}).get("task", {}).get("kind") or "task")
            engine = f"Snipher tool ({task_kind})"
            template_mode = "tool-grounded"
        elif used_core:
            engine = core.engine_name()
            template_mode = "distilled-numpy"
        elif material:
            engine = "Snipher 知識ベース + composer"
            template_mode = "knowledge+composer"
        else:
            engine = "Snipher composer"
            template_mode = "composer"

        yield {"type": "start", "engine": engine, "template_mode": template_mode}
        for piece in _chunk_for_stream(final, pieces=2):
            yield {"type": "delta", "text": piece}

        stats = {
            "engine": engine,
            "template_mode": template_mode,
            "neural_used": used_core,
            "assist": "light",
            "route": route,
            "plan": reply.plan,
            "draft": rule_text,
            "draft_confidence": conf,
            "confidence": round(float(reply.confidence), 4),
            "intent": draft.get("intent"),
            "fixes": fixes,
            "generated": chosen_from_model or bool(extra),
            "candidates": 1 + (1 if gen_text else 0),
            "task": (reply.notes or {}).get("task") if authoritative else None,
            "knowledge": info_kb,
            "neural_confidence": round(report_conf, 4) if report_conf is not None else None,
            "lm_confidence": lm_conf,
            "perplexity": ppl,
            "seconds": round(time.time() - t0, 4),
            "new_tokens": None,
            "tokens_per_second": None,
        }
        _srcs = _extract_sources(info_kb, (reply.notes or {}).get("task"))
        if _srcs:
            stats["sources"] = _srcs
        if stats_extra:
            stats.update(stats_extra)
        if note:
            stats["note"] = note
            stats["fallback_reason"] = "言葉を整理して応答しました (追加の文脈があれば深掘りします)"
        if core is None:
            stats["fallback_reason"] = "内蔵ニューラルコアの重みが未ビルドです"
        gate_val = neural_conf if neural_conf is not None else conf
        if gate_val < self.cfg.light_gate and not self.neural_available():
            # 確信度が低い → フルウェイトが使える環境なら裏で起動準備（応答は止めない）
            self.ensure_started()
            stats["escalation"] = "queued_full_weights"
            stats["escalation_gate"] = self.cfg.light_gate
        elif not authoritative and not used_core and not material:
            stats["fallback_reason"] = "確信度が足りるので生成は見送りました"
        yield {"type": "done", "text": final, "stats": stats}

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
        if backend is None:  # 判定直後に落ちた場合の安全網（→ 内蔵蒸留コア）
            if self.light_ready() or self.kb is not None:
                yield from self._light_reply(messages, draft, ROUTE_LIGHT, mode=mode)
                return
        if backend is None:  # ニューラル整体が使えない場合の最終安全網
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
            kb = self.kb_answer(last_user_of(messages))
            if kb:
                sp += (f"\n（Snipher 知識ベースが引けた事実: {str(kb.get('text'))[:520]}）"
                       "この事実を踏まえて、短く自然に答えてください。")

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
        # 重いコアが無ければ内蔵蒸留コア（LFM2.5 と同じ役割）で補う
        light = None if backend is not None or not use_neural else self.light_core()
        if light is not None and _looks_incomplete(fixed["text"]):
            comp = light.complete(fixed["text"])
            if comp.get("changed") and float(comp.get("confidence", 0)) >= 0.25:
                return {"text": comp["text"],
                        "fixes": fixed["fixes"] + ["neural_completion"],
                        "engine": light.engine_name(),
                        "added": comp.get("added", ""),
                        "confidence": comp.get("confidence")}
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


def _clean_sentence(text: str) -> str:
    """生成候補を軽く整形（長すぎ・空・記号だけの候補を落とす）。"""
    t = str(text or "").strip().replace("\n", "")
    return t if 6 <= len(t) <= 60 else ""


def last_user_of(messages: list[dict]) -> str:
    """直近のユーザー発話を取り出す（コア内部の共通処理）。"""
    return next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")


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


def _extract_sources(*dicts: dict | None) -> list[dict]:
    """reply/task/knowledge から Web出典だけを抜く (UIの favicon 行用)。"""
    out: list[dict] = []
    seen: set[str] = set()
    for d in dicts:
        if not isinstance(d, dict):
            continue
        srcs = d.get("sources")
        if isinstance(srcs, list):
            for s in srcs:
                if not isinstance(s, dict):
                    continue
                url = str(s.get("url") or "").strip()
                if not url or url in seen or not url.startswith(("http://", "https://")):
                    continue
                seen.add(url)
                out.append({"title": str(s.get("title") or url)[:80], "url": url})
                if len(out) >= 6:
                    return out
        # task ラッパー (notes.task.metadata.sources) も掘る
        meta = d.get("metadata") if isinstance(d.get("metadata"), dict) else None
        if meta and isinstance(meta.get("sources"), list):
            for s in meta["sources"]:
                if not isinstance(s, dict):
                    continue
                url = str(s.get("url") or "").strip()
                if not url or url in seen or not url.startswith(("http://", "https://")):
                    continue
                seen.add(url)
                out.append({"title": str(s.get("title") or url)[:80], "url": url})
                if len(out) >= 6:
                    return out
    return out


def _chunk_for_stream(text: str, pieces: int = 3) -> list[str]:
    """軽量経路のテキストを擬似ストリーミング用に分割する。

    通常文は文単位、コード・長文 (小説等) は行単位で刻む。
    """
    import re as _re

    text = str(text or "")
    if not text:
        return [text]
    # コードブロック・長文は行単位で 600 字ずつ
    if "```" in text or len(text) > 900:
        lines = text.split("\n")
        chunks: list[str] = []
        buf = ""
        for ln in lines:
            if len(buf) + len(ln) + 1 > 600 and buf:
                chunks.append(buf + "\n")
                buf = ln
            else:
                buf = (buf + "\n" + ln) if buf else ln
        if buf:
            chunks.append(buf)
        return chunks or [text]
    parts = [p for p in _re.split(r"(?<=。)|(?<=？)|(?<=！)", text) if p]
    if len(parts) <= pieces:
        return parts or [text]
    merged: list[str] = []
    per = max(1, -(-len(parts) // pieces))
    for i in range(0, len(parts), per):
        merged.append("".join(parts[i : i + per]))
    return merged
