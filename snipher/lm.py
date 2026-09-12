"""巨大 n-gram 言語モデル（文字 1〜N gram・線形補間）。

Snipher の「流暢さの審判」です。役割は 3 つ:

    1. ニューラルコアが作った文を採点して、日本語として壊れていたら捨てる
    2. composer / ニューラルの候補を並べ替えて、自然な言い回しを選ぶ
    3. numpy のニューラルコアが使えない環境でも、文字単位で続きを生成できる

なぜ n-gram か:

* **速い** … 1 文の採点は numpy の searchsorted を次数ぶん呼ぶだけ（1ms 未満）
* **軽い** … 量子化 + 圧縮で数 MB。ロードは 100ms 前後
* **巨大** … エントリ数は数百万〜数千万。「パラメータ数」では小型ニューラルネットより
  1〜2 桁多く、しかも推論は行列積ではなく検索と四則演算

学習データは外部から落としません。`tools/build_lm.py` が知識ベース（人が書いた日本語）
と語彙テーブルから文法で組み立てた文だけで構築します。
"""

from __future__ import annotations

import json
import math
import re
import time
import zlib
from pathlib import Path

import numpy as np

DEFAULT_PATH = Path(__file__).resolve().parent / "data" / "lm.npz"
_MAGIC = b"SNPLM001"

BOS = "\x02"      # 文の始まり
EOS = "\x03"      # 文の終わり
UNK = "\x01"      # 語彙外の文字

_CLEAN = re.compile(r"[\t\r\f\v]+")


def normalize(text: str) -> str:
    t = str(text or "").replace("\u3000", " ")
    t = _CLEAN.sub(" ", t)
    t = t.replace("？", "?").replace("！", "!")
    return t.strip()


# ---------------------------------------------------------------------- #
# ハッシュ（64bit に混ぜる。この規模の衝突は無視できる）
# ---------------------------------------------------------------------- #
_M64 = (1 << 64) - 1
_MIX_A = 0x9E3779B97F4A7C15
_MIX_B = 0xBF58476D1CE4E5B9


def _mix(ctx, tok: int) -> int:
    """文脈（トークン ID 列）+ 次の文字 → 64bit キー。順序を保つ多項式ハッシュ。"""
    h = len(ctx) + 1
    for c in ctx:
        h = ((h ^ (c + 1)) * _MIX_A) & _M64
        h ^= h >> 29
    h = ((h ^ (int(tok) + 1)) * _MIX_B) & _M64
    return h ^ (h >> 32)


def _ctx_key(ctx) -> int:
    """(n-1)-gram そのもののキー（補間の分母を引くときに使う）。"""
    if not ctx:
        return 0
    return _mix(ctx[:-1], ctx[-1])


# 1 文字の log 確率がこれより下なら「日本語として壊れている」扱い
BAD_CHAR_LOGP = -13.0


def confidence_from_ppl(ppl: float, calib: dict | None = None) -> float:
    """perplexity → 0..1 の確信度。

    calib = {"lo": 自然な文の ppl, "hi": 文字をシャッフルした文の ppl} があれば、
    その 2 点の対数スケールで線形に引く（ビルド時に実データから決まる）。
    無ければ汎用カーブ（ppl 2 → 0.95, 6 → 0.66, 12 → 0.48, 40 → 0.28）。
    """
    p = max(1.0001, float(ppl))
    if calib:
        lo = max(1.0001, float(calib.get("lo", 0.0)))
        hi = max(lo * 1.2, float(calib.get("hi", 0.0)))
        span = math.log(hi) - math.log(lo)
        if span > 1e-6:
            t = (math.log(hi) - math.log(p)) / span
            return float(max(0.02, min(0.97, 0.05 + 0.90 * t)))
    return float(max(0.0, min(0.99, 1.0 / (1.0 + math.log(p / 1.9) / 1.05))))


class CharNgramLM:
    """文字 n-gram（線形補間）。重みはファイルから読むか、その場で学習する。"""

    def __init__(self, order: int = 5, lambdas: tuple[float, ...] | None = None):
        self.order = max(1, int(order))
        default = (0.0, 0.08, 0.12, 0.18, 0.26, 0.36)
        self.lambdas = tuple(lambdas) if lambdas else tuple(
            list(default) + [0.0] * (self.order + 1 - len(default)))[: self.order + 1]
        self.vocab: list[str] = [UNK, BOS, EOS]
        self.stoi: dict[str, int] = {c: i for i, c in enumerate(self.vocab)}
        # 次数ごとの n-gram テーブル（keys: uint64 昇順, counts: float32）
        self.keys: list[np.ndarray] = [np.zeros(0, dtype=np.uint64) for _ in range(self.order + 1)]
        self.counts: list[np.ndarray] = [np.zeros(0, dtype=np.float32) for _ in range(self.order + 1)]
        self.total = 0.0
        self.trained_at: float | None = None
        self.metrics: dict = {}
        # 確信度の校正（ビルド時に実データから決める）。無ければ既定カーブ
        self.calib: dict = {}
        self._unigram: np.ndarray | None = None
        self.path: Path | None = None
        self.load_seconds: float | None = None

    # ------------------------------------------------------------------ #
    @property
    def n_vocab(self) -> int:
        return len(self.vocab)

    def n_params(self) -> int:
        """エントリ数 = このモデルの「パラメータ数」。"""
        return int(sum(k.size for k in self.keys))

    @property
    def is_ready(self) -> bool:
        return self.total > 0 and len(self.vocab) > 3

    def engine_name(self) -> str:
        return f"Snipher char-{self.order}gram LM ({self.n_params():,} entries)"

    def bytes_on_disk(self) -> int:
        try:
            return self.path.stat().st_size if self.path and self.path.exists() else 0
        except OSError:
            return 0

    # ------------------------------------------------------------------ #
    # 学習
    # ------------------------------------------------------------------ #
    def build_vocab(self, texts: list[str], max_vocab: int = 4000) -> None:
        from collections import Counter

        cnt: Counter[str] = Counter()
        for t in texts:
            cnt.update(normalize(t))
        self.vocab = [UNK, BOS, EOS] + [c for c, _ in cnt.most_common(max_vocab)
                                        if c not in (UNK, BOS, EOS)]
        self.stoi = {c: i for i, c in enumerate(self.vocab)}

    def encode(self, text: str) -> list[int]:
        g = self.stoi.get
        return [g(ch, 0) for ch in normalize(text)]

    def train(self, texts, *, min_counts: tuple[int, ...] | None = None,
              build_vocab: bool = True) -> dict:
        """テキストから n-gram を数える。min_counts[n] 未満のエントリは捨てる。"""
        t0 = time.time()
        texts = list(texts)
        if build_vocab or len(self.vocab) <= 3:
            self.build_vocab(texts)
        mc = tuple(min_counts or (1,) * (self.order + 1))
        tables: list[dict[int, int]] = [dict() for _ in range(self.order + 1)]
        order = self.order
        bos = self.stoi[BOS]
        eos = self.stoi[EOS]
        total = 0
        for text in texts:
            ids = self.encode(text)
            if not ids:
                continue
            seq = [bos] * order + ids + [eos]
            total += len(ids) + 1
            for i in range(order, len(seq)):
                tok = seq[i]
                hist = seq[i - order:i]
                # tables[n] = n-gram（文脈 n-1 + 対象 1）
                t1 = tables[1]
                k = tok + 1
                t1[k] = t1.get(k, 0) + 1
                for n in range(2, order + 1):
                    t = tables[n]
                    k = _mix(hist[len(hist) - (n - 1):], tok)
                    t[k] = t.get(k, 0) + 1
        self.keys, self.counts = [], []
        kept: list[int] = []
        for n in range(self.order + 1):
            if n == 0:
                self.keys.append(np.zeros(0, dtype=np.uint64))
                self.counts.append(np.zeros(0, dtype=np.float32))
                kept.append(0)
                continue
            cut = mc[n] if n < len(mc) else 1
            items = sorted((k, float(c)) for k, c in tables[n].items() if c >= max(1, cut))
            self.keys.append(np.fromiter((k for k, _ in items), dtype=np.uint64, count=len(items)))
            self.counts.append(np.fromiter((c for _, c in items), dtype=np.float32, count=len(items)))
            kept.append(len(items))
        self.total = float(total)
        self._unigram = None
        self.trained_at = time.time()
        self.metrics = {"tokens": total, "entries": kept, "order": order,
                        "seconds": round(time.time() - t0, 2), "vocab": len(self.vocab)}
        return self.metrics

    # ------------------------------------------------------------------ #
    # 保存 / 読み込み（uint64 キー + uint16 量子化カウント + zlib ヘッダ）
    # ------------------------------------------------------------------ #
    def save(self, path: str | Path = DEFAULT_PATH) -> dict:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        header = {"order": self.order, "lambdas": list(self.lambdas), "vocab": self.vocab,
                  "total": self.total, "trained_at": self.trained_at, "metrics": self.metrics,
                  "calib": self.calib}
        blob = _MAGIC + zlib.compress(json.dumps(header, ensure_ascii=False).encode("utf-8"), 9)
        arrays: dict[str, np.ndarray] = {"header": np.frombuffer(blob, dtype=np.uint8)}
        for n in range(1, self.order + 1):
            k, c = self.keys[n], self.counts[n]
            if k.size == 0:
                continue
            arrays[f"k{n}"] = k
            scale = max(1.0, float(c.max()) / 65534.0) if c.size else 1.0
            arrays[f"c{n}"] = np.round(c / scale).astype(np.uint16)
            arrays[f"s{n}"] = np.array([scale], dtype=np.float32)
        out = p if p.suffix == ".npz" else p.with_suffix(".npz")
        np.savez_compressed(out, **arrays)
        self.path = out
        return {"path": str(out), "bytes": out.stat().st_size, "params": self.n_params()}

    @classmethod
    def load(cls, path: str | Path = DEFAULT_PATH) -> "CharNgramLM | None":
        p = Path(path)
        if not p.exists():
            p = p.with_suffix(".npz")
        if not p.exists():
            return None
        t0 = time.time()
        try:
            with np.load(p, allow_pickle=False) as z:
                blob = z["header"].tobytes()
                if not blob.startswith(_MAGIC):
                    return None
                head = json.loads(zlib.decompress(blob[len(_MAGIC):]).decode("utf-8"))
                lm = cls(order=int(head["order"]), lambdas=tuple(head["lambdas"]))
                lm.vocab = list(head["vocab"])
                lm.stoi = {c: i for i, c in enumerate(lm.vocab)}
                lm.total = float(head.get("total", 0.0))
                lm.trained_at = head.get("trained_at")
                lm.metrics = head.get("metrics", {})
                lm.calib = head.get("calib", {}) or {}
                lm.keys = [np.zeros(0, dtype=np.uint64) for _ in range(lm.order + 1)]
                lm.counts = [np.zeros(0, dtype=np.float32) for _ in range(lm.order + 1)]
                for n in range(1, lm.order + 1):
                    if f"k{n}" not in z.files:
                        continue
                    lm.keys[n] = z[f"k{n}"]
                    scale = float(z[f"s{n}"][0]) if f"s{n}" in z.files else 1.0
                    lm.counts[n] = z[f"c{n}"].astype(np.float32) * np.float32(scale)
        except Exception:  # noqa: BLE001
            return None
        lm.path = p
        lm.load_seconds = round(time.time() - t0, 4)
        return lm

    # ------------------------------------------------------------------ #
    # 推論
    # ------------------------------------------------------------------ #
    def _lookup(self, n: int, keys: np.ndarray) -> np.ndarray:
        """次数 n のテーブルからカウントを引く（無いものは 0）。"""
        if n < 1 or n > self.order:
            return np.zeros(keys.shape, dtype=np.float32)
        table = self.keys[n]
        if table.size == 0 or keys.size == 0:
            return np.zeros(keys.shape, dtype=np.float32)
        pos = np.clip(np.searchsorted(table, keys), 0, table.size - 1)
        hit = table[pos] == keys
        return np.where(hit, self.counts[n][pos], 0.0).astype(np.float32)

    def _keys_for(self, ids: list[int], n: int) -> tuple[np.ndarray, np.ndarray]:
        """各位置の n-gram キーと、その分母になる (n-1)-gram キーを作る。"""
        bos = self.stoi[BOS]
        seq = [bos] * self.order + list(ids)
        off, L = self.order, len(ids)
        out = np.zeros(L, dtype=np.uint64)
        ctx = np.zeros(L, dtype=np.uint64)
        if n == 1:
            for i in range(L):
                out[i] = seq[off + i] + 1
            return out, ctx
        for i in range(L):
            start = off + i - (n - 1)
            window = seq[start: off + i + 1]
            if len(window) < n:
                window = [bos] * (n - len(window)) + window
            out[i] = _mix(window[:-1], window[-1])
            ctx[i] = _mix(window[:-2], window[-2])
        return out, ctx

    def unigram_vector(self) -> np.ndarray:
        """語彙全体の unigram カウント（生成と補間の土台）。"""
        if self._unigram is not None:
            return self._unigram
        V = self.n_vocab
        self._unigram = self._lookup(1, np.arange(1, V + 1, dtype=np.uint64))
        return self._unigram

    def unigram_total(self) -> float:
        return float(self.counts[1].sum()) if self.counts[1].size else max(1.0, self.total)

    def logprob(self, text: str) -> tuple[float, int]:
        """1 文字あたり平均 log 確率と文字数。"""
        ids = self.encode(text)
        if not ids:
            return 0.0, 0
        return float(self._logprob_ids(ids).mean()), len(ids)

    def _logprob_ids(self, ids: list[int]) -> np.ndarray:
        lam = self.lambdas
        V = self.n_vocab
        probs = np.zeros(len(ids), dtype=np.float64)
        tot1 = max(1.0, self.unigram_total())
        k1, _ = self._keys_for(ids, 1)
        probs += lam[1] * (self._lookup(1, k1) / tot1)
        for n in range(2, self.order + 1):
            if lam[n] <= 0.0:
                continue
            kn, ck = self._keys_for(ids, n)
            cn = self._lookup(n, kn)
            cd = self._lookup(n - 1, ck)
            with np.errstate(divide="ignore", invalid="ignore"):
                pn = np.where(cd > 0, cn / np.maximum(cd, 1e-9), 0.0)
            probs += lam[n] * pn
        # 残りの確率質量を語彙全体に薄く配る（未知文字でも 0 にはしない）
        floor = max(1e-9, 1.0 - sum(lam[1:self.order + 1]))
        probs = np.maximum(probs, floor / max(1, V))
        return np.log(probs)

    def score(self, text: str) -> dict:
        ids = self.encode(text)
        if not ids:
            return {"logprob": 0.0, "perplexity": 1e6, "confidence": 0.0, "chars": 0,
                    "worst": 0.0, "bad_ratio": 0.0, "ok": False}
        lps = self._logprob_ids(ids)
        lp = float(lps.mean())
        n = len(ids)
        ppl = math.exp(min(30.0, -lp))
        # 平均だけでは「1 文字だけ壊れている文」を見逃すので、局所の指標も持つ
        worst = float(lps.min())
        bad = int((lps < BAD_CHAR_LOGP).sum())
        conf = confidence_from_ppl(ppl, self.calib)
        conf *= 1.0 - 0.5 * (bad / max(1, n))          # 壊れた文字のぶんだけ減点
        return {"logprob": round(lp, 4), "perplexity": round(ppl, 3),
                "confidence": round(max(0.0, conf), 4), "chars": n,
                "worst": round(worst, 4), "bad": bad, "bad_ratio": round(bad / max(1, n), 4),
                "ok": True}

    def confidence(self, text: str) -> float:
        return float(self.score(text)["confidence"])

    def perplexity(self, text: str) -> float:
        return float(self.score(text)["perplexity"])

    def rank(self, candidates, *, prefix: str = "") -> list[tuple[float, str]]:
        """候補文を自然さの高い順に並べ替える。→ [(合計 log 確率, 文)]"""
        scored: list[tuple[float, str]] = []
        for c in candidates:
            t = str(c or "").strip()
            if not t:
                continue
            ids = self.encode(prefix + t)
            total = float(self._logprob_ids(ids).sum()) if ids else -1e9
            scored.append((total, t))
        scored.sort(key=lambda kv: -kv[0])
        return scored

    # ------------------------------------------------------------------ #
    def next_probs(self, prefix: str, *, temperature: float = 1.0) -> np.ndarray:
        """prefix の次に続く文字の確率分布。"""
        ids = self.encode(prefix)
        V = self.n_vocab
        lam = self.lambdas
        seq = [self.stoi[BOS]] * self.order + ids
        tot1 = max(1.0, self.unigram_total())
        probs = lam[1] * (self.unigram_vector() / tot1)
        for n in range(2, self.order + 1):
            if lam[n] <= 0.0:
                continue
            ctx = seq[len(seq) - (n - 1):]
            if len(ctx) < n - 1:
                ctx = [self.stoi[BOS]] * (n - 1 - len(ctx)) + ctx
            denom = float(self._lookup(n - 1, np.array([_ctx_key(ctx)], dtype=np.uint64))[0])
            if denom <= 0.0:
                continue
            keys = np.fromiter((_mix(ctx, c) for c in range(V)), dtype=np.uint64, count=V)
            probs = probs + lam[n] * (self._lookup(n, keys) / denom)
        probs = np.maximum(probs, 1e-12)
        if temperature and abs(temperature - 1.0) > 1e-6:
            probs = probs ** (1.0 / max(1e-3, temperature))
        s = float(probs.sum())
        return probs / s if s > 0 else probs

    def generate(self, prompt: str = "", *, max_chars: int = 48, temperature: float = 0.8,
                 top_k: int = 24, seed: int | None = None,
                 stop: tuple[str, ...] = ("。", "!", "?")) -> str:
        """文字単位で続きを生成する（ニューラルコアが使えないときの予備）。"""
        rng = np.random.default_rng(seed)
        cur = prompt or ""
        out: list[str] = []
        eos = self.stoi[EOS]
        for _ in range(max_chars):
            p = self.next_probs(cur, temperature=temperature)
            p[eos] = 0.0
            if top_k and 0 < top_k < len(p):
                cut = np.partition(p, -top_k)[-top_k]
                p = np.where(p < cut, 0.0, p)
            s = float(p.sum())
            if s <= 0:
                break
            nxt = int(rng.choice(len(p), p=p / s))
            if nxt >= len(self.vocab):
                break
            ch = self.vocab[nxt]
            if ch in (UNK, BOS, EOS):
                break
            out.append(ch)
            cur += ch
            if ch in stop and len(out) > 4:
                break
        return "".join(out).strip()

    # ------------------------------------------------------------------ #
    def status(self) -> dict:
        return {
            "kind": "ngram",
            "state": "ready" if self.is_ready else "unavailable",
            "engine": self.engine_name(),
            "order": self.order,
            "params": self.n_params(),
            "vocab": self.n_vocab,
            "tokens": int(self.total),
            "bytes": self.bytes_on_disk(),
            "load_seconds": self.load_seconds,
            "trained_at": self.trained_at,
            "metrics": self.metrics,
            "calib": self.calib,
            "path": str(self.path) if self.path else None,
            "runtime": "numpy",
            "download_required": False,
        }


# ---------------------------------------------------------------------- #
_SHARED: CharNgramLM | None = None


def shared(path: str | Path | None = None) -> CharNgramLM | None:
    """プロセス共通インスタンス（重みファイルが無ければ None）。"""
    global _SHARED
    if _SHARED is not None and path is None:
        return _SHARED
    lm = CharNgramLM.load(path or DEFAULT_PATH)
    if lm is not None and path is None:
        _SHARED = lm
    return lm


def reset() -> None:
    global _SHARED
    _SHARED = None
