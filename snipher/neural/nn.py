"""内蔵ニューラルコアの計算核（pure numpy・forward/backward 両対応）。

LFM2.5 と同じ **Hybrid 構造**をそのまま小型化したものです。

    x = Embedding(chars)
    for block in blocks:                      # conv / attn を交互に積む
        if conv:  # LFM2 の短距離畳み込み（局所依存をほぼタダで捉える）
            h = RMSNorm(x)
            gate, val = split(h @ W_gu)
            u = val * silu(gate)                          # ゲート
            u = CausalDepthwiseConv(u, W_conv, dilation)  # 未来を見ない
            x = x + silu(u) @ W_out
        else:     # 限定的な注意（長距離の文脈・文末整合用）
            h = RMSNorm(x)
            q, k, v = split(h @ W_qkv)  (+ RoPE)
            x = x + CausalSoftmax(q kᵗ/√d) v @ W_o
    logits = RMSNorm(x) @ Embeddingᵗ          # 重み共有

実装は numpy のみ。 int8（行別スケール）で量子化した重みを同梱し、ロード時に
float32 へ逆量子化します（約 1MB / 使用メモリ 数 MB）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SPECIAL_N = 7  # tokenizer.SPECIAL_TOKENS の数


@dataclass
class NNConfig:
    n_vocab: int = 640
    d_model: int = 192
    n_layers: int = 6
    n_heads: int = 6
    conv_kernel: int = 4
    max_pos: int = 320
    rope_base: float = 10000.0
    eps: float = 1e-5
    blocks: tuple[str, ...] | None = None       # None → conv/attn を交互
    dilations: tuple[int, ...] | None = None    # None → conv 層ごとに 1,2,4,...

    def __post_init__(self) -> None:
        if self.blocks is None:
            self.blocks = tuple("conv" if i % 2 == 0 else "attn" for i in range(self.n_layers))
        self.blocks = tuple(self.blocks)
        self.n_conv = max(1, sum(1 for b in self.blocks if b == "conv"))
        if self.dilations is None:
            self.dilations = tuple(2 ** i for i in range(self.n_conv))
        self.dilations = tuple(self.dilations) or (1,)
        if self.d_model % self.n_heads:
            raise ValueError("d_model は n_heads の倍数である必要があります")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def to_dict(self) -> dict:
        return {
            "n_vocab": self.n_vocab,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "conv_kernel": self.conv_kernel,
            "max_pos": self.max_pos,
            "rope_base": self.rope_base,
            "blocks": list(self.blocks),
            "dilations": list(self.dilations),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "NNConfig":
        keep = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**keep)


# ---------------------------------------------------------------------- #
# 素朴な演算ヘルパ
# ---------------------------------------------------------------------- #
def _lag(a: np.ndarray, sh: int) -> np.ndarray:
    """out[t] = a[t-sh]（sh だけ未来へ送る。先頭はゼロ）。"""
    if sh <= 0:
        return a
    out = np.zeros_like(a)
    out[:, sh:] = a[:, :-sh]
    return out


def _lead(a: np.ndarray, sh: int) -> np.ndarray:
    """out[t] = a[t+sh]（sh だけ過去へ引く。末尾はゼロ）。"""
    if sh <= 0:
        return a
    out = np.zeros_like(a)
    out[:, :-sh] = a[:, sh:]
    return out


def _rmsnorm(x: np.ndarray, g: np.ndarray, eps: float):
    r = np.sqrt((x * x).mean(-1, keepdims=True) + eps)
    return x / r * g, r


def _silu(x: np.ndarray):
    sig = 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))
    return x * sig, sig


def _softmax_last(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def _causal_mask(T: int, pad: np.ndarray | None) -> np.ndarray:
    """(B,1,T,T) or (1,1,T,T) の加算用マスク（許可=0 / 不可=-1e30）。"""
    neg = np.float32(-1e30)
    keep = np.tril(np.ones((T, T), dtype=bool))
    m = np.where(keep, np.float32(0.0), neg)
    if pad is None:
        return m[None, None].astype(np.float32)
    m = np.broadcast_to(m, (pad.shape[0], 1, T, T)).copy()
    bad = (pad == 0)[:, None, None, :]          # (B,1,1,T) → 列を無効化
    return np.where(bad, neg, m).astype(np.float32)


def _rope_apply(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    c, s = cos[None, None], sin[None, None]
    return np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], axis=-1)


def _rope_grad(g: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    half = g.shape[-1] // 2
    g1, g2 = g[..., :half], g[..., half:]
    c, s = cos[None, None], sin[None, None]
    return np.concatenate([g1 * c + g2 * s, g2 * c - g1 * s], axis=-1)


# ---------------------------------------------------------------------- #
# モデル
# ---------------------------------------------------------------------- #
@dataclass
class MicroNet:
    """内蔵ニューラルコア。重みは float32 の dict。"""

    cfg: NNConfig
    params: dict[str, np.ndarray] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    @classmethod
    def random(cls, cfg: NNConfig, seed: int = 0, emb_std: float = 0.08) -> "MicroNet":
        rng = np.random.default_rng(seed)
        d = cfg.d_model
        p: dict[str, np.ndarray] = {}
        p["emb"] = rng.normal(0.0, emb_std, size=(cfg.n_vocab, d)).astype(np.float32)
        p["emb"][0] = 0.0                     # <pad> の埋め込みは常にゼロ
        p["g_final"] = np.ones(d, dtype=np.float32)
        for i, kind in enumerate(cfg.blocks):
            p[f"g_{i}"] = np.ones(d, dtype=np.float32)
            if kind == "conv":
                k = cfg.conv_kernel
                p[f"gu_W_{i}"] = rng.normal(0, (2 * d) ** -0.5, size=(d, 2 * d)).astype(np.float32)
                p[f"gu_b_{i}"] = np.zeros(2 * d, dtype=np.float32)
                p[f"conv_W_{i}"] = rng.normal(0.05, 1.0, size=(d, k)).astype(np.float32)
                p[f"conv_b_{i}"] = np.zeros(d, dtype=np.float32)
                p[f"out_W_{i}"] = np.zeros((d, d), dtype=np.float32)
                p[f"out_b_{i}"] = np.zeros(d, dtype=np.float32)
            else:
                p[f"qkv_W_{i}"] = rng.normal(0, d ** -0.5, size=(d, 3 * d)).astype(np.float32)
                p[f"qkv_b_{i}"] = np.zeros(3 * d, dtype=np.float32)
                p[f"o_W_{i}"] = np.zeros((d, d), dtype=np.float32)   # 初期出力ゼロ（残差安定化）
                p[f"o_b_{i}"] = np.zeros(d, dtype=np.float32)
        return cls(cfg=cfg, params=p)

    def n_params(self) -> int:
        return int(sum(v.size for v in self.params.values()))

    def rope_freqs(self, T: int) -> tuple[np.ndarray, np.ndarray]:
        half = self.cfg.head_dim // 2
        inv = self.cfg.rope_base ** (-np.arange(half, dtype=np.float32) / max(1, half))
        ang = np.arange(T, dtype=np.float32)[:, None] * inv[None, :]
        return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)

    # ------------------------------------------------------------------ #
    def forward(self, ids: np.ndarray, pad: np.ndarray | None = None, with_grad: bool = False,
                fill_cache: "DecodeCache | None" = None):
        """(B,T) の全位置を一度に計算する。

        `fill_cache` に DecodeCache を渡すと、逐次デコード（KV キャッシュ）に必要な
        中間状態（conv の直近 u 列 / attention の K,V）を同じ計算から取り出す。
        """
        cfg, P = self.cfg, self.params
        B, T = ids.shape
        d = cfg.d_model
        x = P["emb"][ids]
        if pad is not None:
            x = x * pad[:, :, None]
        cache: dict[str, object] = {}
        cos, sin = self.rope_freqs(T)
        dil_i = 0

        for i, kind in enumerate(cfg.blocks):
            h, r = _rmsnorm(x, P[f"g_{i}"], cfg.eps)
            if with_grad:
                cache[f"h_{i}"], cache[f"r_{i}"], cache[f"xin_{i}"] = h, r, x
            if kind == "conv":
                gu = h @ P[f"gu_W_{i}"] + P[f"gu_b_{i}"]
                gate, val = np.split(gu, 2, axis=-1)
                sg, sig = _silu(gate)
                u = val * sg
                k = cfg.conv_kernel
                Wc = P[f"conv_W_{i}"]
                dil = cfg.dilations[dil_i % len(cfg.dilations)]
                shifts = [(k - 1 - j) * dil for j in range(k)]
                c = np.broadcast_to(P[f"conv_b_{i}"], (B, T, d)).copy()
                taps = []
                for j, sh in enumerate(shifts):
                    lag = _lag(u, sh)
                    taps.append(lag)
                    c = c + lag * Wc[None, :, j]
                s2, sig2 = _silu(c)
                o = s2 @ P[f"out_W_{i}"] + P[f"out_b_{i}"]
                x = x + o
                if fill_cache is not None:
                    fill_cache.put_conv(i, u, dil)
                if with_grad:
                    cache[f"c_{i}"] = (gate, val, sig, c, s2, sig2, taps, shifts)
            else:
                qkv = h @ P[f"qkv_W_{i}"] + P[f"qkv_b_{i}"]
                q, kk, v = np.split(qkv, 3, axis=-1)
                H, dh = cfg.n_heads, cfg.head_dim
                q = _rope_apply(q.reshape(B, T, H, dh).transpose(0, 2, 1, 3), cos, sin)
                kr = _rope_apply(kk.reshape(B, T, H, dh).transpose(0, 2, 1, 3), cos, sin)
                v = v.reshape(B, T, H, dh).transpose(0, 2, 1, 3)
                scale = np.float32(dh ** -0.5)
                mask = _causal_mask(T, pad)
                p_ = _softmax_last((q @ kr.transpose(0, 1, 3, 2)) * scale + mask)
                ao = (p_ @ v).transpose(0, 2, 1, 3).reshape(B, T, d)
                o = ao @ P[f"o_W_{i}"] + P[f"o_b_{i}"]
                x = x + o
                if fill_cache is not None:
                    fill_cache.put_attn(i, kr, v)
                if with_grad:
                    cache[f"a_{i}"] = (q, kr, v, p_, ao, scale, mask)
            dil_i += 1 if kind == "conv" else 0

        xf, rf = _rmsnorm(x, P["g_final"], cfg.eps)
        logits = xf @ P["emb"].T
        if fill_cache is not None:
            fill_cache.length = T
        if with_grad:
            cache["xf"], cache["rf"], cache["xfin"] = xf, rf, x
            return logits, cache
        return logits, cache


# ---------------------------------------------------------------------- #
# loss + backward
# ---------------------------------------------------------------------- #
def forward_backward(net: MicroNet, ids: np.ndarray, target: np.ndarray,
                     pad: np.ndarray) -> tuple[float, dict[str, np.ndarray]]:
    """next-token 交差エントロピーと全パラメータ勾配。

    ids/target/pad は (B,T)。 pad=1 の位置だけを損失に含める。
    """
    cfg, P = net.cfg, net.params
    G: dict[str, np.ndarray] = {}
    B, T = ids.shape
    d, V = cfg.d_model, cfg.n_vocab
    logits, cache = net.forward(ids, pad=pad, with_grad=True)

    valid = pad.astype(np.float32)
    nvalid = max(1.0, float(valid.sum()))
    e = np.exp(logits - logits.max(-1, keepdims=True))
    probs = e / e.sum(-1, keepdims=True)
    tgt_prob = np.take_along_axis(probs, target[:, :, None], axis=-1)[:, :, 0]
    loss = float((-np.log(np.clip(tgt_prob, 1e-9, None)) * valid).sum() / nvalid)
    scale = (valid / nvalid)[:, :, None]
    dlogits = ((probs - np.eye(V, dtype=np.float32)[target]) * scale).astype(np.float32)

    # ---- final RMSNorm + tied embedding --------------------------------
    xf, rf, xfin = cache["xf"], cache["rf"], cache["xfin"]
    G["emb"] = (dlogits.reshape(-1, V).T @ xf.reshape(-1, d)).astype(np.float32)
    dx = (dlogits @ P["emb"]) * pad[:, :, None]
    gfin = P["g_final"]
    xn = xfin / rf                                  # = RMSNorm の入力側 / r
    G["g_final"] = (dx * xn).sum((0, 1)).astype(np.float32)
    c3 = (dx * gfin * xfin).sum(-1, keepdims=True)
    dx = dx * (gfin / rf) - xn * (c3 / (rf * rf * d))

    # ---- blocks ---------------------------------------------------------
    for i in reversed(range(cfg.n_layers)):
        dx_res = dx                     # 残差（恒等）経路で上の層から来た勾配
        kind = cfg.blocks[i]
        h, r, xin = cache[f"h_{i}"], cache[f"r_{i}"], cache[f"xin_{i}"]
        if kind == "conv":
            gate, val, sig, c, s2, sig2, taps, shifts = cache[f"c_{i}"]
            W_o = P[f"out_W_{i}"]
            Wc = P[f"conv_W_{i}"]
            G[f"out_W_{i}"] = (s2.reshape(-1, d).T @ dx.reshape(-1, d)).astype(np.float32)
            G[f"out_b_{i}"] = dx.sum((0, 1)).astype(np.float32)
            dc = (dx @ W_o.T) * (sig2 * (1.0 + c * (1.0 - sig2)))
            du = np.zeros_like(dx)
            gw = np.zeros_like(Wc)
            gb = np.zeros(d, dtype=np.float32)
            for j, sh in enumerate(shifts):
                du += _lead(dc * Wc[None, :, j], sh)
                gw[:, j] = (dc * taps[j]).sum((0, 1))
            gb = dc.sum((0, 1))
            G[f"conv_W_{i}"] = gw.astype(np.float32)
            G[f"conv_b_{i}"] = gb.astype(np.float32)
            dval = du * (gate * sig)              # ∂(val*silu(gate))/∂val
            dgate = du * val * (sig * (1.0 + gate * (1.0 - sig)))
            dgu = np.concatenate([dgate, dval], axis=-1)
            G[f"gu_W_{i}"] = (h.reshape(-1, d).T @ dgu.reshape(-1, 2 * d)).astype(np.float32)
            G[f"gu_b_{i}"] = dgu.sum((0, 1)).astype(np.float32)
            dh = dgu @ P[f"gu_W_{i}"].T
        else:
            q, kr, v, p_, ao, scale_att, mask = cache[f"a_{i}"]
            H, dh_ = cfg.n_heads, cfg.head_dim
            dao = (dx @ P[f"o_W_{i}"].T).reshape(B, T, H, dh_).transpose(0, 2, 1, 3)
            G[f"o_W_{i}"] = (ao.reshape(-1, d).T @ dx.reshape(-1, d)).astype(np.float32)
            G[f"o_b_{i}"] = dx.sum((0, 1)).astype(np.float32)
            dv = p_.transpose(0, 1, 3, 2) @ dao
            dp = dao @ v.transpose(0, 1, 3, 2)
            ds = (p_ * (dp - (p_ * dp).sum(-1, keepdims=True))).astype(np.float32)
            dq = (ds @ kr) * scale_att
            dk = (ds.transpose(0, 1, 3, 2) @ q) * scale_att
            cos, sin = net.rope_freqs(T)
            dq_raw = _rope_grad(dq, cos, sin).transpose(0, 2, 1, 3).reshape(B, T, d)
            dk_raw = _rope_grad(dk, cos, sin).transpose(0, 2, 1, 3).reshape(B, T, d)
            dqkv = np.concatenate([dq_raw, dk_raw, dv.transpose(0, 2, 1, 3).reshape(B, T, d)], axis=-1)
            G[f"qkv_W_{i}"] = (h.reshape(-1, d).T @ dqkv.reshape(-1, 3 * d)).astype(np.float32)
            G[f"qkv_b_{i}"] = dqkv.sum((0, 1)).astype(np.float32)
            dh = dqkv @ P[f"qkv_W_{i}"].T
        gg = P[f"g_{i}"]
        G[f"g_{i}"] = (dh * xin / r).sum((0, 1)).astype(np.float32)
        c3 = (dh * gg * xin).sum(-1, keepdims=True)
        dx = dx_res + (dh * (gg / r) - (xin / r) * (c3 / (r * r * d)))

    # ---- 入力側埋め込みの勾配（重み共有なので出力側の勾配に足す） ----------
    dx0 = (dx * pad[:, :, None]).reshape(-1, d).astype(np.float32)
    flat = ids.reshape(-1)
    demb = np.zeros((V, d), dtype=np.float32)
    for j in range(d):
        demb[:, j] = np.bincount(flat, weights=dx0[:, j], minlength=V)
    G["emb"] = G["emb"] + demb

    return loss, G


def clip_grads(G: dict[str, np.ndarray], norm: float = 1.0) -> float:
    """global norm クリッピング。返り値はクリップ前のノルム。"""
    tn = float(np.sqrt(sum(float(np.dot(v.ravel(), v.ravel())) for v in G.values())))
    if tn > norm > 0:
        k = norm / max(tn, 1e-12)
        for key in G:
            G[key] = G[key] * k
    return tn


# ---------------------------------------------------------------------- #
# 逐次デコード（KV キャッシュ）
# ---------------------------------------------------------------------- #
class DecodeCache:
    """1 文字ずつ進めるための状態。

    * conv ブロック … 直近 `span` 個の u（ゲート後・畳み込み前）を環状に保持
    * attn ブロック … RoPE 適用済みの K / V を全位置ぶん保持

    `prefill()` で prompt を一括計算してから `step()` を呼ぶと、
    1 文字あたりの計算量が O(T) → O(1) になり、生成が 10 倍以上速くなります。
    """

    def __init__(self, net: "MicroNet", batch: int = 1):
        cfg = net.cfg
        self.cfg = cfg
        self.batch = int(batch)
        self.d = cfg.d_model
        self.length = 0                       # 既に確定しているトークン数
        self._conv: dict[int, tuple[int, np.ndarray]] = {}
        self._attn: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        dil_i = 0
        for i, kind in enumerate(cfg.blocks):
            if kind == "conv":
                dil = cfg.dilations[dil_i % len(cfg.dilations)]
                span = (cfg.conv_kernel - 1) * dil + 1
                self._conv[i] = (span, np.zeros((span, self.batch, self.d), dtype=np.float32))
                dil_i += 1
            else:
                dh = cfg.head_dim
                z = np.zeros((self.batch, cfg.n_heads, 0, dh), dtype=np.float32)
                self._attn[i] = (z, z.copy())

    # -- forward() から呼ばれる書き込み -------------------------------- #
    def put_conv(self, i: int, u: np.ndarray, dil: int) -> None:
        span, buf = self._conv[i]                         # buf: (span, B, d)
        take = np.transpose(u[:, -span:, :], (1, 0, 2))   # (n, B, d) … 時刻の末尾 span 個
        n = take.shape[0]
        if n < span:                                      # 先頭はゼロ埋め（＝未来を見ていない）
            buf[: span - n] = 0.0
            buf[span - n:] = take
        else:
            buf[:] = take

    def put_attn(self, i: int, k: np.ndarray, v: np.ndarray) -> None:
        self._attn[i] = (np.ascontiguousarray(k, dtype=np.float32),
                         np.ascontiguousarray(v, dtype=np.float32))

    # -- 逐次デコード -------------------------------------------------- #
    def step(self, net: "MicroNet", ids: np.ndarray) -> np.ndarray:
        """次の 1 トークン（(B,) または (B,1)）から logits (B,V) を得る。"""
        cfg, P = net.cfg, net.params
        ids = np.asarray(ids, dtype=np.int64).reshape(-1)
        B = ids.shape[0]
        if B != self.batch:
            raise ValueError(f"batch 不一致: cache={self.batch} ids={B}")
        d, H, dh = self.d, cfg.n_heads, cfg.head_dim
        t = self.length
        cos, sin = net.rope_freqs(t + 1)
        cos1, sin1 = cos[t:t + 1], sin[t:t + 1]
        x = P["emb"][ids][:, None, :]                      # (B,1,d)
        dil_i = 0
        for i, kind in enumerate(cfg.blocks):
            h = _rmsnorm(x, P[f"g_{i}"], cfg.eps)[0][:, 0]        # (B,d)
            if kind == "conv":
                gu = h @ P[f"gu_W_{i}"] + P[f"gu_b_{i}"]    # (B,2d)
                gate, val = gu[:, :d], gu[:, d:]
                sg, _s = _silu(gate)
                u = val * sg                                # (B,d)
                span, buf = self._conv[i]
                buf[:-1] = buf[1:]
                buf[-1] = u
                k = cfg.conv_kernel
                dil = cfg.dilations[dil_i % len(cfg.dilations)]
                Wc = P[f"conv_W_{i}"]
                c = np.broadcast_to(P[f"conv_b_{i}"], (B, d)).copy()
                for j in range(k):
                    sh = (k - 1 - j) * dil
                    idx = span - 1 - sh
                    if idx < 0:
                        continue                            # 未来 → ゼロ
                    c = c + buf[idx] * Wc[None, :, j]
                s2, _s2 = _silu(c)
                o = s2 @ P[f"out_W_{i}"] + P[f"out_b_{i}"]
                x = x + o[:, None, :]
                dil_i += 1
            else:
                qkv = h @ P[f"qkv_W_{i}"] + P[f"qkv_b_{i}"]  # (B,3d)
                q, kk, v = qkv[:, :d], qkv[:, d:2 * d], qkv[:, 2 * d:]
                q = _rope_apply(q.reshape(B, 1, H, dh).transpose(0, 2, 1, 3), cos1, sin1)
                kr = _rope_apply(kk.reshape(B, 1, H, dh).transpose(0, 2, 1, 3), cos1, sin1)
                vv = v.reshape(B, 1, H, dh).transpose(0, 2, 1, 3)
                K0, V0 = self._attn[i]
                K = np.concatenate([K0, kr], axis=2) if K0.shape[2] else kr
                V = np.concatenate([V0, vv], axis=2) if V0.shape[2] else vv
                self._attn[i] = (K, V)
                scale = np.float32(dh ** -0.5)
                att = (q @ K.transpose(0, 1, 3, 2)) * scale  # (B,H,1,T+1)
                p_ = _softmax_last(att)
                ao = (p_ @ V).transpose(0, 2, 1, 3).reshape(B, 1, d)
                o = ao[:, 0] @ P[f"o_W_{i}"] + P[f"o_b_{i}"]
                x = x + o[:, None, :]
        xf, _rf = _rmsnorm(x, P["g_final"], cfg.eps)
        self.length = t + 1
        return xf[:, 0] @ P["emb"].T

    def prefill(self, net: "MicroNet", ids: np.ndarray) -> np.ndarray:
        """prompt を一括計算してキャッシュを埋め、最後の位置の logits を返す。"""
        ids = np.asarray(ids, dtype=np.int64)
        if ids.ndim == 1:
            ids = ids[None, :]
        self.batch = ids.shape[0]
        logits, _cache = net.forward(ids, with_grad=False, fill_cache=self)
        self.length = int(ids.shape[1])
        return logits[:, -1, :]

    def clone_empty(self, net: "MicroNet") -> "DecodeCache":
        return DecodeCache(net, batch=self.batch)
