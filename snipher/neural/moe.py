"""MoE (Mixture-of-Experts) + GQA (Grouped-Query Attention) の極小トランスフォーマ。

v8 の「本気で賢い」アーキテクチャの中核。ユーザー設計図の ①②③④ をそのまま
実装したもの:

    x = Embedding(chars)
    for block in blocks:                       # attn / moe を交互に積む
        if attn:  # ② RoPE + ① GQA（KV ヘッドを集約し長文の KV キャッシュを 1/N に）
            h = RMSNorm(x)
            q = RoPE(h @ W_q)                  # 全ヘッド
            k, v = RoPE(h @ W_kv)              # n_kv_heads 個だけ（GQA）
            k, v = repeat_interleave(k, v, H // KVH)
            x = x + CausalSoftmax(q kᵗ/√d) v @ W_o
        if moe:   # ③ SwiGLU × Mixture-of-Experts（ルータが Top-K を動的選択）
            h = RMSNorm(x)
            g = softmax(h @ W_router)          # 学習時は dense（全 expert の加重和）
            for e in experts:
                x = x + g[..., e] * SwiGLU_e(h)
    logits = RMSNorm(x) @ Embeddingᵀ           # 重み共有

* 正規化は ④ RMSNorm（平均減算なし・二乗平均平方根のみ）
* 位置は RoPE（相対距離を回転行列で埋め込む）
* ルーティングは学習時に dense（勾配が全経路に流れる soft MoE）、推論時に
  Top-K の hard routing（計算コストを 1/K に）。スパース性は推論時に現れる。

``MicroNet``（nn.py）とは独立した自己完結モジュールで、既存の内蔵コアの
挙動を一切変えない。重みの保存・読み込みは ``store.py`` を共有する
（``kind: moe`` ヘッダで ``MoENet`` を再構成する）。

依存は numpy のみ。forward / backward は float64 の中心差分で勾配検証済み
（tests/test_moe.py）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .nn import (  # 既存コアと同じ RMSNorm / RoPE / softmax の実装を共有
    _causal_mask,
    _rope_apply,
    _rope_grad,
    _rmsnorm,
    _silu,
    _softmax_last,
)

# 保存形式の識別子（store.py の load() がこれを見て MoENet を再構成する）
KIND = "moe"


@dataclass
class MoEConfig:
    n_vocab: int = 640
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 8
    n_kv_heads: int = 2            # GQA: KV ヘッド数をクエリヘッドより少なくする
    n_experts: int = 4
    top_k: int = 2
    expert_dim: int | None = None  # None → d_model * 2
    max_pos: int = 320
    rope_base: float = 10000.0
    eps: float = 1e-5
    blocks: tuple[str, ...] | None = None   # None → attn/moe を交互

    def __post_init__(self) -> None:
        if self.blocks is None:
            self.blocks = tuple("attn" if i % 2 == 0 else "moe" for i in range(self.n_layers))
        self.blocks = tuple(self.blocks)
        if self.d_model % self.n_heads:
            raise ValueError("d_model は n_heads の倍数である必要があります")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads は n_kv_heads の倍数である必要があります (GQA)")
        if self.top_k > self.n_experts:
            raise ValueError("top_k は n_experts 以下である必要があります")
        if self.expert_dim is None:
            self.expert_dim = self.d_model * 2

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def kv_dim(self) -> int:
        return self.n_kv_heads * self.head_dim

    @property
    def group(self) -> int:
        """1 つの KV ヘッドを共有するクエリヘッド数。"""
        return self.n_heads // self.n_kv_heads

    def to_dict(self) -> dict:
        return {
            "n_vocab": self.n_vocab,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "n_kv_heads": self.n_kv_heads,
            "n_experts": self.n_experts,
            "top_k": self.top_k,
            "expert_dim": self.expert_dim,
            "max_pos": self.max_pos,
            "rope_base": self.rope_base,
            "blocks": list(self.blocks),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MoEConfig":
        keep = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**keep)


def _softmax_rows(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def _topk_softmax(r: np.ndarray, k: int) -> np.ndarray:
    """推論用のハードルーティング。Top-K 以外は 0 にして再正規化。"""
    E = r.shape[-1]
    if k >= E:
        return _softmax_rows(r)
    flat = r.reshape(-1, E)
    top = np.argpartition(-flat, k, axis=-1)[:, :k]
    out = np.full_like(flat, -np.inf)
    rows = np.arange(flat.shape[0])[:, None]
    out[rows, top] = flat[rows, top]
    return _softmax_rows(out).reshape(r.shape)


# ---------------------------------------------------------------------- #
# モデル
# ---------------------------------------------------------------------- #
@dataclass
class MoENet:
    """GQA + MoE(SwiGLU) の極小トランスフォーマ。重みは float32 の dict。"""

    cfg: MoEConfig
    params: dict[str, np.ndarray] = field(default_factory=dict)

    KIND = "moe"

    # ------------------------------------------------------------------ #
    @classmethod
    def random(cls, cfg: MoEConfig, seed: int = 0, emb_std: float = 0.08) -> "MoENet":
        rng = np.random.default_rng(seed)
        d = cfg.d_model
        m = cfg.expert_dim
        E = cfg.n_experts
        p: dict[str, np.ndarray] = {}
        p["emb"] = rng.normal(0.0, emb_std, size=(cfg.n_vocab, d)).astype(np.float32)
        p["emb"][0] = 0.0
        p["g_final"] = np.ones(d, dtype=np.float32)
        for i, kind in enumerate(cfg.blocks):
            p[f"g_{i}"] = np.ones(d, dtype=np.float32)
            if kind == "attn":
                w = d + 2 * cfg.kv_dim
                p[f"qkv_W_{i}"] = rng.normal(0, d ** -0.5, size=(d, w)).astype(np.float32)
                p[f"qkv_b_{i}"] = np.zeros(w, dtype=np.float32)
                p[f"o_W_{i}"] = np.zeros((d, d), dtype=np.float32)
                p[f"o_b_{i}"] = np.zeros(d, dtype=np.float32)
            elif kind == "moe":
                p[f"moe_{i}_router_W"] = rng.normal(0, d ** -0.5, size=(d, E)).astype(np.float32)
                p[f"moe_{i}_router_b"] = np.zeros(E, dtype=np.float32)
                for e in range(E):
                    p[f"moe_{i}_e{e}_up_W"] = rng.normal(0, d ** -0.5, size=(d, m)).astype(np.float32)
                    p[f"moe_{i}_e{e}_gate_W"] = rng.normal(0, d ** -0.5, size=(d, m)).astype(np.float32)
                    p[f"moe_{i}_e{e}_down_W"] = np.zeros((m, d), dtype=np.float32)
                    p[f"moe_{i}_e{e}_up_b"] = np.zeros(m, dtype=np.float32)
                    p[f"moe_{i}_e{e}_gate_b"] = np.zeros(m, dtype=np.float32)
                    p[f"moe_{i}_e{e}_down_b"] = np.zeros(d, dtype=np.float32)
            else:
                raise ValueError(f"未知のブロック種別: {kind}")
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
                fill_cache: "MoEDecodeCache | None" = None, hard: bool = False):
        """(B,T) の全位置を一度に計算する。

        ``hard=True`` のとき推論用の Top-K ルーティング（計算量 1/K）。
        学習時は ``hard=False``（dense soft MoE）で全 expert に勾配が流れる。
        """
        cfg, P = self.cfg, self.params
        B, T = ids.shape
        d = cfg.d_model
        H, KVH, dh = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        rep = cfg.group
        x = P["emb"][ids]
        if pad is not None:
            x = x * pad[:, :, None]
        cache: dict[str, object] = {}
        cos, sin = self.rope_freqs(T)

        for i, kind in enumerate(cfg.blocks):
            h, r = _rmsnorm(x, P[f"g_{i}"], cfg.eps)
            if with_grad:
                cache[f"h_{i}"], cache[f"r_{i}"], cache[f"xin_{i}"] = h, r, x
            if kind == "attn":
                qkv = h @ P[f"qkv_W_{i}"] + P[f"qkv_b_{i}"]
                q = qkv[:, :, :d]
                k = qkv[:, :, d:d + cfg.kv_dim]
                v = qkv[:, :, d + cfg.kv_dim:]
                q = _rope_apply(q.reshape(B, T, H, dh).transpose(0, 2, 1, 3), cos, sin)
                kr = _rope_apply(k.reshape(B, T, KVH, dh).transpose(0, 2, 1, 3), cos, sin)
                vr = v.reshape(B, T, KVH, dh).transpose(0, 2, 1, 3)
                krep = np.repeat(kr, rep, axis=1)          # (B,H,T,dh)
                vrep = np.repeat(vr, rep, axis=1)
                scale = np.float32(dh ** -0.5)
                mask = _causal_mask(T, pad)
                p_ = _softmax_last((q @ krep.transpose(0, 1, 3, 2)) * scale + mask)
                ao = (p_ @ vrep).transpose(0, 2, 1, 3).reshape(B, T, d)
                o = ao @ P[f"o_W_{i}"] + P[f"o_b_{i}"]
                x = x + o
                if fill_cache is not None:
                    fill_cache.put_attn(i, kr, vr)
                if with_grad:
                    cache[f"a_{i}"] = (q, kr, vr, p_, ao, scale, rep)
            elif kind == "moe":
                router = h @ P[f"moe_{i}_router_W"] + P[f"moe_{i}_router_b"]  # (B,T,E)
                g = _topk_softmax(router, cfg.top_k) if hard else _softmax_rows(router)
                o = np.zeros_like(x)
                expert_acts: list[tuple] = []
                for e in range(cfg.n_experts):
                    up = h @ P[f"moe_{i}_e{e}_up_W"] + P[f"moe_{i}_e{e}_up_b"]
                    gate = h @ P[f"moe_{i}_e{e}_gate_W"] + P[f"moe_{i}_e{e}_gate_b"]
                    sg, sig = _silu(gate)
                    act = up * sg
                    eo = act @ P[f"moe_{i}_e{e}_down_W"] + P[f"moe_{i}_e{e}_down_b"]
                    o = o + g[:, :, e:e + 1] * eo
                    expert_acts.append((up, gate, sg, sig, act, eo))
                x = x + o
                if with_grad:
                    cache[f"m_{i}"] = (router, g, expert_acts)
            else:
                raise ValueError(f"未知のブロック種別: {kind}")

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
def forward_backward(net: MoENet, ids: np.ndarray, target: np.ndarray,
                     pad: np.ndarray | None = None) -> tuple[float, dict[str, np.ndarray]]:
    """損失と全パラメータの勾配を返す（dense soft MoE で学習する）。"""
    cfg, P = net.cfg, net.params
    B, T = ids.shape
    V, d = cfg.n_vocab, cfg.d_model
    H, KVH, hd = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
    rep = cfg.group
    G: dict[str, np.ndarray] = {k: np.zeros_like(v) for k, v in P.items()}
    if pad is None:
        pad = np.ones((B, T), dtype=np.float32)
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
    xn = xfin / rf
    G["g_final"] = (dx * xn).sum((0, 1)).astype(np.float32)
    c3 = (dx * gfin * xfin).sum(-1, keepdims=True)
    dx = dx * (gfin / rf) - xn * (c3 / (rf * rf * d))

    # ---- blocks ---------------------------------------------------------
    for i in reversed(range(len(cfg.blocks))):
        dx_res = dx
        kind = cfg.blocks[i]
        h, r, xin = cache[f"h_{i}"], cache[f"r_{i}"], cache[f"xin_{i}"]
        if kind == "attn":
            q, kr, vr, p_, ao, scale_att, _rep = cache[f"a_{i}"]
            dao = (dx @ P[f"o_W_{i}"].T).reshape(B, T, H, hd).transpose(0, 2, 1, 3)
            G[f"o_W_{i}"] = (ao.reshape(-1, d).T @ dx.reshape(-1, d)).astype(np.float32)
            G[f"o_b_{i}"] = dx.sum((0, 1)).astype(np.float32)
            # v は rep 回 repeat されているので、勾配はグループで足し合わせる
            dv = p_.transpose(0, 1, 3, 2) @ dao                # (B,H,T,hd)
            dp = dao @ np.repeat(vr, rep, axis=1).transpose(0, 1, 3, 2)
            ds = (p_ * (dp - (p_ * dp).sum(-1, keepdims=True))).astype(np.float32)
            dq = (ds @ np.repeat(kr, rep, axis=1)) * scale_att
            dk = (ds.transpose(0, 1, 3, 2) @ q) * scale_att
            cos, sin = net.rope_freqs(T)
            dq_raw = _rope_grad(dq, cos, sin).transpose(0, 2, 1, 3).reshape(B, T, d)
            dk_full = _rope_grad(dk, cos, sin).transpose(0, 2, 1, 3).reshape(B, T, H, hd)
            dv_full = dv.transpose(0, 2, 1, 3).reshape(B, T, H, hd)
            dk_kv = dk_full.reshape(B, T, KVH, rep, hd).sum(axis=3).reshape(B, T, cfg.kv_dim)
            dv_kv = dv_full.reshape(B, T, KVH, rep, hd).sum(axis=3).reshape(B, T, cfg.kv_dim)
            dqkv = np.concatenate([dq_raw, dk_kv, dv_kv], axis=-1)
            G[f"qkv_W_{i}"] = (h.reshape(-1, d).T @ dqkv.reshape(-1, d + 2 * cfg.kv_dim)).astype(np.float32)
            G[f"qkv_b_{i}"] = dqkv.sum((0, 1)).astype(np.float32)
            dh = dqkv @ P[f"qkv_W_{i}"].T
        else:  # moe
            router, g, expert_acts = cache[f"m_{i}"]
            E = cfg.n_experts
            # d(out)/d(g_e) = dx · eo_e（eo_e は (B,T,d) ベクトル）
            dg = np.zeros_like(g)
            for e in range(E):
                _up, _gate, _sg, _sig, _act, eo = expert_acts[e]
                dg[:, :, e] = (dx * eo).sum(-1)
            # dense softmax の逆伝播
            dr = (g * (dg - (g * dg).sum(-1, keepdims=True))).astype(np.float32)
            G[f"moe_{i}_router_W"] = (h.reshape(-1, d).T @ dr.reshape(-1, E)).astype(np.float32)
            G[f"moe_{i}_router_b"] = dr.sum((0, 1)).astype(np.float32)
            dh = dr @ P[f"moe_{i}_router_W"].T
            for e in range(E):
                up, gate, sg, sig, act, eo = expert_acts[e]
                deo = dx * g[:, :, e:e + 1]
                G[f"moe_{i}_e{e}_down_W"] = (act.reshape(-1, cfg.expert_dim).T @ deo.reshape(-1, d)).astype(np.float32)
                G[f"moe_{i}_e{e}_down_b"] = deo.sum((0, 1)).astype(np.float32)
                dact = deo @ P[f"moe_{i}_e{e}_down_W"].T
                dup = dact * sg
                dgate = dact * up * (sig * (1.0 + gate * (1.0 - sig)))
                G[f"moe_{i}_e{e}_up_W"] = (h.reshape(-1, d).T @ dup.reshape(-1, cfg.expert_dim)).astype(np.float32)
                G[f"moe_{i}_e{e}_up_b"] = dup.sum((0, 1)).astype(np.float32)
                G[f"moe_{i}_e{e}_gate_W"] = (h.reshape(-1, d).T @ dgate.reshape(-1, cfg.expert_dim)).astype(np.float32)
                G[f"moe_{i}_e{e}_gate_b"] = dgate.sum((0, 1)).astype(np.float32)
                dh = dh + dup @ P[f"moe_{i}_e{e}_up_W"].T + dgate @ P[f"moe_{i}_e{e}_gate_W"].T
        gg = P[f"g_{i}"]
        G[f"g_{i}"] = (dh * xin / r).sum((0, 1)).astype(np.float32)
        c3 = (dh * gg * xin).sum(-1, keepdims=True)
        dx = dx_res + (dh * (gg / r) - (xin / r) * (c3 / (r * r * d)))

    # ---- 入力側埋め込みの勾配（重み共有） --------------------------------
    dx0 = (dx * pad[:, :, None]).reshape(-1, d).astype(np.float32)
    flat = ids.reshape(-1)
    demb = np.zeros((V, d), dtype=np.float32)
    for j in range(d):
        demb[:, j] = np.bincount(flat, weights=dx0[:, j], minlength=V)
    G["emb"] = G["emb"] + demb

    return loss, G


def clip_grads(G: dict[str, np.ndarray], norm: float = 1.0) -> float:
    tn = float(np.sqrt(sum(float(np.dot(v.ravel(), v.ravel())) for v in G.values())))
    if tn > norm > 0:
        k = norm / max(tn, 1e-12)
        for key in G:
            G[key] = G[key] * k
    return tn


# ---------------------------------------------------------------------- #
# 逐次デコード（KV キャッシュ + Top-K ルーティング）
# ---------------------------------------------------------------------- #
class MoEDecodeCache:
    """1 トークンずつ進めるための状態。

    * attn ブロック … RoPE 適用済みの K / V を全位置ぶん保持（GQA なので KV が 1/N）
    * moe ブロック … 状態なし（点ごと FFN）。Top-K で expert の K/E だけ計算する。
    """

    def __init__(self, net: "MoENet", batch: int = 1):
        cfg = net.cfg
        self.cfg = cfg
        self.batch = int(batch)
        self.d = cfg.d_model
        self.length = 0
        self._attn: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        dh = cfg.head_dim
        z = np.zeros((self.batch, cfg.n_kv_heads, 0, dh), dtype=np.float32)
        for i, kind in enumerate(cfg.blocks):
            if kind == "attn":
                self._attn[i] = (z.copy(), z.copy())

    def put_attn(self, i: int, k: np.ndarray, v: np.ndarray) -> None:
        self._attn[i] = (np.ascontiguousarray(k, dtype=np.float32),
                         np.ascontiguousarray(v, dtype=np.float32))

    def _moe_step(self, net: "MoENet", i: int, h: np.ndarray, x: np.ndarray) -> np.ndarray:
        """点ごとの MoE を Top-K で計算する（選択された expert だけを走らせる）。"""
        cfg, P = net.cfg, net.params
        E, m = cfg.n_experts, cfg.expert_dim
        router = h @ P[f"moe_{i}_router_W"] + P[f"moe_{i}_router_b"]   # (B,E)
        g = _topk_softmax(router, cfg.top_k)
        o = np.zeros((h.shape[0], self.d), dtype=np.float32)
        active = np.where(g > 0)
        for b in range(h.shape[0]):
            for e in np.unique(active[1][active[0] == b]):
                w = float(g[b, e])
                up = h[b] @ P[f"moe_{i}_e{e}_up_W"] + P[f"moe_{i}_e{e}_up_b"]
                gate = h[b] @ P[f"moe_{i}_e{e}_gate_W"] + P[f"moe_{i}_e{e}_gate_b"]
                sg, _sig = _silu(gate)
                o[b] += w * (up * sg @ P[f"moe_{i}_e{e}_down_W"] + P[f"moe_{i}_e{e}_down_b"])
        return o

    def step(self, net: "MoENet", ids: np.ndarray) -> np.ndarray:
        cfg, P = net.cfg, net.params
        ids = np.asarray(ids, dtype=np.int64).reshape(-1)
        B = ids.shape[0]
        if B != self.batch:
            raise ValueError(f"batch 不一致: cache={self.batch} ids={B}")
        d, H, KVH, dh = self.d, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        rep = cfg.group
        t = self.length
        cos, sin = net.rope_freqs(t + 1)
        cos1, sin1 = cos[t:t + 1], sin[t:t + 1]
        x = P["emb"][ids][:, None, :]                        # (B,1,d)
        for i, kind in enumerate(cfg.blocks):
            h = _rmsnorm(x, P[f"g_{i}"], cfg.eps)[0][:, 0]   # (B,d)
            if kind == "attn":
                qkv = h @ P[f"qkv_W_{i}"] + P[f"qkv_b_{i}"]
                q = qkv[:, :d]
                k = qkv[:, d:d + cfg.kv_dim]
                v = qkv[:, d + cfg.kv_dim:]
                q = _rope_apply(q.reshape(B, 1, H, dh).transpose(0, 2, 1, 3), cos1, sin1)
                kr = _rope_apply(k.reshape(B, 1, KVH, dh).transpose(0, 2, 1, 3), cos1, sin1)
                vr = v.reshape(B, 1, KVH, dh).transpose(0, 2, 1, 3)
                K0, V0 = self._attn[i]
                K = np.concatenate([K0, kr], axis=2) if K0.shape[2] else kr
                V = np.concatenate([V0, vr], axis=2) if V0.shape[2] else vr
                self._attn[i] = (K, V)
                scale = np.float32(dh ** -0.5)
                att = (q @ np.repeat(K, rep, axis=1).transpose(0, 1, 3, 2)) * scale
                p_ = _softmax_last(att)
                ao = (p_ @ np.repeat(V, rep, axis=1)).transpose(0, 2, 1, 3).reshape(B, 1, d)
                o = ao[:, 0] @ P[f"o_W_{i}"] + P[f"o_b_{i}"]
                x = x + o[:, None, :]
            elif kind == "moe":
                o = self._moe_step(net, i, h, x[:, 0])
                x = x + o[:, None, :]
        xf, _rf = _rmsnorm(x, P["g_final"], cfg.eps)
        self.length = t + 1
        return xf[:, 0] @ P["emb"].T

    def prefill(self, net: "MoENet", ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(ids, dtype=np.int64)
        if ids.ndim == 1:
            ids = ids[None, :]
        self.batch = ids.shape[0]
        logits, _cache = net.forward(ids, with_grad=False, fill_cache=self, hard=False)
        self.length = int(ids.shape[1])
        return logits[:, -1, :]

    def clone_empty(self, net: "MoENet") -> "MoEDecodeCache":
        return MoEDecodeCache(net, batch=self.batch)
