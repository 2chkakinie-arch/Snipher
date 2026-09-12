"""重みの保存・読み込み（行単位 int8 量子化 + zlib/json ヘッダ）。

Snipher の内蔵ニューラルコアは **リポジトリ同梱**で即ロードできることが要件です
（Vercel のようなサーバーレスでは実行時に 731MB をダウンロードできないため）。
そこで次の形式にします:

    npz:  header(bytes: _MAGIC + zlib(json)) + "q/<name>" int8 配列 + scales 平坦配列

  * 2 次元の重みは「出力列ごと」または「語彙行ごと」に 1 スケール（行別量子化）
  * 1 次元（bias / norm の gain）は配列全体で 1 スケール
  * ロード時に 1 度だけ float32 へ逆量子化して推論速度を稼ぐ
"""

from __future__ import annotations

import json
import zlib
from pathlib import Path

import numpy as np

from .nn import MicroNet, NNConfig

_MAGIC = b"SNPNCR01"


def _axis_for(name: str, arr: np.ndarray) -> int:
    """量子化グループの軸。-1 = 配列全体で 1 スケール。"""
    if arr.ndim < 2:
        return -1
    if name == "emb" or name.startswith("conv_W"):
        return 1        # 行（語彙エントリ / チャネル）ごと
    return 0            # 2D 射影は列（出力次元）ごと


def quantize(params: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    q: dict[str, np.ndarray] = {}
    meta: dict[str, dict] = {}
    for name, a in params.items():
        a32 = np.asarray(a, dtype=np.float32)
        axis = _axis_for(name, a32)
        if axis < 0:
            amax = float(np.abs(a32).max()) or 1.0
            scale = amax / 127.0
            q[name] = np.round(a32 / scale).clip(-127, 127).astype(np.int8)
            meta[name] = {"shape": list(a32.shape), "axis": -1, "scale": np.float32(scale)}
            continue
        amax = np.abs(a32).max(axis=axis, keepdims=True)
        amax = np.where(amax < 1e-8, 1.0, amax)
        scale = (amax / 127.0)
        q[name] = np.round(a32 / scale).clip(-127, 127).astype(np.int8)
        meta[name] = {"shape": list(a32.shape), "axis": int(axis), "scale": scale.ravel()}
    return q, meta


def dequantize(q: dict[str, np.ndarray], meta: dict[str, dict]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for name, qi in q.items():
        m = meta[name]
        a = qi.astype(np.float32)
        s = m["scale"]
        if m["axis"] == -1:
            out[name] = a * np.float32(s)
            continue
        # axis を削減してグループ化しているので、スケールの本数はもう一方の軸に対応する
        grp = 1 - int(m["axis"]) if a.ndim == 2 else 0
        shape = [-1 if i == grp else 1 for i in range(a.ndim)]
        out[name] = (a * np.asarray(s, dtype=np.float32).reshape(shape)).astype(np.float32)
    return out


def save(path: str | Path, net: MicroNet, vocab: list[str], extra: dict | None = None) -> dict:
    q, meta = quantize(net.params)
    flat: list[np.ndarray] = []
    slices: dict[str, list[int]] = {}
    off = 0
    for name, m in meta.items():
        s = np.asarray(m["scale"], dtype=np.float32).ravel()
        slices[name] = [off, int(s.size)]
        flat.append(s)
        off += int(s.size)
    scales = np.concatenate(flat) if flat else np.zeros(0, dtype=np.float32)
    header = {
        "format": 2,
        "config": net.cfg.to_dict(),
        "vocab": vocab,
        "meta": {n: {"shape": m["shape"], "axis": m["axis"], "scale_slice": slices[n]} for n, m in meta.items()},
        "extra": extra or {},
    }
    blob = _MAGIC + zlib.compress(json.dumps(header, ensure_ascii=False).encode("utf-8"), 9)
    out = Path(path).with_suffix(".npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, header=np.frombuffer(blob, dtype=np.uint8), scales=scales,
                        **{f"q/{k}": v for k, v in q.items()})
    return {"path": str(out), "bytes": out.stat().st_size, "params": net.n_params()}


def load(path: str | Path) -> tuple[MicroNet, list[str], dict]:
    p = Path(path)
    if not p.exists():
        cand = p.with_suffix(".npz")
        if cand.exists():
            p = cand
    with np.load(p, allow_pickle=False) as z:
        blob = z["header"].tobytes()
        scales = z["scales"]
        q = {k[2:]: z[k] for k in z.files if k.startswith("q/")}
    if not blob.startswith(_MAGIC):
        raise ValueError("内蔵ニューラルコアの形式が違います（tools/distill_neural.py で再生成してください）")
    header = json.loads(zlib.decompress(blob[len(_MAGIC):]).decode("utf-8"))
    meta: dict[str, dict] = {}
    for name, m in header["meta"].items():
        o, n = m["scale_slice"]
        meta[name] = {"shape": m["shape"], "axis": m["axis"],
                      "scale": scales[o:o + n] if n > 1 else float(scales[o])}
    params = dequantize(q, meta)
    net = MicroNet(cfg=NNConfig.from_dict(header["config"]), params=params)
    return net, header["vocab"], header.get("extra", {})
