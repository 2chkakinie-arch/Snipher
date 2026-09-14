"""LFM テスト共通フィクスチャ（torch/transformers が無い場合はスキップ）。"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _load_make_test_model():
    spec = importlib.util.spec_from_file_location(
        "make_test_model", REPO_ROOT / "tools" / "make_test_model.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def tiny_model(tmp_path_factory):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    out = tmp_path_factory.mktemp("lfm") / "tiny-lfm2"
    info = _load_make_test_model().build_model(out)
    info["store_dir"] = str(tmp_path_factory.mktemp("lfm_store"))
    return info


@pytest.fixture()
def lfm_env(tiny_model, monkeypatch):
    """セッション共有の小型モデルを指す環境変数を設定。"""
    os.environ["SNIPHER_LFM_MODEL"] = tiny_model["model_dir"]
    os.environ["SNIPHER_LFM_STORE_DIR"] = tiny_model["store_dir"]
    os.environ["SNIPHER_LFM_AUTOSTART"] = "1"
    os.environ["SNIPHER_LFM_PROMPT_BUDGET"] = "384"
    os.environ["SNIPHER_LFM_LEARN_STEPS"] = "2"
    os.environ["SNIPHER_LFM_QUANTIZE"] = "1"
    os.environ["SNIPHER_LFM_RESERVED"] = "64"
    yield tiny_model


def fresh_engine():
    """環境変数設定後の新しい LfmEngine を作る（シングルトンを差し替え）。"""
    from snipher.lfm import config as lfm_config
    from snipher.lfm import engine as lfm_engine_mod

    lfm_engine_mod._ENGINE = None
    cfg = lfm_config.LfmConfig()
    eng = lfm_engine_mod.LfmEngine(cfg)
    lfm_engine_mod._ENGINE = eng
    return eng


# --------------------------------------------------------------------------- #
# v8（MoE + 再帰的思考）テスト用の小型学習済みコア
# --------------------------------------------------------------------------- #
_V8_CORPUS = [
    "<user>赤信号ではどうする？\n<asst><think>信号の意味を思い出す。赤は停止。だから止まる。</think>\n止まります。",
    "<user>3×7は？\n<asst><think>3を7回足すと21。</think>\n21",
    "<user>私は猫が\n<asst>好きです。毎日なでています。",
    "<user>雨の日は何をする？\n<asst><think>濡れない工夫を考える。傘を持つ。部屋で本を読む。</think>\n家で本を読みます。",
    "<user>好きな食べ物は？\n<asst><think>好きな物を思い浮かべる。果物とごはん。</think>\n果物が好きです。",
]


@pytest.fixture(scope="session")
def tiny_moe(tmp_path_factory):
    """v8 テスト用の小型学習済み MoE コア（MoECore）。制限アルファベットで
    学習するので UNK がほぼ出ず、決定的で速い。"""
    pytest.importorskip("numpy")
    import numpy as np

    from snipher.neural.moe import MoEConfig, MoENet
    from snipher.neural.moe_core import MoECore
    from snipher.neural.moe_train import MoETrainer
    from snipher.neural.tokenizer import BOS, CharTokenizer, EOS
    from snipher.neural.train import TextDataset, TrainConfig

    corpus = _V8_CORPUS * 60
    tok = CharTokenizer.from_text("\n".join(corpus), max_vocab=160)
    ids: list[int] = []
    for t in corpus:
        ids += [BOS] + tok.encode(t) + [EOS]
    cfg = MoEConfig(n_vocab=tok.size(), d_model=32, n_layers=2, n_heads=4, n_kv_heads=2,
                    n_experts=4, top_k=2, expert_dim=40, max_pos=128)
    net = MoENet.random(cfg, seed=7)
    ds = TextDataset(np.array(ids, dtype=np.int64), 48, np.random.default_rng(0))
    tc = TrainConfig(seq_len=48, batch=32, epochs=3, lr=8e-3, min_lr=1e-4, warmup_ratio=0.0)
    MoETrainer(net, ds, ds, tc).train()
    return {"core": MoECore(net=net, tok=tok), "tok": tok, "net": net,
            "path": str(tmp_path_factory.mktemp("v8"))}
