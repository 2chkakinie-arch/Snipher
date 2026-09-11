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
