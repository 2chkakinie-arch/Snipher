"""デプロイ設定の回帰テスト。"""

from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def test_pyproject_declares_vercel_runtime_dependencies():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject["project"]
    deps = set(project.get("dependencies", []))

    assert any(dep.startswith("fastapi") for dep in deps)
    assert any(dep.startswith("uvicorn") for dep in deps)
    assert any(dep.startswith("pydantic") for dep in deps)


def test_pyproject_declares_vercel_entrypoint():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["tool"]["vercel"]["entrypoint"] == "snipher.api:app"


def test_requirements_include_numpy_for_distilled_core():
    """内蔵ニューラルコアは numpy 依存。Vercel のバンドルに入っていないと動かない。"""
    req = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert any(line.strip().startswith("numpy") for line in req.splitlines() if not line.startswith("#"))
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert any(d.startswith("numpy") for d in pyproject["project"]["dependencies"])


def test_neural_package_is_packaged_with_weights():
    """`snipher.neural` と 同梱重みがパッケージに含むことを確認（サーバーレスの生命線）。"""
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    pkgs = pyproject["tool"]["setuptools"]["packages"]
    assert "snipher.neural" in pkgs
    data = pyproject["tool"]["setuptools"]["package-data"]["snipher"]
    assert any("neural/*.npz" in pat for pat in data)
    assert (ROOT / "snipher" / "neural" / "core.py").exists()
    assert (ROOT / "snipher" / "data" / "neural" / "core.npz").exists()


def test_vercel_json_bounds_function():
    """Vercel 関数の上限設定（731MB は読まない設計なので小さく速く）。"""
    import json

    cfg = json.loads((ROOT / "vercel.json").read_text(encoding="utf-8"))
    fn = cfg["functions"]["snipher/api.py"]
    assert fn["maxDuration"] <= 300
    assert fn["memory"] >= 512
