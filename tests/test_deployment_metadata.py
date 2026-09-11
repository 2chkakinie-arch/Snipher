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
