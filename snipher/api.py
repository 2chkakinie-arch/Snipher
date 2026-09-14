"""Snipher の FastAPI サーバ。

Vercel では WSGI/ASGI として、Render では `uvicorn snipher.api:app` で起動する。
"""

from __future__ import annotations

import contextlib
import os
import threading

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from snipher.engine import SnipherEngine

engine = SnipherEngine()

# ----------------------------------------------------------------------
# Snipher Core — LFM2.5-1.2B-JP を内部ニューラルコアとして動かす
# （モデルは起動時に全自動取得・バックエンド自動選択・アップロード不要）
# ----------------------------------------------------------------------
from snipher.lfm import LLM_DEPS_AVAILABLE
from snipher.lfm.config import is_serverless

_lfm_engine = None
_lfm_lock = threading.Lock()


def lfm_engine():
    """LFM エンジンのシングルトン（依存が無い場合は None）。"""
    global _lfm_engine
    if not LLM_DEPS_AVAILABLE:
        return None
    with _lfm_lock:
        if _lfm_engine is None:
            from snipher.lfm.engine import get_engine

            _lfm_engine = get_engine()
        return _lfm_engine


def _maybe_start_lfm() -> None:
    c = core()
    if c.cfg.autostart:
        c.ensure_started()


_core = None
_core_lock = threading.Lock()


def core():
    """SnipherCore のシングルトン（LFM2.5-1.2B-JP = 内部ニューラルコア）。"""
    global _core
    with _core_lock:
        if _core is None:
            from snipher.core import SnipherCore

            _core = SnipherCore(torch_provider=lfm_engine)
        return _core


@contextlib.asynccontextmanager
async def _lifespan(app: FastAPI):
    _maybe_start_lfm()
    yield
    try:
        core().shutdown()
    except Exception:  # noqa: BLE001
        pass


app = FastAPI(
    title="Snipher API",
    description=(
        "LFM2.5-1.2B-JP を内部構造として動かす高速日本語チャット。"
        "モデルは起動時に全自動取得（llama.cpp GGUF / torch INT8 を自動選択）。"
        "確率的に不安な応答だけニューラルコアが生成し、確実な定形は即答。"
        "未知文字の自動学習とテンプレートフォールバック付き。"
    ),
    version="6.0.0",
    lifespan=_lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------------------------------------------------
# 内部パイプライン: Snipher Core（高速コア + LFM2.5-1.2B-JP ニューラルコア）
# ----------------------------------------------------------------------
from snipher.lfm.assist import AssistConfig, HybridAssist  # noqa: E402,F401  (互換 export)


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000, description="解析する日本語文")


class GenerateRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    prompt: str | None = Field(None, description="話題のヒントとなる日本語(任意)")
    register_style: str | None = Field(None, alias="register", pattern="^(polite|casual)$", description="文体: polite(です/ます) / casual(だ/た)")
    tense: str = Field("nonpast", pattern="^(nonpast|past)$", description="時制")
    n: int = Field(1, ge=1, le=20, description="生成する文の数")
    seed: int | None = Field(None, description="乱数シード(再現性のため)")


def _chat_html() -> str:
    try:
        from importlib import resources

        return resources.files("snipher").joinpath("web/chat.html").read_text(encoding="utf-8")
    except Exception:
        return _INDEX_HTML  # フォールバック: 旧 UI


@app.get("/", response_class=HTMLResponse)
def root():
    return _chat_html()


@app.get("/classic", response_class=HTMLResponse)
def classic():
    return _INDEX_HTML


_INDEX_HTML = """<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Snipher — 超小型・確率的日本語AI</title>
<style>
  :root { --bg:#0f1220; --panel:#1a1e33; --accent:#6c8cff; --text:#e8eaf6; --muted:#9aa0b5; }
  * { box-sizing: border-box; }
  body { margin:0; font-family:"Hiragino Kaku Gothic ProN","Noto Sans JP",system-ui,sans-serif;
         background:var(--bg); color:var(--text); line-height:1.7; }
  .wrap { max-width:860px; margin:0 auto; padding:40px 20px 80px; }
  h1 { font-size:2rem; margin:0 0 4px; letter-spacing:.5px; }
  .sub { color:var(--muted); margin:0 0 28px; font-size:.95rem; }
  .card { background:var(--panel); border:1px solid #2a2f4a; border-radius:14px;
          padding:20px 22px; margin-bottom:20px; }
  .card h2 { margin:0 0 12px; font-size:1.05rem; color:var(--accent); }
  textarea, input, select { width:100%; background:#0c0e1a; color:var(--text);
          border:1px solid #2a2f4a; border-radius:8px; padding:10px 12px;
          font-size:.95rem; font-family:inherit; resize:vertical; }
  .row { display:flex; gap:10px; flex-wrap:wrap; margin-top:10px; }
  .row > * { flex:1 1 140px; }
  button { background:var(--accent); color:#0c0e1a; border:none; border-radius:8px;
          padding:10px 18px; font-size:.95rem; font-weight:700; cursor:pointer; }
  button:hover { filter:brightness(1.1); }
  pre { background:#0c0e1a; border:1px solid #2a2f4a; border-radius:8px;
        padding:14px; overflow-x:auto; font-size:.85rem; white-space:pre-wrap; min-height:24px; }
  .tag { color:var(--muted); font-size:.8rem; }
  .sent { padding:8px 0; border-bottom:1px dashed #2a2f4a; }
  .sent:last-child { border-bottom:none; }
</style>
</head>
<body>
<div class="wrap">
  <h1>Snipher</h1>
  <p class="sub">超小型・確率的日本語AI — 動詞・助動詞・名詞・文構造をあらかじめ決め、
  独自確率式で日本語を高速に生成する試み（日本語のみ対応）</p>

  <div class="card">
    <h2>解析</h2>
    <p class="tag">文構造・助動詞・要点を解析します</p>
    <textarea id="analyzeText" rows="2">私は猫が好きです。</textarea>
    <div class="row"><button onclick="analyze()">解析する</button></div>
    <pre id="analyzeOut"></pre>
  </div>

  <div class="card">
    <h2>生成</h2>
    <p class="tag">話題のヒントを入力すると確率的に日本語文を出力します</p>
    <input id="prompt" placeholder="話題のヒント（例: 猫 / 旅行 / 朝の習慣）">
    <div class="row">
      <select id="register">
        <option value="polite">丁寧体（です・ます）</option>
        <option value="casual">普通体（だ・た）</option>
      </select>
      <select id="n"><option>1</option><option>2</option><option>3</option><option selected>5</option><option>8</option></select>
      <button onclick="generate()">生成する</button>
    </div>
    <div id="genOut"></div>
  </div>

  <div class="card">
    <h2>モデル情報</h2>
    <pre id="infoOut"></pre>
  </div>
</div>

<script>
async function analyze() {
  const out = document.getElementById("analyzeOut");
  out.textContent = "解析中…";
  const r = await fetch("/analyze", {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({text: document.getElementById("analyzeText").value})});
  const d = await r.json();
  out.textContent = "文型: " + (d.structure?.name || "-") + "\\n話題: " + (d.topic || "-")
    + "\\n助動詞: " + (d.auxiliaries?.map(t=>t.surface).join(", ") || "-")
    + "\\n要点: " + (d.key_points?.map(p => p.value || p.role).join(" / ") || "-")
    + "\\n\\n" + JSON.stringify(d, null, 2);
}
async function generate() {
  const out = document.getElementById("genOut");
  out.innerHTML = "生成中…";
  const body = {
    prompt: document.getElementById("prompt").value || null,
    register: document.getElementById("register").value,
    n: Number(document.getElementById("n").value)};
  const r = await fetch("/generate", {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify(body)});
  const d = await r.json();
  const list = d.sentences || [d];
  out.innerHTML = list.map(s => {
    const shown = (!s.valid && s.repaired) ? s.repaired : s.text;
    const flag = s.valid === false
      ? ` <span class="tag" style="color:#b45309">要校正: ${s.invalid_reason || "文法検査"}</span>` : "";
    return `<div class="sent"><strong>${shown}</strong><br>
     <span class="tag">${s.pattern_name} / 話題: ${s.topic || "-"}</span>${flag}</div>`;
  }).join("");
}
(async () => {
  const r = await fetch("/info"); const d = await r.json();
  document.getElementById("infoOut").textContent =
    "パラメータ数: " + d.total_parameters + "（テーブル " + d.table_entries + " + 確率式の重み " + d.probability_weights + "）\\n"
    + "確率式: " + d.formula + "\\n重み: " + JSON.stringify(d.weights) + "\\n辞書: " + JSON.stringify(d.lexicon_stats);
})();
</script>
</body>
</html>
"""


@app.get("/info")
def info():
    return engine.info()


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    return engine.analyze(req.text)


@app.post("/generate")
def generate(req: GenerateRequest):
    out = engine.generate(
        prompt=req.prompt,
        register=req.register_style,
        tense=req.tense,
        n=req.n,
        seed=req.seed,
    )
    # 確率生成のデモでも、文章の検査は本体と同じものをとおします。
    from .composer import validate as _validate

    def mark(item: dict) -> dict:
        ok, why = _validate(str(item.get("text") or ""), max_len=200)
        item["valid"] = ok
        if not ok:
            item["invalid_reason"] = why
            # 壊れた文は *壊れている* と分かる形で添えるだけ。整った言い方を composer が作り直す。
            fixed = _tidy_demo_text(str(item.get("text") or ""))
            if fixed and fixed != item.get("text"):
                item["repaired"] = fixed
        return item

    if "sentences" in out:
        out["sentences"] = [mark(x) for x in out["sentences"]]
    else:
        mark(out)
    return out


def _tidy_demo_text(text: str) -> str:
    """生成デモの文から、明らかな重複語尾・二重敬体を寄せて返す（通らなければ空）。"""
    import re as _re

    from .composer import balance_quotes

    t = balance_quotes(text)
    t = _re.sub(r"(ます|です)\s*(ます|です)", r"\1", t)
    t = _re.sub(r"(よ|ね|わ)(よ|ね|わ)", r"\1", t)
    t = _re.sub(r"[。、]{2,}", "。", t).strip()
    return t


@app.get("/health")
def health():
    return {"status": "ok"}


# ====================================================================== #
# LFM2.5-1.2B-JP チャット API
# ====================================================================== #
class ChatMessage(BaseModel):
    role: str = Field("user", pattern="^(system|user|assistant)$")
    content: str = Field(..., max_length=4000)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(..., min_length=1, max_length=64)
    max_new_tokens: int | None = Field(None, ge=8, le=512)
    temperature: float | None = Field(None, ge=0.0, le=2.0)
    top_k: int | None = Field(None, ge=1, le=200)
    repetition_penalty: float | None = Field(None, ge=1.0, le=2.0)
    use_template: bool = Field(True, description="False でテンプレートなし生成")
    system_prompt: str | None = Field(None, max_length=2000)
    # auto: 現在情報/出典が必要なときだけ Edge HTML 検索。on/off で明示指定。
    web: str = Field("auto", pattern="^(auto|on|off)$",
                     description="ウェブ調査: auto=必要時のみ / on=明示的に検索 / off=無効")
    # JS クライアントや旧 API の bool 指定も受け付ける。
    web_search: bool | None = Field(None, description="web の旧互換 bool。指定時は web より優先")
    mode: str = Field(
        "auto",
        pattern="^(auto|fast|lfm|neural|light)$",
        description=(
            "auto=内部パイプライン(確実な定形は即答/確率的に不安な応答は LFM2.5 が生成), "
            "fast=高速コアのみ, lfm=常に LFM2.5 が生成, light=内蔵蒸留コア＋知識ベース固定"
        ),
    )
    hybrid: bool | None = Field(
        None,
        description="旧互換フラグ。False は mode=lfm、True は mode=auto と同じ",
    )


def _sse(obj: dict) -> str:
    import json

    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


def _neural_deps() -> bool:
    """どちらかのニューラルランタイム(torch / llama.cpp)が入っているか。"""
    if LLM_DEPS_AVAILABLE:
        return True
    try:
        from snipher.lfm.gguf_backend import runtime_kind

        return runtime_kind(core().cfg) is not None
    except Exception:  # noqa: BLE001
        return False


@app.get("/api/status")
def api_status():
    """Snipher Core の状態（知能階層・自動取得の進捗・学習済み語彙・知識ベース）。

    依存ランタイムが無い環境（Vercel 等）でも 200 で、内蔵蒸留コアと
    知識ベースの有無を必ず返す（＝「何も無し」に見えないようにする）。
    """
    c = core()
    deps = _neural_deps()
    if deps and c.cfg.autostart:
        c.ensure_started()
    st = c.status()
    ready = bool(st.get("neural_ready"))
    light = st.get("light") or {"state": c._light_state}
    return {
        "deps": deps,
        "backend": st.get("backend_kind"),
        "lfm": st,
        "acquire": st.get("acquire"),
        "fallback": "Snipher-mini+",
        "hybrid": {
            "available": ready,
            "assist": "lfm" if ready else ("light" if light.get("state") == "ready" else "rule"),
            "threshold": c.assist.cfg.threshold,
            "enabled": c.assist.cfg.enabled,
            "light_available": light.get("state") == "ready",
        },
        "light": light,
        "lm": st.get("lm"),
        "composer": {"available": True, "engine": "Snipher composer (文の設計図)"},
        "knowledge": st.get("knowledge"),
        "research": st.get("research"),
        "tiers": st.get("tiers"),
        "serverless": is_serverless(),
        "note": (None if ready else
                 ("LFM2.5-1.2B-JP のフルウェイトは読み込みません。内蔵ニューラルコア"
                  "（LFM2.5 を蒸留した同梱スナップショット）と知識ベースで応答します。")),
    }


@app.get("/api/neural")
def api_neural():
    """内蔵ニューラルコア（LFM2.5 蒸留スナップショット）の詳細と生成テスト。"""
    from .neural.cache import get_core

    c = get_core()
    if c is None:
        return {"available": False, "reason": "重みが未ビルドです（tools/distill_neural.py）"}
    return {"available": True, **c.status()}


class NeuralProbeRequest(BaseModel):
    text: str = Field(..., max_length=500)
    max_chars: int = Field(48, ge=4, le=200)
    temperature: float = Field(0.7, ge=0.0, le=2.0)


@app.post("/api/neural/probe")
def api_neural_probe(req: NeuralProbeRequest):
    """内蔵コアに生成・補完・採点させてみる（デバッグ/紹介用）。"""
    from .neural.cache import get_core

    c = get_core()
    if c is None:
        return {"ok": False, "reason": "内蔵ニューラルコアが未ビルドです"}
    import time

    t0 = time.time()
    gen = c.generate(req.text, max_chars=req.max_chars, temperature=req.temperature)
    comp = c.complete(req.text)
    sc = c.score(req.text)
    return {"ok": True, "generate": gen, "complete": comp, "score": sc,
            "seconds": round(time.time() - t0, 3)}


# ---------------- 内蔵ニューラルコアの再蒸留（全自動・任意実行） ------------- #
_REBUILD: dict = {"running": False, "started_at": None, "finished_at": None,
                  "log": [], "returncode": None, "profile": None}
_REBUILD_LOCK = threading.Lock()


def _rebuild_worker(profile: str) -> None:
    import subprocess
    import sys
    import time

    cmd = [sys.executable, str(REPO_ROOT / "tools" / "distill_neural.py"), "--profile", profile,
           "--out", str(REPO_ROOT / "snipher" / "data" / "neural" / "core.npz"),
           "--report", str(REPO_ROOT / "snipher" / "data" / "neural" / "report.json")]
    _REBUILD["log"] = ["$ " + " ".join(cmd)]
    try:
        proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            log_buf = _REBUILD["log"]
            log_buf.append(line.rstrip())
            if len(log_buf) > 400:
                del log_buf[:100]
        proc.wait()
        _REBUILD["returncode"] = proc.returncode
        if proc.returncode == 0:
            from .neural.cache import reset as _reset_cache

            _reset_cache()      # 新しい重みを即座に差し替え
            log_buf = _REBUILD["log"]
            log_buf.append("内蔵ニューラルコアを再ビルドしました（自動で差し替え済み）")
            if len(log_buf) > 400:
                del log_buf[:100]
    except Exception as exc:  # noqa: BLE001
        _REBUILD["returncode"] = -1
        _REBUILD["log"].append(f"エラー: {exc}")
    finally:
        import time as _t

        _REBUILD["finished_at"] = _t.time()
        _REBUILD["running"] = False


class RebuildRequest(BaseModel):
    profile: str = Field("base", pattern="^(tiny|base|big)$",
                         description="tiny=数秒（検証用） / base=既定 / big=高品質（遅い）")


@app.post("/api/neural/rebuild")
def api_neural_rebuild(req: RebuildRequest):
    """語彙テーブルや知識ベースを増やしたら、内蔵ニューラルコアを再蒸留できる。

    サーバーが自前のデータから自分で作り直すので、ユーザーがファイルを
    用意したりアップロードしたりする必要は無い（実行はバックグラウンド）。
    """
    import time

    with _REBUILD_LOCK:
        if _REBUILD["running"]:
            return {"ok": False, "reason": "すでに再ビルドが実行中です", **_rebuild_state()}
        if is_serverless():
            return {"ok": False,
                    "reason": "サーバーレス環境では再ビルド（CPU 学習）は行えません。"
                              "同梱済みのスナップショットが使われます。",
                            **_rebuild_state()}
        _REBUILD.update({"running": True, "started_at": time.time(), "finished_at": None,
                         "returncode": None, "profile": req.profile})
        threading.Thread(target=_rebuild_worker, args=(req.profile,),
                         name="snipher-rebuild", daemon=True).start()
    return {"ok": True, "started": True, "profile": req.profile, **_rebuild_state()}


def _rebuild_state() -> dict:
    return {"rebuild": {k: v for k, v in _REBUILD.items() if k != "log"} |
            {"log_tail": _REBUILD["log"][-12:]}}


@app.get("/api/neural/rebuild")
def api_neural_rebuild_status():
    return {"ok": True, **_rebuild_state()}


@app.get("/api/kb")
def api_kb(q: str = "", k: int = 3):
    """知識ベース（BM25）を検索する。"""
    kb = core().kb
    if kb is None:
        return {"ok": False, "reason": "知識ベースが使えません"}
    if not q.strip():
        return {"ok": True, "stats": kb.stats(), "results": []}
    k = max(1, min(10, int(k)))
    return {"ok": True, "stats": kb.stats(), "query": q,
            "answer": kb.answer(q), "results": kb.search(q, k)}


class ResearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    limit: int = Field(5, ge=1, le=10)
    fetch_pages: int = Field(2, ge=0, le=3)
    web: bool | None = Field(True, description="False ならネットワークを使わず判定だけ行う")


@app.post("/api/research")
def api_research(req: ResearchRequest):
    """Edge/Bing HTML 検索と本文取得を行い、出典付き材料を返す。"""
    result = core().research.research(req.query, explicit=req.web,
                                      limit=req.limit, fetch_pages=req.fetch_pages)
    return {"ok": True, **result.as_dict()}


@app.get("/api/search")
def api_search(q: str = "", k: int = 5, web: bool = True):
    """軽量検索 API（/api/research の GET 互換）。"""
    if not q.strip():
        return {"ok": False, "error": "q is required"}
    result = core().research.research(q, explicit=web, limit=max(1, min(10, int(k))), fetch_pages=0)
    return {"ok": True, **result.as_dict()}


class HtmlFetchRequest(BaseModel):
    url: str = Field(..., min_length=8, max_length=2000)


@app.post("/api/fetch")
def api_fetch(req: HtmlFetchRequest):
    """安全な HTML fetcher を直接使うデバッグ/統合 API。"""
    result = core().research.html_fetch(req.url)
    return {"ok": result.error is None, **result.as_dict()}


@app.get("/api/fetch")
def api_fetch_get(url: str = ""):
    if not url.strip():
        return {"ok": False, "error": "url is required"}
    result = core().research.html_fetch(url)
    return {"ok": result.error is None, **result.as_dict()}


@app.get("/api/model/remote")
def api_remote_status():
    """LFM2.5 フルウェイトのリモート委譲（任意設定）の疎通確認。"""
    c = core()
    b = c.remote_backend()
    if b is None:
        return {"configured": False,
                "hint": "SNIPHER_LFM_REMOTE_URL に LFM2.5 を常駐させた Snipher の URL を設定すると、"
                        "サーバーレスでもフルウェイトの生成を委譲できます。"}
    ok = b.probe()
    return {"configured": True, "alive": ok, **b.status()}


@app.get("/api/model/acquire")
def api_model_acquire():
    """自動取得ジョブの進捗（UI がポーリングする）。"""
    c = core()
    return {"boot_state": c.boot_state, "boot_error": c.boot_error,
            "backend": c.backend_kind, **c.acquire_status()}


class FetchRequest(BaseModel):
    backend: str | None = Field(None, pattern="^(gguf|torch)$", description="取得する形式の強制指定")


@app.post("/api/model/fetch")
def api_model_fetch(req: FetchRequest):
    """モデルの自動取得を（再）トリガーする。通常は起動時に全自動で走る。"""
    return core().fetch_now(req.backend)


class CompleteRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    text: str = Field(..., min_length=1, max_length=2000)
    register_style: str = Field("polite", alias="register", pattern="^(polite|casual)$")
    use_neural: bool = True


@app.post("/api/complete")
def api_complete(req: CompleteRequest):
    """助動詞の補い: 文末・助動詞が欠けた断片文を内部パイプラインで補完する。"""
    try:
        return {"ok": True, **core().complete_fragment(
            req.text, register=req.register_style, use_neural=req.use_neural)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


class SteerRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000, description="リアルタイム・ステアリング用の追加入力")
    strength: float = Field(1.0, ge=0.1, le=5.0, description="バイアス強度")


@app.post("/api/steer")
def api_steer(req: SteerRequest):
    """生成中でも受け付けるリアルタイム・ステアリング (確率の波への干渉).

    前端は stream 中に別プロンプトをここへ POST する。サーバは
    ロジット・バイアスとして蓄積し、直後の生成トークンから反映する。
    出力は止まらない — 波は次のサンプリングから滑らかに乗る。
    """
    try:
        c = core()
        bus = c.steering_bus()
        if bus is None:
            return {"ok": False, "error": "ステアリングバスが利用できません"}
        tok = None
        try:
            from snipher.neural.cache import get_core
            _core_neural = get_core()
            if _core_neural is not None:
                tok = getattr(_core_neural, "tok", None)
        except Exception:  # noqa: BLE001
            tok = None
        sid = bus.steer(req.text, strength=req.strength, tokenizer=tok)
        if sid < 0:
            return {"ok": False, "error": "確率波に変換できませんでした（tokenizer 未ロード）"}
        st = bus.status()
        return {"ok": True, "id": sid, "queued": req.text[:80],
                "strength": req.strength, "active_waves": st["pending"]}
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)


@app.post("/api/chat")
def api_chat(req: ChatRequest):
    """SSE ストリーミングで応答するチャット（Snipher Core 内部パイプライン）。

    - 確実な定形応答(挨拶・感謝など)  → 高速コアが数ミリ秒で即答（速度維持）
    - 確率的に不安な応答(質問・雑談)  → LFM2.5-1.2B-JP が内部で本文を生成し、
      助動詞の補い(polisher)を通して返す
    - フルウェイトが無い環境         → 内蔵ニューラルコア（LFM2.5 蒸留）+知識ベースで生成
    - ニューラルコアが一切使えない    → 高速コアが安全な応答を返し、
      自動取得の進捗をイベントに載せる
    """
    c = core()
    if c.cfg.autostart:
        c.ensure_started()

    msgs = [m.model_dump() for m in req.messages]
    mode = req.mode
    if req.hybrid is False:
        mode = "lfm"
    elif req.hybrid is True and mode == "auto":
        mode = "auto"

    def gen():
        try:
            for ev in c.stream_reply(
                msgs,
                mode=mode,
                max_new_tokens=req.max_new_tokens,
                temperature=req.temperature,
                top_k=req.top_k,
                repetition_penalty=req.repetition_penalty,
                use_template=req.use_template,
                system_prompt=req.system_prompt,
                web=(req.web_search if req.web_search is not None else
                     (True if req.web == "on" else False if req.web == "off" else None)),
            ):
                yield _sse(ev)
        except Exception as exc:  # noqa: BLE001
            yield _sse({"type": "error", "message": f"内部エラー: {exc}"})

    return StreamingResponse(
        gen(),
        media_type="text/event-stream; charset=utf-8",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


class VocabCheckRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000)


@app.post("/api/vocab/check")
def api_vocab_check(req: VocabCheckRequest):
    """テキスト中の未知文字をスキャンする（GGUF バックエンドは byte-fallback で常に 0）。"""
    c = core()
    backend = c.active_backend()
    if backend is None:
        return {"ok": False, "error": "ニューラルエンジンが利用できません"}
    scan = backend.scan_unknown(req.text)
    if scan is None:
        return {"ok": False, "error": "ニューラルエンジンが利用できません"}
    if getattr(backend, "kind", "") != "gguf":
        learned = {ch["char"] for ch in backend.status()["learned_chars"]}
        for ch in scan.get("unknown", []):
            if ch["char"] in learned:
                ch["token_id"] = 1
                ch["learned"] = True
    return {"ok": True, **scan}


class LearnRequest(BaseModel):
    chars: list[str] = Field(default_factory=list, max_length=64)
    text: str | None = Field(None, max_length=8000)
    examples: list[str] = Field(default_factory=list, max_length=32)
    mode: str = Field("instant", pattern="^(instant|deep)$")
    steps: int | None = Field(None, ge=2, le=64)


_learn_job: dict = {"phase": "idle", "detail": None, "result": None, "error": None}


@app.post("/api/learn")
def api_learn(req: LearnRequest):
    """未知文字を学習する（LFM2.5 の学習済み埋め込みを適用）。mode=deep は非同期ジョブ。

    予約トークン方式の学習は torch バックエンド専用。GGUF バックエンドは
    byte-fallback トークナイザなので未知文字は発生せず、学習は不要。
    """
    if core().backend_kind == "gguf" and core().neural_available():
        return {"ok": False,
                "error": "GGUF バックエンドでは未知文字は発生しません（byte-fallback のため学習不要）。"
                         "埋め込み学習を使う場合は SNIPHER_LFM_BACKEND=torch で起動してください。"}
    eng = lfm_engine()
    if eng is None or not eng.is_ready:
        return {"ok": False, "error": "ニューラルエンジンが利用できません（モデル未ロード）"}

    chars = [c for c in req.chars if c.strip()]
    if not chars and req.text:
        scan = eng.scan_unknown(req.text)
        chars = [c["char"] for c in scan.get("unknown", [])]
    if not chars:
        return {"ok": False, "error": "学習対象の未知文字が見つかりません（文字を直接 chars で指定することもできます）"}

    if req.mode == "deep":
        if _learn_job["phase"] in ("running",):
            return JSONResponse({"ok": False, "error": "学習ジョブが進行中です"}, status_code=409)
        _learn_job.update(phase="running", detail=None, result=None, error=None)

        def _prog(i, n, loss):
            _learn_job["detail"] = f"step {i}/{n} loss={loss:.3f}"

        def _run():
            try:
                res = eng.learn_deep(chars, req.examples, steps=req.steps, on_progress=_prog)
                _learn_job.update(phase="done", result=res)
            except Exception as exc:  # noqa: BLE001
                _learn_job.update(phase="error", error=str(exc))

        threading.Thread(target=_run, name="lfm-learn", daemon=True).start()
        return {"ok": True, "mode": "deep", "chars": chars, "job": "started"}

    try:
        res = eng.learn_instant(chars, req.examples)
        return {"ok": True, **res}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


@app.get("/api/learn/status")
def api_learn_status():
    eng = lfm_engine()
    return {**_learn_job, "engine_state": eng.state if eng else None}


@app.delete("/api/learn")
def api_learn_reset():
    eng = lfm_engine()
    if eng is None:
        return {"ok": False, "removed": 0}
    try:
        return {"ok": True, **eng.reset_learned()}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


# ====================================================================== #
# モデル取り込み（HuggingFace に接続できない環境向け）
# ====================================================================== #
# このサンドボックスのように huggingface.co への外向き接続が遮断されている
# 環境では、ユーザーの PC でモデルをダウンロードし、ブラウザからこの UI 経由で
# アップロードする（または tools/fetch_model.py のミラー機能を使う）。
from pathlib import Path  # noqa: E402

from fastapi import File, Form, UploadFile  # noqa: E402

from snipher.lfm.config import REPO_ROOT  # noqa: E402

UPLOAD_DIR = Path(os.environ.get("SNIPHER_LFM_UPLOAD_DIR", str(REPO_ROOT / "var" / "models" / "upload")))
_ALLOWED_EXTS = {".json", ".safetensors", ".jinja", ".txt", ".model", ".bin", ".gguf"}
_REQUIRED = ["config.json", "*token*", "*.safetensors"]


def _model_dir_status(d: Path) -> dict:
    files = []
    gguf = None
    if d.exists():
        for p in sorted(d.iterdir()):
            if p.is_file():
                files.append({"name": p.name, "size": p.stat().st_size})
                if p.suffix.lower() == ".gguf" and p.stat().st_size > 10 * 1024 * 1024:
                    gguf = str(p)
    has_config = any(f["name"] == "config.json" for f in files)
    has_tokenizer = any(("token" in f["name"].lower()) for f in files)
    has_weights = any(f["name"].endswith(".safetensors") for f in files)
    complete = bool(gguf) or (has_config and has_tokenizer and has_weights)
    missing = [] if gguf else [r for r, ok in (
        ("config.json", has_config),
        ("tokenizer (tokenizer.json / tokenizer_config.json)", has_tokenizer),
        ("*.safetensors (または *.gguf)", has_weights),
    ) if not ok]
    return {"dir": str(d), "files": files, "complete": complete, "missing": missing, "gguf": gguf}


@app.get("/api/model/import")
def api_model_import():
    """ローカル取り込みディレクトリの状態(不足ファイルの判定つき)。"""
    eng = lfm_engine()
    return {
        "upload": _model_dir_status(UPLOAD_DIR),
        "current_source": eng.cfg.model_source if eng else None,
        "engine_state": eng.state if eng else None,
        "engine_error": eng.error if eng else None,
        "hint": (
            "モデルは起動時に全自動取得されます（操作不要）。この手動取り込みは、"
            "ネットワークが完全に遮断された環境向けの最後の手段です。"
            "LiquidAI/LFM2.5-1.2B-JP-202606 の config.json / tokenizer.json / "
            "tokenizer_config.json / model.safetensors（または GGUF 1 ファイル）を"
            "ドロップしてください。"
        ),
    }


@app.post("/api/model/upload")
async def api_model_upload(files: list[UploadFile] = File(...), activate: str = Form("0")):
    """モデルファイルをブラウザからアップロードする（複数可・一覧は GET /api/model/import）。

    完了後 activate=1 でエンジンをホットリロードする。
    """
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    saved = []
    skipped = []
    for f in files:
        name = Path(f.filename or "").name  # パス区切りを除去
        ext = Path(name).suffix.lower()
        if not name or ext not in _ALLOWED_EXTS:
            skipped.append(name or "(unnamed)")
            continue
        dest = UPLOAD_DIR / name
        size = 0
        with open(dest, "wb") as out:  # 100MB 単位でディスクへ書き流す
            while chunk := await f.read(100 * 1024 * 1024):
                out.write(chunk)
                size += len(chunk)
        saved.append({"name": name, "size": size})
    status = _model_dir_status(UPLOAD_DIR)
    result: dict = {"ok": True, "saved": saved, "skipped": skipped, "status": status}
    if activate == "1" and status["complete"]:
        eng = lfm_engine()
        if eng is not None:
            eng.reload(str(UPLOAD_DIR))
            result["activated"] = True
    return result


class ModelLoadRequest(BaseModel):
    source: str | None = Field(None, max_length=1000, description="ローカルのモデルディレクトリ(既定: アップロード済みディレクトリ)")


@app.post("/api/model/load")
def api_model_load(req: ModelLoadRequest):
    """アップロード済み（または指定のローカル）モデルで Snipher Core を再ロードする。

    通常これを使う必要はない（起動時の自動取得が本体）。これはネットワークが
    完全に遮断された環境向けの最後の手段。
    """
    source = req.source
    if not source:
        st = _model_dir_status(UPLOAD_DIR)
        if not st["complete"]:
            return {"ok": False, "error": f"アップロードが不完全です。不足: {', '.join(st['missing'])}"}
        source = st.get("gguf") or str(UPLOAD_DIR)
    p = Path(source)
    if not (p.is_dir() or (p.is_file() and p.suffix.lower() == ".gguf")):
        return {"ok": False, "error": f"ローカルのモデルが見つかりません: {source}"}
    core().reload(str(p))
    return {"ok": True, "source": str(p), "state": "loading"}
