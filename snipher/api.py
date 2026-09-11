"""Snipher の FastAPI サーバ。

Vercel では WSGI/ASGI として、Render では `uvicorn snipher.api:app` で起動する。
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field

from snipher.engine import SnipherEngine

engine = SnipherEngine()

app = FastAPI(
    title="Snipher API",
    description="超小型・確率的日本語AI。構文解析 + 独自確率式による高速な日本語生成。",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000, description="解析する日本語文")


class GenerateRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    prompt: str | None = Field(None, description="話題のヒントとなる日本語(任意)")
    register_style: str | None = Field(None, alias="register", pattern="^(polite|casual)$", description="文体: polite(です/ます) / casual(だ/た)")
    tense: str = Field("nonpast", pattern="^(nonpast|past)$", description="時制")
    n: int = Field(1, ge=1, le=20, description="生成する文の数")
    seed: int | None = Field(None, description="乱数シード(再現性のため)")


@app.get("/", response_class=HTMLResponse)
def root():
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
  out.innerHTML = list.map(s =>
    `<div class="sent"><strong>${s.text}</strong><br>
     <span class="tag">${s.pattern_name} / 話題: ${s.topic || "-"}</span></div>`).join("");
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
    return engine.generate(
        prompt=req.prompt,
        register=req.register_style,
        tense=req.tense,
        n=req.n,
        seed=req.seed,
    )


@app.get("/health")
def health():
    return {"status": "ok"}
