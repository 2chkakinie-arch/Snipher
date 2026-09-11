"""Snipher の FastAPI サーバ。

Vercel では WSGI/ASGI として、Render では `uvicorn snipher.api:app` で起動する。
"""

from __future__ import annotations

import contextlib
import random
import threading

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from snipher.engine import SnipherEngine

engine = SnipherEngine()

# ----------------------------------------------------------------------
# LFM2.5-1.2B-JP ニューラルエンジン（オプション。torch が無ければ素通り）
# ----------------------------------------------------------------------
from snipher.lfm import LLM_DEPS_AVAILABLE

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
    eng = lfm_engine()
    if eng is not None and eng.cfg.autostart:
        eng.ensure_started()


@contextlib.asynccontextmanager
async def _lifespan(app: FastAPI):
    _maybe_start_lfm()
    yield


app = FastAPI(
    title="Snipher API",
    description=(
        "超小型・確率的日本語AI + LFM2.5-1.2B-JP チャット。"
        "未知文字の学習とテンプレートフォールバック付きの高速な日常会話。"
    ),
    version="0.2.0",
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
# Snipher-mini フォールバック（ニューラルエンジンが使えないときの応答）
# ----------------------------------------------------------------------
_MINI_LEADS = ["なるほど。", "うんうん。", "そうなんですね。", "へえ、面白いですね。", ""]


def _mini_reply(messages: list[dict]) -> tuple[str, dict]:
    """超小型エンジン(476パラメータ)で会話っぽい一文を作る。"""
    last_user = next(
        (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), ""
    )
    topic = None
    try:
        if last_user:
            topic = engine.analyze(last_user).get("topic")
    except Exception:
        topic = None
    try:
        gen = engine.generate(prompt=topic if topic else None, n=1)
        body = gen["text"]
    except Exception:
        body = "今日もいい一日になりますように。"
    lead = random.choice(_MINI_LEADS)
    text = f"{lead}{body}" if lead else body
    note = "（モデル未ロードのため Snipher-mini で応答中）"
    return f"{text}{note}", {
        "engine": "Snipher-mini",
        "template_mode": "rule-based",
        "new_tokens": None,
        "tokens_per_second": None,
    }


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


def _sse(obj: dict) -> str:
    import json

    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


@app.get("/api/status")
def api_status():
    """ニューラルエンジンと学習済み語彙の状態。"""
    eng = lfm_engine()
    if eng is None:
        return {"deps": False, "lfm": None, "fallback": "Snipher-mini"}
    if eng.cfg.autostart:
        eng.ensure_started()
    st = eng.status()
    st["engine_label"] = eng.engine_name() if eng.is_ready else None
    return {"deps": True, "lfm": st}


@app.post("/api/chat")
def api_chat(req: ChatRequest):
    """SSE ストリーミングで応答するチャット。"""
    eng = lfm_engine()
    if eng is not None and eng.cfg.autostart:
        eng.ensure_started()

    msgs = [m.model_dump() for m in req.messages]
    last_user = next((m["content"] for m in reversed(msgs) if m["role"] == "user"), "")

    def gen():
        ready = eng is not None and eng.is_ready
        # 未知文字の検出（Ready のときだけ）
        if ready:
            try:
                scan = eng.scan_unknown(last_user)
                if scan and scan["unknown"]:
                    learned_map = {c["char"] for c in eng.status()["learned_chars"]}
                    for c in scan["unknown"]:
                        if c["char"] in learned_map:
                            c["learned"] = True  # 学習済みマーク（UI 用）
                    yield _sse({"type": "meta", "unknown_chars": scan["unknown"]})
            except Exception:
                pass

        if not ready:
            reason = None
            if eng is None:
                reason = "torch/transformers 未インストール"
            elif eng.state == "loading":
                reason = "モデル読込中"
            elif eng.state == "learning":
                reason = "学習処理中"
            else:
                reason = eng.error or "モデル未ロード"
            text, stats = _mini_reply(msgs)
            stats["fallback_reason"] = reason
            yield _sse({"type": "start", "engine": "Snipher-mini", "template_mode": "rule-based"})
            yield _sse({"type": "delta", "text": text})
            yield _sse({"type": "done", "text": text, "stats": stats})
            return

        try:
            for ev in eng.stream_chat(
                msgs,
                max_new_tokens=req.max_new_tokens,
                temperature=req.temperature,
                top_k=req.top_k,
                repetition_penalty=req.repetition_penalty,
                use_template=req.use_template,
                system_prompt=req.system_prompt,
            ):
                yield _sse(ev)
        except Exception as exc:  # noqa: BLE001
            yield _sse({"type": "error", "message": str(exc)})

    return StreamingResponse(
        gen(),
        media_type="text/event-stream; charset=utf-8",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


class VocabCheckRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000)


@app.post("/api/vocab/check")
def api_vocab_check(req: VocabCheckRequest):
    """テキスト中の未知文字をスキャンする。"""
    eng = lfm_engine()
    if eng is None or not eng.is_ready:
        return {"ok": False, "error": "ニューラルエンジンが利用できません"}
    scan = eng.scan_unknown(req.text)
    learned = {c["char"] for c in eng.status()["learned_chars"]}
    for c in scan.get("unknown", []):
        if c["char"] in learned:
            c["token_id"] = 1
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
    """未知文字を学習する。mode=deep は非同期ジョブ。"""
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
