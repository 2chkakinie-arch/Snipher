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
    version="0.3.0",
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
# ハイブリッド補正: Snipher-mini の下書き → 不安な部分だけ LFM2.5 が書き直し
# ----------------------------------------------------------------------
from snipher.lfm.assist import AssistConfig, HybridAssist

_assist = HybridAssist()
_ASSIST_CFG = _assist.cfg


def _chunk_for_stream(text: str, pieces: int = 3) -> list[str]:
    """軽量経路のテキストを擬似ストリーミング用に文単位で分割する。"""
    import re as _re

    parts = [p for p in _re.split(r"(?<=。)|(?<=？)|(?<=！)", text) if p]
    if len(parts) <= pieces:
        return parts or [text]
    merged: list[str] = []
    per = max(1, -(-len(parts) // pieces))
    for i in range(0, len(parts), per):
        merged.append("".join(parts[i : i + per]))
    return merged


def _mini_reply(messages: list[dict], safe_only: bool | None = None) -> tuple[str, dict]:
    """超小型エンジンで会話応答を作る(対話テーブル + 助動詞の補い)。

    ニューラルエンジンが使えない環境でも、意図に沿った日本語の返答を
    数ミリ秒で組み立てる。confidence が閾値未満の確率的生成文は、
    LFM が無い限り安全な骨子(base_text)に退避させて出力する。
    LFM が使えるときは HybridAssist がこの下書きを LFM に書き直させる。
    """
    last_user = next(
        (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), ""
    )
    draft = _assist.draft(last_user)
    uncertain = _assist.needs_lfm(draft)
    if uncertain and (safe_only or safe_only is None):
        text = draft.get("base_text") or draft["text"]
    else:
        text = draft["text"]
    stats = {
        "engine": "Snipher-mini+",
        "template_mode": "rule-based",
        "new_tokens": None,
        "tokens_per_second": None,
        "assist": "rule",
        "draft": draft["text"],
        "draft_confidence": draft.get("confidence"),
        "intent": draft.get("intent"),
        "fixes": draft.get("fixes", []),
        "draft_seconds": draft.get("draft_seconds"),
    }
    if uncertain:
        stats["degraded_to_base"] = text != draft["text"]
    return text, stats


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
    hybrid: bool | None = Field(
        None,
        description="True/False でハイブリッド補正を強制。未指定時は LFM が使えるなら有効",
    )


def _sse(obj: dict) -> str:
    import json

    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


@app.get("/api/status")
def api_status():
    """ニューラルエンジンと学習済み語彙の状態。"""
    eng = lfm_engine()
    if eng is None:
        return {
            "deps": False,
            "lfm": None,
            "fallback": "Snipher-mini+",
            "hybrid": {"available": False, "assist": "rule", "reason": "torch/transformers 未インストール"},
        }
    if eng.cfg.autostart:
        eng.ensure_started()
    st = eng.status()
    st["engine_label"] = eng.engine_name() if eng.is_ready else None
    return {
        "deps": True,
        "lfm": st,
        "fallback": "Snipher-mini+",
        "hybrid": {
            "available": eng.is_ready,
            "assist": "lfm" if eng.is_ready else "rule",
            "threshold": _ASSIST_CFG.threshold,
            "enabled": _ASSIST_CFG.enabled,
        },
    }


@app.post("/api/chat")
def api_chat(req: ChatRequest):
    """SSE ストリーミングで応答するチャット。

    ハイブリッド経路（既定）:
        Snipher-mini が一瞬で下書きを作り、確率的に不安な返答
        （confidence が閾値未満）のときだけ LFM2.5 が書き直す。
        助動詞・文体の欠落は常にルールで補う（polisher）。
    """
    eng = lfm_engine()
    if eng is not None and eng.cfg.autostart:
        eng.ensure_started()

    msgs = [m.model_dump() for m in req.messages]
    last_user = next((m["content"] for m in reversed(msgs) if m["role"] == "user"), "")
    use_hybrid = req.hybrid if req.hybrid is not None else True

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
                if "huggingface" in (reason or "").lower() or "ConnectionError" in (reason or ""):
                    reason = "HuggingFace に未接続（モデルをローカルから取り込んでください）"

            # ---- 軽量経路: 対話テーブル + 助動詞の補い（数ミリ秒） ----
            text, stats = _mini_reply(msgs)
            stats["fallback_reason"] = reason
            yield _sse({"type": "start", "engine": "Snipher-mini+", "template_mode": "rule-based"})
            for piece in _chunk_for_stream(text):
                yield _sse({"type": "delta", "text": piece})
            yield _sse({"type": "done", "text": text, "stats": stats})
            return

        # ---- LFM 利用可能 ----
        if use_hybrid:
            try:
                draft = _assist.draft(last_user)
                if _assist.needs_lfm(draft):
                    # 確率的に不安な下書き → LFM2.5 が書き直す（賢い経路）
                    yield _sse({
                        "type": "assist",
                        "mode": "lfm",
                        "draft": draft["text"],
                        "confidence": draft.get("confidence"),
                        "reason": "low_confidence",
                    })
                    holder: dict = {}

                    def _polish():
                        text, stats = yield from _assist.polish_with_lfm(
                            eng, draft, last_user,
                            temperature=req.temperature if req.temperature is not None else 0.3,
                            max_new_tokens=max(
                                24, min(req.max_new_tokens or 64, _ASSIST_CFG.max_new_tokens)
                            ),
                        )
                        holder["text"] = text
                        holder["stats"] = stats

                    try:
                        for ev in _polish():
                            yield _sse(ev)
                    except Exception as exc:  # noqa: BLE001 — LFM 補正が失敗しても下書きで応答
                        holder["error"] = str(exc)

                    text = holder.get("text") or draft["text"]
                    stats = holder.get("stats") or {
                        "engine": "Snipher-mini+", "assist": "rule",
                        "template_mode": "rule-based",
                    }
                    if holder.get("error"):
                        stats["assist_fallback"] = holder["error"]
                    yield _sse({"type": "done", "text": text, "stats": stats})
                    return
                # 確信を持てる下書き → そのまま高速返答（LFM は未使用・速度維持）
                stats = {
                    "engine": "Snipher-mini+ (LFM 未使用)",
                    "template_mode": "rule-based",
                    "new_tokens": None,
                    "tokens_per_second": None,
                    "assist": "rule",
                    "draft": draft["text"],
                    "draft_confidence": draft.get("confidence"),
                    "draft_seconds": draft.get("draft_seconds"),
                    "intent": draft.get("intent"),
                    "fixes": draft.get("fixes", []),
                }
                yield _sse({
                    "type": "assist", "mode": "rule", "confidence": draft.get("confidence"),
                })
                yield _sse({"type": "start", "engine": stats["engine"], "template_mode": "rule-based"})
                for piece in _chunk_for_stream(draft["text"]):
                    yield _sse({"type": "delta", "text": piece})
                yield _sse({"type": "done", "text": draft["text"], "stats": stats})
                return
            except Exception:  # noqa: BLE001 — 補正経路が壊れても LFM 直接応答へ
                pass

        # ---- LFM 直接応答（hybrid=False または補正経路の失敗時） ----
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
_ALLOWED_EXTS = {".json", ".safetensors", ".jinja", ".txt", ".model", ".bin"}
_REQUIRED = ["config.json", "*token*", "*.safetensors"]


def _model_dir_status(d: Path) -> dict:
    files = []
    if d.exists():
        for p in sorted(d.iterdir()):
            if p.is_file():
                files.append({"name": p.name, "size": p.stat().st_size})
    has_config = any(f["name"] == "config.json" for f in files)
    has_tokenizer = any(("token" in f["name"].lower()) for f in files)
    has_weights = any(f["name"].endswith(".safetensors") for f in files)
    complete = has_config and has_tokenizer and has_weights
    missing = [r for r, ok in (
        ("config.json", has_config),
        ("tokenizer (tokenizer.json / tokenizer_config.json)", has_tokenizer),
        ("*.safetensors", has_weights),
    ) if not ok]
    return {"dir": str(d), "files": files, "complete": complete, "missing": missing}


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
            "HuggingFace に直接接続できない環境では、ローカル PC で "
            "LiquidAI/LFM2.5-1.2B-JP-202606 をダウンロードし、"
            "config.json / tokenizer.json / tokenizer_config.json / model.safetensors "
            "をここにアップロードしてください。"
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
    """アップロード済み（または指定のローカル）モデルでエンジンを再ロードする。"""
    eng = lfm_engine()
    if eng is None:
        return {"ok": False, "error": "torch/transformers 未インストール"}
    source = req.source
    if not source:
        st = _model_dir_status(UPLOAD_DIR)
        if not st["complete"]:
            return {"ok": False, "error": f"アップロードが不完全です。不足: {', '.join(st['missing'])}"}
        source = str(UPLOAD_DIR)
    p = Path(source)
    if not p.is_dir():
        return {"ok": False, "error": f"ローカルディレクトリが見つかりません: {source}"}
    eng.reload(str(p))
    return {"ok": True, "source": str(p), "state": "loading"}
