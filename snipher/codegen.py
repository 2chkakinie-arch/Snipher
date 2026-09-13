"""Snipher 万能コードシンセサイザー。

小さな言語モデルにコードを確率生成させると、文法はそれっぽくても
動かない。そこで「動くこと」を最優先に、要求の言語・課題を判定して
実行可能なコードを 0 から組み立てる。

対応:
  - HTML/CSS/JS 単一ファイル (オセロ/AI対戦・TODO・電卓・汎用ページ)
  - Python / JavaScript / TypeScript / Java / Go / Rust の実用スニペット
  - 未知の依頼でも言語に合った雛形 + 実行手順を返す (「できない」とは言わない)

依存は標準ライブラリのみ。生成は全て決定的テンプレート + 要求語の
スロット展開で、毎回同じコピペにならないよう要求語を埋め込む。
"""

from __future__ import annotations

import re


# ---------------------------------------------------------------------------
# 言語・課題の判定
# ---------------------------------------------------------------------------

_LANG_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("html", ("html", "index.html", "ホームページ", "ホームぺージ", "ウェブページ", "webページ", "サイト", "ページ", "オセロ", "リバーシ")),
    ("javascript", ("javascript", "js", "node", "node.js", "typescript", "ts", "ブラウザ", "フロント")),
    ("python", ("python", "パイソン", "パイソン")),
    ("java", ("java",)),
    ("go", ("golang", " go", "go言語")),
    ("rust", ("rust",)),
    ("css", ("css", "スタイル")),
)

_CODE_VERBS = (
    "コード", "プログラム", "実装", "スクリプト", "書いて", "書き方", "関数", "バグ", "デバッグ",
    "コード例", "作成", "作って", "作る", "開発", "アプリ", "ゲーム", "電卓", "todo", "TODO",
    "オセロ", "リバーシ", "将棋", "テトリス", "ソート", "掲示板", "タイマー", "時計", "カウンター",
)

_GAME_WORDS = ("オセロ", "リバーシ", "othello", "reversi")


def detect_language(text: str) -> str:
    t = str(text or "").lower()
    # 明示指定が最優先
    if "index.html" in t or ".html" in t or re.search(r"html\s*(?:で|に|の|を)", t):
        return "html"
    if "typescript" in t or re.search(r"\bts\b", t):
        return "typescript"
    if "javascript" in t or re.search(r"\bjs\b", t) or "node" in t:
        return "javascript"
    if "python" in t or "パイソン" in t:
        return "python"
    if re.search(r"\bjava\b", t):
        return "java"
    if "rust" in t:
        return "rust"
    if re.search(r"\bgo\b", t) or "golang" in t:
        return "go"
    # ゲーム系はブラウザで動く HTML 単一ファイルが最も喜ばれる
    if any(w in str(text or "") for w in _GAME_WORDS):
        return "html"
    if any(w in t for w in ("ゲーム", "アプリ", "サイト", "ページ", "電卓", "todo")):
        # 言語の指定が無ければブラウザで即動く HTML を選ぶ
        return "html"
    return "python"


_CREATIVE_GUARD = ("小説", "物語", "ストーリー", "エッセイ", "詩", "ポエム", "作文", "随筆", "短編", "長編", "童話")

def is_code_request(text: str) -> bool:
    t = str(text or "")
    tl = t.lower()
    # 創作依頼はコードではない (「小説を書いて」をコードに誤認しない)
    if any(w in t for w in _CREATIVE_GUARD):
        # プログラミング言語の明示があれば code 扱い
        if not re.search(r"python|javascript|typescript|java|\bgo\b|rust|html|css|node\.js|プログラム|コード|関数", tl):
            return False
    if any(v in t for v in _CODE_VERBS):
        return True
    # 言語名 + 動詞の組み合わせ
    if re.search(r"python|javascript|typescript|java|go|rust|html|css|node|next", tl):
        if any(v in tl for v in ("書", "作", "実装", "例", "code", "program", "関数", "function", "class")):
            return True
    # 「○○を作成して」のような直接依頼
    if re.search(r"(を作成|を作って|を開発|を実装|を書いて)", t):
        return True
    return False


def detect_task(text: str) -> str:
    t = str(text or "")
    tl = t.lower()
    if any(w in t for w in _GAME_WORDS):
        return "othello"
    if "テトリス" in t or "tetris" in tl:
        return "tetris_hint"
    if "todo" in tl or "TODO" in t or "やること" in t or "タスク管理" in t:
        return "todo"
    if "電卓" in t or "calculator" in tl or "計算機" in t:
        return "calculator"
    if "タイマー" in t or "timer" in tl or "時計" in t or "clock" in tl:
        return "timer"
    if "fizzbuzz" in tl or "フィズ" in t:
        return "fizzbuzz"
    if "フィボナッチ" in t or "fibonacci" in tl:
        return "fibonacci"
    if "素数" in t or "prime" in tl:
        return "prime"
    if "ソート" in t or "sort" in tl or "並べ替え" in t or "並び替え" in t:
        return "sort"
    if "スクレイピング" in t or "scrap" in tl or "クロール" in t:
        return "scraper"
    if "api" in tl and ("サーバー" in t or "server" in tl or "バックエンド" in t):
        return "api_server"
    return "generic"


# ---------------------------------------------------------------------------
# オセロ (AI対戦) — 単一 index.html としてそのまま動く完全実装
# ---------------------------------------------------------------------------

def othello_html() -> str:
    return r"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AIオセロ — Snipher</title>
<style>
  :root { --bg1:#0f2027; --bg2:#203a43; --bg3:#2c5364; --panel:rgba(255,255,255,.96);
    --ink:#1a2233; --muted:#5b6b82; --green:#0e7a4f; --green-d:#0a5c3c; --accent:#ffb020; }
  * { box-sizing:border-box; }
  body { margin:0; min-height:100vh; font-family:"Hiragino Kaku Gothic ProN","Noto Sans JP",system-ui,sans-serif;
    background:linear-gradient(135deg,var(--bg1),var(--bg2),var(--bg3)); color:#fff;
    display:flex; flex-direction:column; align-items:center; padding:28px 16px 60px; }
  h1 { font-size:1.5rem; margin:0 0 4px; letter-spacing:.5px; }
  .sub { color:#c7d5e3; font-size:.85rem; margin:0 0 18px; }
  .wrap { display:flex; gap:22px; align-items:flex-start; flex-wrap:wrap; justify-content:center; width:100%; max-width:980px; }
  .panel { background:var(--panel); color:var(--ink); border-radius:18px; padding:20px 22px;
    box-shadow:0 24px 60px rgba(0,0,0,.35); }
  .board-panel { display:flex; flex-direction:column; align-items:center; }
  #board { display:grid; grid-template-columns:repeat(8, min(9.5vw,52px)); grid-template-rows:repeat(8, min(9.5vw,52px));
    gap:3px; background:linear-gradient(160deg,var(--green),var(--green-d)); padding:10px; border-radius:14px;
    box-shadow:inset 0 2px 12px rgba(0,0,0,.4); }
  .cell { background:rgba(255,255,255,.08); border-radius:7px; display:flex; align-items:center; justify-content:center;
    cursor:pointer; position:relative; transition:background .12s; }
  .cell:hover { background:rgba(255,255,255,.18); }
  .cell.hint::after { content:""; width:26%; height:26%; border-radius:50%; background:rgba(255,255,255,.45); }
  .disc { width:82%; height:82%; border-radius:50%; box-shadow:0 3px 6px rgba(0,0,0,.45), inset 0 -3px 6px rgba(0,0,0,.25);
    animation:pop .18s ease-out; }
  @keyframes pop { from { transform:scale(.4); } to { transform:scale(1); } }
  .disc.b { background:radial-gradient(circle at 32% 30%, #4a5568, #0b0e14 70%); }
  .disc.w { background:radial-gradient(circle at 32% 30%, #ffffff, #cbd5e0 70%); }
  .side { min-width:min(88vw,300px); max-width:320px; display:flex; flex-direction:column; gap:14px; }
  .score { display:flex; gap:10px; }
  .score .card { flex:1; text-align:center; background:#f1f5f9; border-radius:12px; padding:10px 6px; }
  .score .n { font-size:1.6rem; font-weight:800; }
  .score .l { font-size:.72rem; color:var(--muted); }
  .me { outline:2px solid var(--accent); }
  #status { font-size:.92rem; font-weight:700; min-height:1.6em; }
  .row { display:flex; gap:8px; align-items:center; flex-wrap:wrap; font-size:.85rem; }
  select, button { font:inherit; }
  select { padding:7px 10px; border-radius:9px; border:1px solid #cbd5e0; background:#fff; }
  button { border:none; border-radius:10px; padding:9px 14px; font-weight:700; cursor:pointer; }
  .primary { background:var(--ink); color:#fff; } .primary:hover { opacity:.88; }
  .ghost { background:#e2e8f0; color:var(--ink); } .ghost:hover { background:#cbd5e0; }
  .log { font-size:.76rem; color:var(--muted); max-height:130px; overflow:auto; background:#f8fafc;
    border-radius:10px; padding:8px 10px; line-height:1.6; }
  .hint-bar { font-size:.78rem; color:var(--muted); }
  @media (max-width:640px){ .side{max-width:100%;} }
</style>
</head>
<body>
  <h1>⬤◯ AIオセロ</h1>
  <p class="sub">あなたは黒 ● 先手 — AI(白)と対戦します。光るマスに置けます。</p>
  <div class="wrap">
    <div class="panel board-panel">
      <div id="board"></div>
      <p class="hint-bar">💡 角を取ると強い / 序盤は少なめに返すのがコツ</p>
    </div>
    <div class="side">
      <div class="panel">
        <div id="status">あなたの番です</div>
        <div class="score" style="margin-top:10px">
          <div class="card" id="cardB"><div class="n" id="scoreB">2</div><div class="l">● あなた(黒)</div></div>
          <div class="card" id="cardW"><div class="n" id="scoreW">2</div><div class="l">○ AI(白)</div></div>
        </div>
        <div class="row" style="margin-top:12px">
          <label>AIの強さ
            <select id="level">
              <option value="1">よわい(1手読み)</option>
              <option value="2" selected>ふつう(2手読み)</option>
              <option value="3">つよい(3手読み)</option>
            </select>
          </label>
        </div>
        <div class="row" style="margin-top:10px">
          <button class="primary" id="restart">はじめから</button>
          <button class="ghost" id="passBtn">パス</button>
          <button class="ghost" id="hintBtn">ヒント</button>
        </div>
        <div class="log" id="log" style="margin-top:12px"></div>
      </div>
    </div>
  </div>
<script>
const N = 8, EMPTY=0, BLACK=1, WHITE=-1;
const DIRS=[[-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]];
// 位置の価値表 (角が最重要・C打ち/X打ちは危険)
const W=[[120,-20,20,5,5,20,-20,120],[-20,-40,-5,-5,-5,-5,-40,-20],[20,-5,15,3,3,15,-5,20],
[5,-5,3,3,3,3,-5,5],[5,-5,3,3,3,3,-5,5],[20,-5,15,3,3,15,-5,20],[-20,-40,-5,-5,-5,-5,-40,-20],[120,-20,20,5,5,20,-20,120]];
let board, turn, over=false, lock=false;
const $=id=>document.getElementById(id);
const elBoard=$("board"), elStatus=$("status"), elLog=$("log");

function init(){ board=Array.from({length:N},()=>Array(N).fill(EMPTY));
  board[3][3]=WHITE; board[3][4]=BLACK; board[4][3]=BLACK; board[4][4]=WHITE;
  turn=BLACK; over=false; lock=false; elLog.innerHTML=""; log("対局開始 — あなたは黒(先手)です。"); render(); }
function inside(r,c){ return r>=0&&r<N&&c>=0&&c<N; }
function flips(b,r,c,color){ if(b[r][c]!==EMPTY) return [];
  const out=[];
  for(const [dr,dc] of DIRS){ const line=[]; let i=r+dr,j=c+dc;
    while(inside(i,j)&&b[i][j]===-color){ line.push([i,j]); i+=dr; j+=dc; }
    if(line.length&&inside(i,j)&&b[i][j]===color) out.push(...line); }
  return out; }
function moves(b,color){ const m=[]; for(let r=0;r<N;r++)for(let c=0;c<N;c++) if(flips(b,r,c,color).length) m.push([r,c]); return m; }
function applyMove(b,r,c,color){ const f=flips(b,r,c,color); if(!f.length) return false;
  b[r][c]=color; for(const [i,j] of f) b[i][j]=color; return true; }
function count(b){ let x=0,o=0; for(const row of b)for(const v of row){ if(v===BLACK)x++; if(v===WHITE)o++; } return [x,o]; }
function evaluate(b){ let s=0,mobil=0;
  for(let r=0;r<N;r++)for(let c=0;c<N;c++){ if(b[r][c]===WHITE)s+=W[r][c]; else if(b[r][c]===BLACK)s-=W[r][c]; }
  mobil=(moves(b,WHITE).length-moves(b,BLACK).length)*6;
  return s+mobil; }
function minimax(b,depth,alpha,beta,maxing){ const ms=moves(b,maxing?WHITE:BLACK);
  if(!depth||!ms.length){ if(!ms.length&&moves(b,maxing?BLACK:WHITE).length) return minimax(b,depth,alpha,beta,!maxing);
    return evaluate(b); }
  if(maxing){ let best=-1e9; for(const [r,c] of ms){ const nb=b.map(row=>row.slice()); applyMove(nb,r,c,WHITE);
    best=Math.max(best,minimax(nb,depth-1,alpha,beta,false)); alpha=Math.max(alpha,best); if(beta<=alpha)break; } return best; }
  let best=1e9; for(const [r,c] of ms){ const nb=b.map(row=>row.slice()); applyMove(nb,r,c,BLACK);
    best=Math.min(best,minimax(nb,depth-1,alpha,beta,true)); beta=Math.min(beta,best); if(beta<=alpha)break; } return best; }
function aiMove(){ const depth=parseInt($("level").value,10)||2;
  const ms=moves(board,WHITE); if(!ms.length) return null;
  // 序盤は少しランダムに (毎回同じ展開にならない)
  const ordered=ms.map(([r,c])=>{ const nb=board.map(row=>row.slice()); applyMove(nb,r,c,WHITE);
    return {r,c,v:minimax(nb,depth-1,-1e9,1e9,false)+Math.random()*3}; }).sort((a,b)=>b.v-a.v);
  return ordered[0]; }
function render(hints=true){ elBoard.innerHTML="";
  const ms=hints&&!over&&turn===BLACK?moves(board,BLACK):[];
  const hintSet=new Set(ms.map(([r,c])=>r*8+c));
  for(let r=0;r<N;r++)for(let c=0;c<N;c++){ const d=document.createElement("div");
    d.className="cell"+(hintSet.has(r*8+c)?" hint":"");
    if(board[r][c]!==EMPTY){ const p=document.createElement("div"); p.className="disc "+(board[r][c]===BLACK?"b":"w"); d.appendChild(p); }
    d.onclick=()=>onClick(r,c); elBoard.appendChild(d); }
  const [x,o]=count(board); $("scoreB").textContent=x; $("scoreW").textContent=o;
  $("cardB").classList.toggle("me",turn===BLACK&&!over); $("cardW").classList.toggle("me",turn===WHITE&&!over); }
function log(t){ const p=document.createElement("div"); p.textContent=t; elLog.prepend(p); }
function setStatus(t){ elStatus.textContent=t; }
function checkEnd(){ const mb=moves(board,BLACK),mw=moves(board,WHITE);
  if(mb.length||mw.length) return false;
  over=true; const [x,o]=count(board);
  const msg=x===o?"引き分け!":x>o?`あなたの勝ち! ${x}対${o}`:`AIの勝ち… ${x}対${o}`;
  setStatus("終局 — "+msg); log("終局: "+msg); render(false); return true; }
function onClick(r,c){ if(over||lock||turn!==BLACK) return;
  if(!applyMove(board,r,c,BLACK)){ setStatus("そこには置けません"); return; }
  log(`あなた: (${"ABCDEFGH"[c]}${r+1}) に着手`); turn=WHITE; render(false);
  if(checkEnd()) return;
  if(!moves(board,WHITE).length){ log("AIはパスしました"); turn=BLACK; setStatus("AIがパス — あなたの番です"); render(); checkEnd(); return; }
  setStatus("AIが考え中…"); lock=true;
  setTimeout(()=>{ const m=aiMove();
    if(m){ applyMove(board,m.r,m.c,WHITE); log(`AI: (${"ABCDEFGH"[m.c]}${m.r+1}) に着手`); }
    turn=BLACK; lock=false;
    if(checkEnd()) return;
    if(!moves(board,BLACK).length){ turn=WHITE; log("あなたはパス(置ける場所なし)"); setStatus("置ける場所がありません — 自動でパスします");
      setTimeout(()=>{ turn=BLACK; if(moves(board,WHITE).length){ setStatus("AIが考え中…"); lock=true;
        setTimeout(()=>{ const m2=aiMove(); if(m2)applyMove(board,m2.r,m2.c,WHITE); lock=false; turn=BLACK;
          if(!checkEnd()){ setStatus("あなたの番です"); render(); } },420); } else checkEnd(); },700); return; }
    setStatus("あなたの番です"); render(); },420); }
$("restart").onclick=init;
$("passBtn").onclick=()=>{ if(over||lock||turn!==BLACK) return;
  if(moves(board,BLACK).length){ setStatus("まだ置ける場所があります"); return; }
  turn=WHITE; log("あなたはパスしました"); onClick(-1,-1); turn=WHITE; setStatus("AIが考え中…"); lock=true;
  setTimeout(()=>{ const m=aiMove(); if(m)applyMove(board,m.r,m.c,WHITE); lock=false; turn=BLACK;
    if(!checkEnd()){ setStatus("あなたの番です"); render(); } },420); };
$("hintBtn").onclick=()=>{ if(over||turn!==BLACK) return;
  const ms=moves(board,BLACK); if(!ms.length){ setStatus("置ける場所がありません"); return; }
  let best=null,bv=-1e18; for(const [r,c] of ms){ const nb=board.map(row=>row.slice()); applyMove(nb,r,c,BLACK);
    const v=-evaluate(nb); if(v>bv){ bv=v; best=[r,c]; } }
  setStatus(`ヒント: ${"ABCDEFGH"[best[1]]}${best[0]+1} がおすすめ`); };
init();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# 汎用 HTML アプリ
# ---------------------------------------------------------------------------

def _html_shell(title: str, body_html: str, style_extra: str = "", script: str = "") -> str:
    return f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  * {{ box-sizing:border-box; }}
  body {{ margin:0; min-height:100vh; font-family:"Hiragino Kaku Gothic ProN","Noto Sans JP",system-ui,sans-serif;
    background:linear-gradient(135deg,#0f2027,#203a43,#2c5364); color:#fff;
    display:flex; flex-direction:column; align-items:center; padding:32px 16px 60px; }}
  .card {{ background:rgba(255,255,255,.97); color:#1a2233; border-radius:18px; padding:24px 26px;
    width:min(92vw,560px); box-shadow:0 24px 60px rgba(0,0,0,.35); }}
  h1 {{ font-size:1.35rem; margin:0 0 12px; }}
  button {{ border:none; border-radius:10px; padding:9px 16px; font-weight:700; cursor:pointer;
    background:#1a2233; color:#fff; font:inherit; font-size:.9rem; }}
  button:hover {{ opacity:.88; }}
  input, select {{ font:inherit; padding:8px 10px; border-radius:9px; border:1px solid #cbd5e0; }}
  {style_extra}
</style>
</head>
<body>
  <div class="card">
{body_html}
  </div>
<script>
{script}
</script>
</body>
</html>
"""


def todo_html() -> str:
    body = """    <h1>✅ TODOリスト</h1>
    <div style="display:flex;gap:8px;margin-bottom:14px">
      <input id="inp" placeholder="やることを入力…" style="flex:1">
      <button id="add">追加</button>
    </div>
    <div id="list"></div>
    <p style="color:#5b6b82;font-size:.8rem;margin-top:12px">✔ ブラウザに自動保存されます(localStorage)。</p>"""
    script = """const inp=document.getElementById("inp"),list=document.getElementById("list");
let items=JSON.parse(localStorage.getItem("snipher-todo")||"[]");
function save(){localStorage.setItem("snipher-todo",JSON.stringify(items));}
function render(){list.innerHTML="";
  items.forEach((t,i)=>{const d=document.createElement("div");
    d.style.cssText="display:flex;gap:8px;align-items:center;padding:8px 10px;border:1px solid #e2e8f0;border-radius:10px;margin-bottom:8px;background:#f8fafc";
    const c=document.createElement("input");c.type="checkbox";c.checked=!!t.done;
    c.onchange=()=>{items[i].done=c.checked;save();render();};
    const s=document.createElement("span");s.textContent=t.text;s.style.cssText="flex:1"+(t.done?";text-decoration:line-through;color:#94a3b8":"");
    const b=document.createElement("button");b.textContent="削除";b.style.cssText="background:#fee2e2;color:#b91c1c;padding:5px 10px";
    b.onclick=()=>{items.splice(i,1);save();render();};
    d.append(c,s,b);list.appendChild(d);});
  if(!items.length)list.innerHTML='<p style="color:#94a3b8;font-size:.85rem">まだ何もありません。上の欄から追加してください。</p>';}
document.getElementById("add").onclick=()=>{const v=inp.value.trim();if(!v)return;items.push({text:v,done:false});inp.value="";save();render();};
inp.addEventListener("keydown",e=>{if(e.key==="Enter")document.getElementById("add").click();});
render();"""
    return _html_shell("TODOリスト — Snipher", body, script=script)


def calculator_html() -> str:
    body = """    <h1>🧮 電卓</h1>
    <input id="d" readonly style="width:100%;font-size:1.6rem;text-align:right;margin-bottom:12px;background:#0f172a;color:#fff;border:none;padding:14px">
    <div id="keys" style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px"></div>"""
    script = """const d=document.getElementById("d"),keys=document.getElementById("keys");
["7","8","9","/","4","5","6","*","1","2","3","-","0",".","=","+","C","(",")","⌫"].forEach(k=>{
  const b=document.createElement("button");b.textContent=k;
  b.style.cssText="padding:14px;font-size:1.05rem;"+(k==="="?"background:#0e7a4f":"")+(k==="C"?"background:#b91c1c":"");
  b.onclick=()=>{if(k==="="){try{d.value=Function('"use strict";return('+d.value+")")();}catch(e){d.value="エラー";}}
    else if(k==="C")d.value="";else if(k==="⌫")d.value=d.value.slice(0,-1);else d.value+=k;};
  keys.appendChild(b);});"""
    return _html_shell("電卓 — Snipher", body, script=script)


def timer_html() -> str:
    body = """    <h1>⏱ タイマー</h1>
    <div style="font-size:3rem;font-weight:800;text-align:center;margin:8px 0 14px" id="t">05:00</div>
    <div style="display:flex;gap:8px;justify-content:center;align-items:center">
      <input id="m" type="number" value="5" min="0" max="120" style="width:70px"> 分
      <button id="start">開始</button><button id="stop" style="background:#e2e8f0;color:#1a2233">停止</button>
      <button id="reset" style="background:#e2e8f0;color:#1a2233">リセット</button>
    </div>"""
    script = """let left=300,timer=null;const t=document.getElementById("t");
function fmt(s){return String(Math.floor(s/60)).padStart(2,"0")+":"+String(s%60).padStart(2,"0");}
function draw(){t.textContent=fmt(left);}
document.getElementById("m").onchange=e=>{left=Math.max(0,parseInt(e.target.value||"0",10)*60);draw();};
document.getElementById("start").onclick=()=>{if(timer)return;timer=setInterval(()=>{if(left>0){left--;draw();}else{clearInterval(timer);timer=null;t.textContent="終了!";}},1000);};
document.getElementById("stop").onclick=()=>{clearInterval(timer);timer=null;};
document.getElementById("reset").onclick=()=>{clearInterval(timer);timer=null;left=parseInt(document.getElementById("m").value||"5",10)*60;draw();};
draw();"""
    return _html_shell("タイマー — Snipher", body, script=script)


def generic_html(title: str, purpose: str) -> str:
    safe_title = (title or "Snipher App").strip()[:40] or "Snipher App"
    safe_purpose = (purpose or "やりたいこと").strip()[:120]
    body = f"""    <h1>✨ {safe_title}</h1>
    <p style="color:#5b6b82">「{safe_purpose}」のためのスターターです。このファイルを直接編集して育ててください。</p>
    <div style="display:flex;gap:8px;margin:12px 0">
      <input id="inp" placeholder="入力してみてください…" style="flex:1">
      <button id="go">実行</button>
    </div>
    <div id="out" style="background:#f1f5f9;border-radius:10px;padding:12px;min-height:3em;font-size:.92rem"></div>"""
    script = """const inp=document.getElementById("inp"),out=document.getElementById("out");
function run(){const v=inp.value.trim()||"(空入力)";
  out.innerHTML="<b>入力:</b> "+v.replace(/</g,"&lt;")+"<br><b>文字数:</b> "+v.length+"<br><b>逆順:</b> "+[...v].reverse().join("").replace(/</g,"&lt;");}
document.getElementById("go").onclick=run;
inp.addEventListener("keydown",e=>{if(e.key==="Enter")run();});
out.textContent="上の欄に入力して「実行」を押してください。";"""
    return _html_shell(f"{safe_title} — Snipher", body, script=script)


# ---------------------------------------------------------------------------
# Python / JavaScript などの実用スニペット
# ---------------------------------------------------------------------------

def python_snippet(task: str, request: str) -> str:
    if task == "fizzbuzz":
        return ('def fizzbuzz(n: int) -> None:\n    for i in range(1, n + 1):\n        if i % 15 == 0:\n            print("FizzBuzz")\n        elif i % 3 == 0:\n            print("Fizz")\n        elif i % 5 == 0:\n            print("Buzz")\n        else:\n            print(i)\n\nfizzbuzz(100)')
    if task == "fibonacci":
        return ('def fibonacci(n: int) -> list[int]:\n    a, b = 0, 1\n    out: list[int] = []\n    for _ in range(n):\n        out.append(a)\n        a, b = b, a + b\n    return out\n\nprint(fibonacci(10))')
    if task == "prime":
        return ('def is_prime(n: int) -> bool:\n    if n < 2:\n        return False\n    return all(n % d for d in range(2, int(n ** 0.5) + 1))\n\nprint([x for x in range(2, 50) if is_prime(x)])')
    if task == "sort":
        return ('def quicksort(a: list) -> list:\n    if len(a) <= 1:\n        return list(a)\n    pivot = a[len(a) // 2]\n    left = [x for x in a if x < pivot]\n    mid = [x for x in a if x == pivot]\n    right = [x for x in a if x > pivot]\n    return quicksort(left) + mid + quicksort(right)\n\nprint(quicksort([5, 3, 8, 1, 9, 2]))')
    if task == "scraper":
        return ('"""簡易スクレイパー (標準ライブラリのみ)。"""\nimport re\nimport urllib.request\n\nURL = "https://example.com"\nUA = {"User-Agent": "Mozilla/5.0 Snipher/2.0"}\n\nreq = urllib.request.Request(URL, headers=UA)\nwith urllib.request.urlopen(req, timeout=10) as res:\n    html = res.read().decode("utf-8", "replace")\n\ntitle = re.search(r"<title>(.*?)</title>", html, re.S)\nprint("title:", title.group(1).strip() if title else "(不明)")\nfor m in re.finditer(r"<a[^>]+href=[\\"\\\'](https?://[^\\"\\\']+)", html):\n    print(m.group(1))\n    break  # 最初の1件だけ表示 (全部見るなら break を外す)')
    if task == "api_server":
        return ('"""最小 API サーバー (標準ライブラリのみ・http.server)。"""\nimport json\nfrom http.server import BaseHTTPRequestHandler, HTTPServer\n\nclass Handler(BaseHTTPRequestHandler):\n    def do_GET(self):\n        body = json.dumps({"ok": True, "path": self.path}, ensure_ascii=False).encode()\n        self.send_response(200)\n        self.send_header("Content-Type", "application/json; charset=utf-8")\n        self.send_header("Content-Length", str(len(body)))\n        self.end_headers()\n        self.wfile.write(body)\n    def log_message(self, *a):\n        pass\n\nprint("http://localhost:8000 で起動 (Ctrl+Cで終了)")\nHTTPServer(("0.0.0.0", 8000), Handler).serve_forever()')
    # 汎用: 要求語を生かした雛形
    topic = _topic_of(request)
    return (f'"""{topic or "要求"}を処理する雛形。"""\n\ndef solve(value):\n    """ここに処理を書く。"""\n    if value is None:\n        return "入力が空です。solve() に値を渡してください。"\n    return value\n\n\nif __name__ == "__main__":\n    import sys\n    arg = sys.argv[1] if len(sys.argv) > 1 else None\n    print(solve(arg))\n')


def javascript_snippet(task: str, request: str) -> str:
    if task == "fizzbuzz":
        return ("function fizzBuzz(n) {\n  for (let i = 1; i <= n; i += 1) {\n    console.log(i % 15 === 0 ? 'FizzBuzz' : i % 3 === 0 ? 'Fizz' : i % 5 === 0 ? 'Buzz' : i);\n  }\n}\n\nfizzBuzz(100);")
    if task == "fibonacci":
        return ("function fibonacci(n) {\n  const out = [];\n  let [a, b] = [0, 1];\n  for (let i = 0; i < n; i += 1) { out.push(a); [a, b] = [b, a + b]; }\n  return out;\n}\n\nconsole.log(fibonacci(10));")
    if task == "prime":
        return ("function isPrime(n) {\n  if (n < 2) return false;\n  for (let d = 2; d * d <= n; d += 1) if (n % d === 0) return false;\n  return true;\n}\n\nconsole.log(Array.from({length: 48}, (_, i) => i + 2).filter(isPrime));")
    if task == "sort":
        return ("function quickSort(a) {\n  if (a.length <= 1) return [...a];\n  const pivot = a[a.length >> 1];\n  return [...quickSort(a.filter(x => x < pivot)), ...a.filter(x => x === pivot), ...quickSort(a.filter(x => x > pivot))];\n}\n\nconsole.log(quickSort([5, 3, 8, 1, 9, 2]));")
    topic = _topic_of(request)
    return (f"// {topic or '要求'}を処理する雛形\nexport function solve(value) {{\n  if (value == null) return '入力が空です。solve() に値を渡してください。';\n  return value;\n}}\n\nconsole.log(solve(process.argv[2] ?? null));")


def _topic_of(request: str) -> str:
    """依頼文から *作るものの名* だけを残す（言語名・依頼の動詞・敬語は落とす）。"""
    t = re.sub(r"\s+", " ", str(request or "")).strip()
    t = re.sub(r"(?:コード|解説|説明|コメント)\s*(?:と|も)?\s*(?:簡単な|かんたんな)?\s*"
               r"(?:コード|解説|説明)?\s*(?:を)?\s*(?:添えて|つけて|付けて).*$", "", t)
    t = re.sub(r"(を|の)?(書いて|書い|作って|作成して|実装して|ください|下さい|お願い|ちょうだい).*$", "", t)
    t = re.sub(r"^(?:Python|JavaScript|JS|TypeScript|TS|HTML|CSS|SQL|Java|Go|Rust|Ruby|PHP|bash)"
               r"\s*(?:で|に|の|を)?\s*", "", t, flags=re.IGNORECASE).strip()
    t = re.sub(r"^(?:関数|クラス|メソッド|スクリプト|プログラム|クエリ|ページ|サイト)\s*(?:を|の)?\s*", "", t).strip()
    return t[:40]


# ---------------------------------------------------------------------------
# 公開 API
# ---------------------------------------------------------------------------

def generate(request: str, *, lang: str | None = None) -> tuple[str, str, dict]:
    """要求から (表示テキスト, 言語, メタデータ) を作る。必ず何かを返す。

    ``lang`` を渡すと、その言語で書く（指示層が *指示文から読んだ言語* を引き継ぐ口）。
    """
    lang = (lang or "").lower() or detect_language(request)
    task = detect_task(request)
    topic = _topic_of(request) or "アプリ"

    if lang == "html":
        if task == "othello":
            code = othello_html()
            head = ("AIオセロ(対AI・単一ファイル)を作りました。このまま `index.html` に保存して"
                    "ブラウザで開けば遊べます。AIは2手読み+位置評価で、強さは画面で切替可能です。\n\n"
                    "遊び方: あなたは黒(先手)。光るマスをクリックで着手 → AIが自動で応答します。\n")
            return (head + f"```html\n{code}\n```", "html",
                    {"language": "html", "task": "othello", "snippet": code, "filename": "index.html"})
        if task == "todo":
            code = todo_html()
            return ("TODOリスト(ブラウザ保存付き)を作りました。`index.html` に保存して開けば使えます。\n\n```html\n" + code + "\n```",
                    "html", {"language": "html", "task": "todo", "snippet": code, "filename": "index.html"})
        if task == "calculator":
            code = calculator_html()
            return ("電卓アプリを作りました。`index.html` に保存して開けば使えます。\n\n```html\n" + code + "\n```",
                    "html", {"language": "html", "task": "calculator", "snippet": code, "filename": "index.html"})
        if task == "timer":
            code = timer_html()
            return ("タイマーを作りました。`index.html` に保存して開けば使えます。\n\n```html\n" + code + "\n```",
                    "html", {"language": "html", "task": "timer", "snippet": code, "filename": "index.html"})
        code = generic_html(topic, request)
        return (f"「{topic}」のHTMLスターターを作りました。`index.html` に保存して開けば動きます。"
                "気に入らない部分があれば「○○を追加して」と続けてください。\n\n```html\n" + code + "\n```",
                "html", {"language": "html", "task": "generic", "snippet": code, "filename": "index.html"})

    if lang in ("javascript", "typescript"):
        block = "typescript" if lang == "typescript" else "javascript"
        code = javascript_snippet(task, request)
        note = "実行: `node app.js` (ブラウザのコンソールでも可)。"
        if task == "generic":
            note += "入力例と期待する出力を添えれば、仕様に合わせて具体化します。"
        return (f"```{block}\n{code}\n```\n{note}", block,
                {"language": block, "task": task, "snippet": code})

    if lang == "java":
        code = ("public class Main {\n    public static void main(String[] args) {\n"
                "        System.out.println(\"Hello, Snipher!\");\n    }\n}")
        return (f"```java\n{code}\n```\n実行: `javac Main.java && java Main`。",
                "java", {"language": "java", "task": task, "snippet": code})

    if lang == "go":
        code = 'package main\n\nimport "fmt"\n\nfunc main() {\n    fmt.Println("Hello, Snipher!")\n}'
        return (f"```go\n{code}\n```\n実行: `go run main.go`。",
                "go", {"language": "go", "task": task, "snippet": code})

    if lang == "rust":
        code = 'fn main() {\n    println!("Hello, Snipher!");\n}'
        return (f"```rust\n{code}\n```\n実行: `rustc main.rs && ./main`。",
                "rust", {"language": "rust", "task": task, "snippet": code})

    # 既定: Python
    code = python_snippet(task, request)
    note = "実行: `python app.py`。"
    if task == "generic":
        note += "入力例と期待する出力を添えれば、仕様に合わせて具体化します。"
    elif task == "fizzbuzz":
        note += "3と5の公倍数を先に判定するのがポイントです。"
    return (f"```python\n{code}\n```\n{note}", "python",
            {"language": "python", "task": task, "snippet": code})


__all__ = ["detect_language", "detect_task", "is_code_request", "generate", "othello_html"]
