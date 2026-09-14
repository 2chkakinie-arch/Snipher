// Cloudflare Pages Function — /api/* をさばく 1 本。
//
//  1) `SNIPHER_API_ORIGIN` が設定されていれば、そのまま本体 API に流す（全機能）
//  2) 無ければ、Pages に同梱した `engine.mjs` + `kb.json` で答える（波付き静的デモ）
//
// v6: 静的デモでも「確率の波」が動く。
//   - /api/steer は isolate 内のステアリングキューへ積まれ、生成中の応答が
//     チャンク間で拾って残りの文を *その場で組み直す*（出力は止めない）
//   - 挨拶以外の全プロンプトで出力中に Web 検索が走り、結果が波として干渉する
//   - 並列熟考（deliberateMini）の思考過程が thought イベントとして流れる
//
// SSE の形（start / delta / done + web / thought / steer）は FastAPI 版と揃えて
// あるので、UI は同じコードで動きます。

import { respond, respondStream, status, Index } from "../_engine/engine.mjs";

const CORS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "Content-Type",
  "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
};

let cachedIndex = null;

// ステアリングキュー（isolate 内で生存。生成中の応答がチャンク間で拾う）
const steerQueue = [];

// 静的アセットは Pages Function からも env.ASSETS で読める（追加のストレージ不要）
async function localIndex(env) {
  if (cachedIndex) return cachedIndex;
  const res = await env.ASSETS.fetch(new Request("https://assets.local/kb.json"));
  if (!res.ok) throw new Error("kb.json がデプロイされていません（python tools/build_public.py）");
  cachedIndex = new Index(await res.json());
  return cachedIndex;
}

function jsonReply(obj, status = 200) {
  return new Response(JSON.stringify(obj), {
    status,
    headers: { "Content-Type": "application/json; charset=utf-8", ...CORS },
  });
}

// リアルタイム Web 検索（ベストエフォート）。Workers からの fetch で SERP を叩き、
// スニペットを証拠文として返す。失敗しても生成は止めない（波は 0 件で継続）。
async function webSearch(query, env) {
  if (env && env.SNIPHER_WEB === "off") return { sentences: [], sources: [] };
  const endpoint = (env && env.SNIPHER_SERP_URL)
    || "https://html.duckduckgo.com/html/?q=";
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), 2500);
  try {
    const res = await fetch(endpoint + encodeURIComponent(query), {
      signal: ctl.signal,
      headers: { "User-Agent": "Mozilla/5.0 (compatible; Snipher/6.0)" },
    });
    if (!res.ok) return { sentences: [], sources: [] };
    const html = await res.text();
    const sentences = [];
    const sources = [];
    const re = /<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>([\s\S]*?)<\/a>[\s\S]*?(?:<a[^>]+class="result__snippet"[^>]*>([\s\S]*?)<\/a>)?/g;
    let m;
    while ((m = re.exec(html)) !== null && sources.length < 4) {
      const url = decodeURIComponent(String(m[1]).replace(/^.*uddg=/, "").replace(/&rut=.*$/, ""));
      const title = String(m[2] || "").replace(/<[^>]+>/g, "").trim();
      const snippet = String(m[3] || "").replace(/<[^>]+>/g, "").replace(/&amp;/g, "&").trim();
      if (!/^https?:/.test(url)) continue;
      sources.push({ url, title });
      if (snippet.length >= 12) sentences.push(snippet);
    }
    return { sentences, sources };
  } catch (e) {
    return { sentences: [], sources: [] };
  } finally {
    clearTimeout(timer);
  }
}

// 波付き SSE: respondStream のイベントをそのまま流す
function sseStream(gen) {
  const enc = new TextEncoder();
  const stream = new ReadableStream({
    async start(controller) {
      const send = (obj) => controller.enqueue(enc.encode(`data: ${JSON.stringify(obj)}\n\n`));
      try {
        for await (const ev of gen) send(ev);
      } catch (err) {
        send({ type: "error", message: String((err && err.message) || err) });
      } finally {
        controller.close();
      }
    },
  });
  return new Response(stream, {
    headers: {
      "Content-Type": "text/event-stream; charset=utf-8",
      "Cache-Control": "no-store",
      ...CORS,
    },
  });
}

async function proxy(request, env, origin) {
  const url = new URL(request.url);
  const target = new URL(url.pathname + url.search, String(origin).replace(/\/$/, ""));
  const headers = new Headers(request.headers);
  headers.delete("host");
  const init = {
    method: request.method,
    headers,
    body: request.method === "GET" || request.method === "HEAD" ? null : request.body,
    duplex: "half",
  };
  const res = await fetch(target, init);
  return new Response(res.body, {
    status: res.status, statusText: res.statusText, headers: res.headers,
  });
}

export async function onRequest(context) {
  const { request, env, next } = context;
  const url = new URL(request.url);
  if (request.method === "OPTIONS") return new Response(null, { status: 204, headers: CORS });

  // 本体（Python API）が生きていれば、そちらを先に使う。落ちたら静的デモに落ちる。
  if (env && env.SNIPHER_API_ORIGIN) {
    try {
      return await proxy(request, env, env.SNIPHER_API_ORIGIN);
    } catch (err) {
      /* fall through to the bundled engine */
    }
  }

  if (url.pathname === "/api/status") {
    let idx = null;
    try { idx = await localIndex(env); } catch (e) { idx = null; }
    return jsonReply({ ...status(idx), steering: { pending: steerQueue.length } });
  }

  let body = {};
  if (request.method === "POST") {
    try { body = await request.json(); } catch (e) { body = {}; }
  }

  if (url.pathname === "/api/chat") {
    const idx = await localIndex(env).catch(() => null);
    steerQueue.length = 0;
    const search = (env && env.SNIPHER_WEB !== "off")
      ? (q) => webSearch(q, env)
      : null;
    return sseStream(respondStream(body.messages || [], {
      index: idx,
      turn: Number(body.turn || 1),
      style: body.style || {},
      pollSteer: () => steerQueue.splice(0, steerQueue.length),
      search,
    }));
  }

  if (url.pathname === "/api/steer") {
    // 生成中でもノンストップで受け付け、確率波としてキューへ積む（出力は止まらない）
    const text = String((body && body.text) || "").trim();
    if (!text) return jsonReply({ ok: false, error: "text が空です" });
    steerQueue.push({ text, strength: Number((body && body.strength) || 1.0), at: Date.now() });
    return jsonReply({ ok: true, queued: text.slice(0, 80), mode: "pages-waves",
                       active_waves: steerQueue.length });
  }

  if (url.pathname === "/api/answer") {
    try {
      const idx = await localIndex(env);
      const out = respond([{ role: "user", content: String(body.text || "") }], {
        index: idx, turn: 1, style: body.style || {},
      });
      return jsonReply(out);
    } catch (err) {
      return jsonReply({ error: String((err && err.message) || err) }, 500);
    }
  }

  return next();
}
