// Cloudflare Pages Function — /api/* をさばく 1 本。
//
//  1) `SNIPHER_API_ORIGIN` が設定されていれば、そのまま本体 API に流す（全機能）
//  2) 無ければ、Pages に同梱した `engine.mjs` + `kb.json` で答える（静的デモ）
//
// SSE の形（start / delta / done）は FastAPI 版と揃えてあるので、UI は同じコードで動きます。

import { respond, status, Index } from "../_engine/engine.mjs";

const CORS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "Content-Type",
  "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
};

let cachedIndex = null;

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

// 1 応答を SSE に流す（本文は行単位で刻む。本体 API と同じイベント形）
function sseReply(build) {
  const enc = new TextEncoder();
  const stream = new ReadableStream({
    async start(controller) {
      const send = (obj) => controller.enqueue(enc.encode(`data: ${JSON.stringify(obj)}\n\n`));
      try {
        send({ type: "start", engine: "Snipher pages (lite)" });
        const out = await build();
        const text = String(out.text || "");
        for (const piece of text.split(/(?<=\n)/)) {
          if (piece) send({ type: "delta", text: piece });
        }
        send({
          type: "done", text,
          stats: {
            route: "pages", engine: out.engine || "snipher-pages-lite", plan: out.plan,
            confidence: out.confidence, knowledge: out.knowledge || {}, sources: out.sources || [],
          },
        });
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
    return jsonReply(status(idx));
  }

  let body = {};
  if (request.method === "POST") {
    try { body = await request.json(); } catch (e) { body = {}; }
  }

  if (url.pathname === "/api/chat") {
    return sseReply(async () => {
      const idx = await localIndex(env);
      return respond(body.messages || [], {
        index: idx, turn: Number(body.turn || 1), style: body.style || {},
      });
    });
  }

  if (url.pathname === "/api/steer") {
    // Pages 静的デモではステアリングは no-op だが 200 を返す (前端の生成を止めない)
    return jsonReply({ ok: true, queued: String((body && body.text) || ""), mode: "pages-lite" });
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
