// Cloudflare Pages のローカルプレビュー（デプロイ時は使いません）。
// `public/` を静的に配り、Pages Function と *同じ* /api/status・/api/chat を
// engine.mjs で応答して再現します。本体 API に転送したいときは SNIPHER_API_ORIGIN を設定。
//
//   node tools/preview_pages.mjs            # http://127.0.0.1:8788
//   PORT=8788 node tools/preview_pages.mjs
//   SNIPHER_API_ORIGIN=http://127.0.0.1:8000 node tools/preview_pages.mjs

import { createServer } from "node:http";
import { readFile, stat } from "node:fs/promises";
import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { respond, status, Index } from "../functions/_engine/engine.mjs";

const here = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(here, "..");
const PUBLIC = path.join(ROOT, "public");
const PORT = Number(process.env.PORT || 8788);
const ORIGIN = process.env.SNIPHER_API_ORIGIN || "";

const MIME = { ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8", ".json": "application/json; charset=utf-8",
  ".css": "text/css; charset=utf-8", ".svg": "image/svg+xml", ".ico": "image/x-icon" };

let index = null;
async function kbIndex() {
  if (index) return index;
  const raw = await readFile(path.join(PUBLIC, "kb.json"), "utf8");
  index = new Index(JSON.parse(raw));
  return index;
}

async function proxy(req, res) {
  const url = new URL(req.url, "http://localhost");
  const headers = { ...req.headers, host: new URL(ORIGIN).host };
  delete headers.connection;
  const chunks = [];
  for await (const c of req) chunks.push(c);
  const upstream = await fetch(new URL(url.pathname + url.search, ORIGIN.replace(/\/$/, "")), {
    method: req.method, headers, body: chunks.length ? Buffer.concat(chunks) : undefined,
  });
  res.writeHead(upstream.status, { "content-type": upstream.headers.get("content-type") || "application/json" });
  res.end(Buffer.from(await upstream.arrayBuffer()));
}

const server = createServer(async (req, res) => {
  const url = new URL(req.url, `http://127.0.0.1:${PORT}`);
  try {
    if (url.pathname.startsWith("/api/") && ORIGIN) return await proxy(req, res);

    if (url.pathname === "/api/status") {
      res.writeHead(200, { "content-type": MIME[".json"] });
      return res.end(JSON.stringify({ ...status(await kbIndex()), origin: ORIGIN || "pages-lite" }));
    }

    if (url.pathname === "/api/chat") {
      const chunks = [];
      for await (const c of req) chunks.push(c);
      const body = chunks.length ? JSON.parse(Buffer.concat(chunks).toString("utf8") || "{}") : {};
      const idx = await kbIndex();
      const out = respond(body.messages || [], { index: idx, turn: Number(body.turn || 1),
                                                 style: body.style || {} });
      res.writeHead(200, { "content-type": MIME[".html"].replace("text/html", "text/event-stream"),
                          "cache-control": "no-store" });
      const send = (obj) => res.write(`data: ${JSON.stringify(obj)}\n\n`);
      send({ type: "start", engine: "Snipher pages (lite)" });
      const text = String(out.text || "");
      for (const piece of text.split(/(?<=\n)/)) if (piece) send({ type: "delta", text: piece });
      send({ type: "done", text, stats: { route: "pages", plan: out.plan,
              confidence: out.confidence, knowledge: out.knowledge || {}, sources: out.sources || [],
              engine: "snipher-pages-lite" } });
      return res.end();
    }

    let file = path.join(PUBLIC, url.pathname === "/" ? "index.html" : url.pathname);
    if (!file.startsWith(PUBLIC) || !existsSync(file) || (await stat(file)).isDirectory()) {
      file = path.join(PUBLIC, "index.html");
    }
    const body = await readFile(file);
    res.writeHead(200, { "content-type": MIME[path.extname(file)] || "application/octet-stream" });
    res.end(body);
  } catch (err) {
    res.writeHead(500, { "content-type": "text/plain; charset=utf-8" });
    res.end(String(err && err.message || err));
  }
});

server.listen(PORT, "0.0.0.0", () => {
  console.log(`Snipher pages preview → http://0.0.0.0:${PORT}  (api origin: ${ORIGIN || "pages-lite"})`);
});
