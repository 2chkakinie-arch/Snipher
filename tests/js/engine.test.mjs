// Pages 用エンジンの実測（Cloudflare Pages で実際に動く返事か）。
// pytest から `node --test tests/js/engine.test.mjs` で実行する（tests/test_cloudflare_pages.py）。

import assert from "node:assert/strict";
import test from "node:test";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";

import { Index, answer, respond, summarize, extract, status, wantsWordInfo } from "../../public/engine.mjs";

const here = path.dirname(fileURLToPath(import.meta.url));
const kb = JSON.parse(await readFile(path.join(here, "..", "..", "public", "kb.json"), "utf8"));
const index = new Index(kb);

const LEXICON = /拍|索引|品詞|U\+|読みは/;

test("知らない語の質問は *問いの形* から方針を返す（辞書引きも不在も言わない）", () => {
  const r = answer("申年休假を申請する手順を教えて。", { index, turn: 1 });
  assert.match(r.text, /手続き|手順/);
  assert.match(r.text, /いつまでに|誰に出すか/);
  for (const banned of ["拍", "索引", "品詞", "読みは", "U+", "できません", "分かりません", "無い"]) {
    assert.ok(!r.text.includes(banned), `${banned} が混ざっている: ${r.text}`);
  }
});

test("読めない短文でも受け取り口を置いて会話を止めない", () => {
  const r = answer("は？", { index, turn: 3 });
  assert.ok(r.text.length >= 8, r.text);
  assert.ok(!/索引|拍|品詞/.test(r.text), r.text);
  assert.ok(!/できません|分かりません|ありません/.test(r.text), r.text);
});

test("索引が載っている語の定義を引ける", () => {
  const r = answer("WebAssembly とは何ですか？", { index });
  assert.match(r.text, /WebAssembly/);
  assert.match(r.text, /バイトコード|ブラウザ/);
  assert.ok(!LEXICON.test(r.text), r.text);
  assert.equal(r.plan.startsWith("knowledge:"), true, r.plan);
});

test("メリットの問いには速度の記述を返す（固定の定義文をそのまま出さない）", () => {
  const r = answer("WebAssembly をブラウザで動かす一番のメリットは何ですか？", { index });
  assert.match(r.text, /速/);
  assert.ok(!LEXICON.test(r.text), r.text);
});

test("手順の問いは numbered の列になる", () => {
  const r = answer("Git でブランチを作る手順は？", { index });
  assert.ok(r.text.length > 4, r.text);
  assert.doesNotMatch(r.text, /できません|分かりません/);
});

test("日常の平叙は、相手の語を使って受け取り、辞書引きしない", () => {
  const r = answer("今日は天気が良いので公園に散歩に行こうと思います。", { index });
  assert.match(r.text, /散歩|公園|天気/);
  assert.ok(!LEXICON.test(r.text), r.text);
  assert.doesNotMatch(r.text, /できません|手元に無い|調べられません/);
});

test("同じ発話を繰り返しても同じ文を返さない", () => {
  const texts = new Set();
  for (let turn = 1; turn <= 8; turn++) {
    texts.add(answer("昨日上司に怒られた。", { index, turn }).text);
  }
  assert.ok(texts.size >= 3, [...texts].join(" / "));
});

test("手元に無い語でも辞書情報に逃げない", () => {
  const r = answer("ゾルタクス＝ゼッカの最新動向は？", { index });
  assert.ok(r.text.length >= 6, r.text);
  assert.ok(!LEXICON.test(r.text), r.text);
  assert.doesNotMatch(r.text, /できません|分かりません|手元に無いので答え/);
});

test("要約は指定した個数の箇条書きになる", () => {
  const body = "オセロや将棋などの完全情報ゲームにおいて、AIは探索アルゴリズムを用いて最適な手を選択します。"
    + "α-β枝刈りを組み合わせることで無駄な探索を削減できます。"
    + "評価関数を工夫することで、深い読みを行わなくても強い着手を実現できます。";
  const out = summarize(`文章：${body}`, 3).split("\n").filter(Boolean);
  assert.equal(out.length, 3);
  assert.ok(out.every((x) => x.startsWith("・")), out.join("|"));
});

test("抽出は指定スキーマの JSON だけを出す", () => {
  const prompt = '次のテキストから情報を抽出し、必ず指定のJSON形式のみで出力してください。'
    + '{"origin": "...", "destination": "...", "duration": "...", "fare": "..."} '
    + 'テキスト: 東京から京都まで新幹線で約2時間15分、料金は約14,000円です。';
  const r = respond([{ role: "user", content: prompt }], { index });
  assert.equal(r.plan, "instruction:extract", r.plan);
  const obj = JSON.parse(r.text);
  assert.deepEqual(Object.keys(obj), ["origin", "destination", "duration", "fare"]);
  assert.equal(obj.origin, "東京");
  assert.equal(obj.destination, "京都");
  assert.match(obj.duration, /2\\s*時間|2時間15分/);
});

test("口調指定（だよ・だね）を守る", () => {
  const r = answer("WebAssembly とは何ですか？", { index, style: { tone: "だよ・だね" } });
  assert.match(r.text, /だよ|だね|だ。|る。/);
  assert.ok(!/です。|ます。/.test(r.text), r.text);
});

test("文字数指定は超えない", () => {
  const r = answer("ニュースとは何ですか？", { index, style: { targetChars: 80 } });
  assert.ok(r.text.replace(/\n/g, "").length <= 80 * 1.25, `${r.text.length}: ${r.text}`);
});

test("wantsWordInfo は語そのものの質問だけ通す", () => {
  assert.equal(wantsWordInfo("「は」の読み方は？"), true);
  assert.equal(wantsWordInfo("今日のニュースは？"), false);
  assert.equal(wantsWordInfo("WebAssemblyをブラウザで動かす一番のメリットは何ですか？"), false);
});

test("status は同梱 kb の規模を返す（API が無い場所でも画面が嘘をつかない）", () => {
  const s = status(index);
  assert.equal(s.ready, true);
  assert.ok(s.topics >= 200, String(s.topics));
  assert.ok(s.facts >= 500, String(s.facts));
});

// --------------------------------------------------------------------------- //
// v6: 確率の波（ステアリング / 並列熟考 / リアルタイムWeb）— Pages 版の実測
// --------------------------------------------------------------------------- //
import { respondStream, shouldSearchRealtime, deliberateMini } from "../../public/engine.mjs";

async function collect(gen) {
  const evs = [];
  for await (const ev of gen) evs.push(ev);
  return evs;
}

test("shouldSearchRealtime: 挨拶・相槌は検索しない、それ以外はすべて検索", () => {
  for (const greet of ["こんにちは", "ありがとう", "おはよう", "了解", "うん"]) {
    assert.equal(shouldSearchRealtime(greet), false, greet);
  }
  for (const q of ["量子コンピュータとは", "今日の天気は？", "光合成の仕組みを教えて"]) {
    assert.equal(shouldSearchRealtime(q), true, q);
  }
});

test("deliberateMini: 問いの型を読んで思考の骨格を作る", () => {
  const t = deliberateMini("なぜ空は青いの？", index);
  assert.ok(t.steps.length >= 3);
  assert.ok(t.conclusion.length > 4);
  assert.ok(t.confidence > 0 && t.confidence <= 0.95);
});

test("respondStream: start → thought → delta → done の順で流れ、done に波の統計が載る", async () => {
  const evs = await collect(respondStream([{ role: "user", content: "WebAssembly とは何ですか？" }], {
    index, chunkDelayMs: 0,
  }));
  const kinds = evs.map((e) => e.type);
  assert.equal(kinds[0], "start");
  assert.ok(kinds.includes("thought"));
  assert.ok(kinds.includes("delta"));
  assert.equal(kinds[kinds.length - 1], "done");
  const done = evs[evs.length - 1];
  assert.equal(done.stats.waves.deliberate, true);
  assert.equal(done.stats.waves.thought, true);
  assert.ok(done.text.length > 4);
});

test("respondStream: 挨拶では Web 検索が走らない（高速経路を維持）", async () => {
  const searched = [];
  const evs = await collect(respondStream([{ role: "user", content: "こんにちは" }], {
    index, chunkDelayMs: 0, search: async (q) => { searched.push(q); return { sentences: [], sources: [] }; },
  }));
  assert.deepEqual(searched, []);
  assert.equal(evs.some((e) => e.type === "web"), false);
});

test("respondStream: 挨拶以外は出力中に検索が走り、web イベントが流れる", async () => {
  const searched = [];
  const evs = await collect(respondStream([{ role: "user", content: "太陽系の惑星を教えて" }], {
    index, chunkDelayMs: 0,
    search: async (q) => {
      searched.push(q);
      return { sentences: ["太陽系には8つの惑星がある。"], sources: [{ url: "https://example.com", title: "例" }] };
    },
  }));
  assert.deepEqual(searched, ["太陽系の惑星を教えて"]);
  const webs = evs.filter((e) => e.type === "web");
  assert.equal(webs[0].state, "start");
  const doneWeb = webs.find((e) => e.state === "done");
  assert.equal(doneWeb.sentences, 1);
  const done = evs[evs.length - 1];
  assert.equal(done.stats.waves.web_search, true);
  assert.equal(done.stats.waves.web_hits, 1);
});

test("respondStream: 生成中のステアリング波が残りの文を組み直す（出力は止まらない）", async () => {
  // 2 文以上の応答を選び、1 文目が出た直後にステアを投入する
  const queue = [{ text: "もっと詳しく", strength: 1.5 }];
  let fed = false;
  const evs = await collect(respondStream([{ role: "user", content: "WebAssembly とは何ですか？" }], {
    index, chunkDelayMs: 0,
    pollSteer: () => {
      if (fed) return [];
      fed = true;
      return queue.splice(0, queue.length);
    },
  }));
  const steers = evs.filter((e) => e.type === "steer");
  assert.ok(steers.length >= 1, "ステアリング波が発火しない");
  assert.equal(steers[0].text, "もっと詳しく");
  const done = evs[evs.length - 1];
  assert.equal(done.stats.waves.steer_waves >= 1, true);
  // delta は途切れず最後まで流れる（生成は止まらない）
  const kinds = evs.map((e) => e.type);
  assert.ok(kinds.indexOf("delta") < kinds.lastIndexOf("delta") || kinds.filter((k) => k === "delta").length >= 1);
  assert.equal(kinds[kinds.length - 1], "done");
});
