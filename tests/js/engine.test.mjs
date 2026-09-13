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
