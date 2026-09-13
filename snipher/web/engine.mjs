// Snipher — Pages 用エンジン（検索 → 証拠 → 文の組み立て）
//
// Python 版 `snipher/knowledge.py` + `snipher/mind/voice.py` と同じ考え方の移植です。
//  1) 発話から語を引き抜いて索引を数え、*実際に語が重なった* 話題だけを使う
//  2) 問いの型（定義・理由・手順・長所・いつ・どこ・誰・値段）で引く欄を決める
//  3) 証拠の文を *その場でつないで* 答える（固定の定型文を返さない）
//  4) 辞書の語彙情報（読み・拍・品詞）は、語そのものを尋ねる質問にだけ使う
//
// Pages Function とブラウザの両方から import する（1 本で両方で動く）。

const NFKC = (s) => String(s || "").normalize("NFKC");
const OPEN_SENT = /[。！？!?]$/;

export function normalize(s) {
  let t = NFKC(String(s || "")).toLowerCase();
  t = t.replace(/[\s\u3000]+/g, " ").replace(/[「」『』]/g, "");
  return t.trim();
}

const LATIN = /[a-z][a-z0-9.+#_-]{1,}/g;
const KANJI = /[\u4e00-\u9fff々]{2,8}/g;
const KANA = /[\u30a1-\u30faー]{2,10}/g;
const HIRA = /[\u3041-\u3095]{2,8}/g;

export function tokens(s) {
  const t = normalize(s);
  const out = [];
  for (const re of [LATIN, KATA_RE(), KANJI, HIRA]) {
    for (const m of t.match(re) || []) out.push(m);
  }
  return out;
}
function KATA_RE() { return KANA; }

export function bigrams(s) {
  const t = normalize(s).replace(/[^0-9a-z\u3041-\u30fa\u4e00-\u9fff]/g, "");
  const g = new Set();
  for (let i = 0; i + 2 <= t.length; i++) g.add(t.slice(i, i + 2));
  return g;
}

function dice(a, b) {
  const ga = bigrams(a), gb = bigrams(b);
  if (!ga.size || !gb.size) return 0;
  let n = 0;
  for (const x of ga) if (gb.has(x)) n++;
  return (2 * n) / (ga.size + gb.size);
}

const FUNC_WORDS = new Set(["の", "に", "は", "を", "が", "で", "と", "も", "や", "から", "まで",
  "より", "など", "それ", "この", "その", "あの", "です", "ます", "した", "する", "れる", "られない",
  "What", "what", "the", "and", "for", "with", "について", "相关的"]);

const QTYPE = [
  ["why", /なぜ|どうして|なんで|理由|仕組み|しくみ/],
  ["how", /作り方|やり方|手順|方法|コツ|使い方|選び方|書き方|対策|どうすれば|どうやっ|覚え方/],
  ["pros_cons", /メリット|デメリット|利点|短所|長所|欠点|弱点|強み|弱み|どこが良い|何が悪い/],
  ["cost", /いくら|値段|価格|費用|料金|コスト|何円/],
  ["when", /いつ|何時|時期|何月|季節|旬|タイミング|どれくらい|どのくらい|何分|何時間|何日|何年/],
  ["where", /どこ|何処|場所|どのあたり/],
  ["who", /だれ|誰|何人/],
  ["opinion", /好き|嫌い|おすすめ|どう思う|感想|楽しい|興味/],
  ["count", /いくつ|何個|何種類|何匹|何本/],
  ["definition", /とは|って何|何ですか|なんですか|どういうもの|どんなもの|意味|定義|どんな|なんて言う/],
];

export function questionType(text) {
  const t = normalize(text);
  for (const [kind, re] of QTYPE) if (re.test(t)) return kind;
  if (/[?？]\s*$/.test(t) || /か\s*$/.test(t)) return "general_q";
  return "general";
}

// 語そのものを尋ねる質問のときだけ辞書情報を使ってよい（それ以外は内容で答える）
const WORD_INFO = /とは|って何|読み方|何と読む|なんて読む|ふりがな|何文字|何拍|画数|品詞|活用|スペル|綴り|意味は|語義/;
export function wantsWordInfo(text) {
  const t = normalize(text).replace(/[。！？]+$/g, "");
  return t.length <= 34 && WORD_INFO.test(t);
}

// --------------------------------------------------------------------------- //
// 索引
// --------------------------------------------------------------------------- //
export class Index {
  constructor(payload) {
    this.meta = (payload && payload.meta) || {};
    this.items = (payload && payload.items) || [];
    this.words = this.items.map((it) => {
      const bag = [it.topic, ...(it.aliases || [])].join(" ");
      const body = [it.def || "", ...(it.facts || []), ...(it.why || [])].join(" ");
      return {
        bag: new Set(tokens(bag).filter((w) => !FUNC_WORDS.has(w))),
        body: new Set(tokens(body).filter((w) => !FUNC_WORDS.has(w))),
      };
    });
    this.avgLen = Math.max(1, this.items.reduce((a, it) => a + (this.words[this.items.indexOf(it)].body.size || 1), 0) / Math.max(1, this.items.length));
  }

  search(query, topK = 4) {
    const qn = normalize(query);
    const qWords = new Set(tokens(query).filter((w) => !FUNC_WORDS.has(w)));
    const out = [];
    for (let i = 0; i < this.items.length; i++) {
      const it = this.items[i];
      const w = this.words[i];
      let covered = 0;
      for (const x of qWords) if (w.body.has(x) || w.bag.has(x)) covered++;
      const coverage = qWords.size ? covered / qWords.size : 0;
      let topicHit = false;
      for (const name of [it.topic, ...(it.aliases || [])]) {
        const n = normalize(name);
        if (n.length >= 2 && qn.includes(n)) { topicHit = true; break; }
      }
      let score = 0.55 * coverage + 0.35 * dice(qn, it.topic || "");
      if (topicHit) score += 1.0;
      if (covered >= 2) score += 0.25;
      if (score <= 0.05) continue;
      out.push({ item: it, score, coverage, topicHit, covered });
    }
    out.sort((a, b) => b.score - a.score);
    return out.slice(0, topK);
  }

  // 発話に *実際に含まれる* 語を踏んだ話題だけを返す（薄い語の取り違えを防ぐ）
  topicsFor(query, { min = 0.42, limit = 3 } = {}) {
    return this.search(query, limit * 2)
      .filter((h) => h.score >= min && (h.topicHit || h.covered >= 1))
      .slice(0, limit)
      .map((h) => h.item);
  }

  exactTopic(name) {
    const n = normalize(name);
    if (!n) return null;
    for (const it of this.items) {
      if (normalize(it.topic) === n) return it;
      if ((it.aliases || []).some((a) => normalize(a) === n)) return it;
    }
    return null;
  }
}

// --------------------------------------------------------------------------- //
// 1 話題 → 答える文
// --------------------------------------------------------------------------- //
function pickField(item, qtype, query) {
  const first = (xs) => (Array.isArray(xs) ? xs.find((x) => String(x || "").trim()) || "" : String(xs || "").trim());
  const qa = (item.qa || []).filter(([q, a]) => String(q || "").trim() && String(a || "").trim());
  const qaHit = (() => {
    const qn = normalize(query);
    let best = null;
    for (const [q, a] of qa) {
      const s = dice(qn, String(q));
      const shared = tokens(q).some((t) => t.length >= 2 && qn.includes(t));
      if (shared && (!best || s > best[0])) best = [s, String(a).trim()];
    }
    return best && best[0] >= 0.18 ? best[1] : "";
  })();
  if (qaHit && (qtype === "general" || qtype === "general_q")) return [qaHit, "qa"];
  if (qtype === "pros_cons") {
    const marks = ["メリット", "デメリット", "利点", "短所", "長所", "欠点", "弱点", "強み", "弱み"];
    for (const [q, a] of qa) {
      if (marks.some((m) => String(q).includes(m)) && marks.some((m) => query.includes(m))) {
        return [String(a).trim(), "qa"];
      }
    }
    return [first(item.tips) || item.opinion || first(item.def) || first(item.facts), "tips"];
  }
  if (qtype === "why") return [first(item.why) || qaHit || first(item.def), "why"];
  if (qtype === "how") {
    const steps = (item.how || []).map((s) => String(s).trim()).filter(Boolean);
    return [steps.length ? steps.slice(0, 4).join(" → ") : (first(item.tips) || first(item.def)), "how"];
  }
  if (["when", "where", "who", "cost"].includes(qtype)) {
    return [item[qtype === "cost" ? "cost" : qtype] || qaHit || first(item.facts) || first(item.def), qtype];
  }
  if (qtype === "opinion") return [item.opinion || first(item.tips) || first(item.def), "opinion"];
  if (qtype === "count") return [first(item.facts) || first(item.def), "fact"];
  if (qaHit) return [qaHit, "qa"];
  return [first(item.def) || first(item.facts) || item.opinion || "", "def"];
}

const CONNECT = {
  add: ["", "", "また、", "加えて、", "それと、"],
  reason: ["", "理由としては、", "これは、", "背景には、"],
  answer: ["", "答えは、", "まず、"],
  advice: ["", "コツは、", "気をつけたいのは、"],
};
const rot = (kind, n) => {
  const table = CONNECT[kind] || [""];
  return table[Math.abs(n) % table.length];
};
const done = (s) => (/[。！？!?]$/.test(s) ? s : s + "。");

// 会話に混ぜてよい 1 文か（辞書の語彙情報・壊れた文は使わない）
const LEX_TRIVIA = /拍|索引|品詞|読みは|U\+|文字数/;
function usable(line) {
  const s = String(line || "").trim();
  return s.length >= 10 && s.length <= 140 && !LEX_TRIVIA.test(s);
}

export function styleTail(text, tone) {
  // 口調指定（〜だよ、〜だね 等）を文末に反映する。です・ますの混在は作らない。
  let t = String(text || "");
  if (!tone) return t;
  if (/だよ|だよね|だね|だろ/.test(tone)) {
    t = t.replace(/です。/g, "だ。").replace(/ます。/g, "る。").replace(/ですね/g, "だね");
    if (/だよ/.test(tone) && /\.$|。$/.test(t.slice(-1)) && !/だ。$/.test(t.slice(-2))) {
      t = t.replace(/。$/, "だよ。");
    }
  } else if (/です.*ます|ます.*です/.test(tone)) {
    if (!/です|ます/.test(t)) t = t.replace(/。$/, "です。");
  }
  return t;
}

function fitTo(text, target) {
  if (!target) return text;
  const lines = String(text).split("\n").filter((x) => x.trim());
  if (lines.join("").length <= target * 1.25) return text;
  const kept = [];
  let used = 0;
  for (const line of lines) {
    if (used + line.length > target * 1.15 && kept.length >= 2) continue;
    kept.push(line);
    used += line.length;
  }
  return kept.join("\n");
}

// --------------------------------------------------------------------------- //
// 応答の合成
// --------------------------------------------------------------------------- //
const REACT = {
  plan: ["{n}の予定ですね。前に進むための準備を一つ先に決めると、当日が楽になります。",
         "{n}の計画、いいですね。いつ動くかも決めておきますか。",
         "{n}なら、順番だけ先に決めると続きやすいです。"],
  report: ["{n}を終えられたんですね。次に同じことがあれば変えたい所はどこですか。",
           "{n}、片付いたんですね。いちばん効いた部分是でしたか。",
           "{n}が終わったなら、次は 1 手で済む形に直す番です。"],
  trouble: ["{n}が続くと堪えますね。今日できることを一つに絞りますか。",
            "{n}のはつらいですね。いちばん効いているのはどこですか。",
            "それは重たい{n}ですね。手を付けるなら今日どこまでですか。"],
  feel: ["{n}とのこと。きっかけを一言もらえますか。", "{n}のはいいですね。続いていますか。",
         "{n}を良いと思えるのが一番です。"],
  ask: ["{n}の話ですね。", "{n}、そこから数えます。", "{n}の件、引き受けます。"],
  declare: ["{n}の話をもらえました。", "{n}の件、受け取りました。"],
};
const SLOTS = [
  [/(今日|明日|昨日|来週|今週|週末|夜|朝|昼)/, "何時頃から始めますか。", "時間帯はもう決めてありますか。"],
  [/(うち|家|会社| office|公園|駅|大阪|東京|京都|どこ)/, "場所はもう決めていますか。", "どこでやりますか。"],
  [/(したい|行こう|やる|作る|買いたい|食べたい)/, "いちばん先に片付けたいのはどの部分ですか。",
   "やらないことも決めておきますか。"],
];

export function answer(text, { index, turn = 1, history = [], style = {}, kb } = {}) {
  const idx = index || (kb ? new Index(kb) : null);
  const raw = String(text || "").trim();
  if (!raw) return { text: "", plan: "empty", confidence: 0 };
  const qtype = questionType(raw);
  const polite = !/だよ|だね|だよね|じゃん|やん/.test(JSON.stringify(style) + raw);
  const target = Number(style.targetChars || 0);
  let hits = [];
  if (idx) hits = idx.search(raw, 5);

  const claims = [];
  let topic = "";
  const best = hits.find((h) => h.score >= 0.42 && (h.topicHit || h.covered >= 1));
  // 質問の形でない発話（報告・こぼれ言）は、*受け取りを先* にします（辞典を返さない）
  const asking = /[?？]\s*$/.test(raw) || /(ですか|ますか|でしょうか|とは|って何|どう|なぜ|いつ|どこ|いくら|何|which|what|how|why)/i.test(raw);
  const nounsEarly = tokens(raw).filter((w) => /[\u4e00-\u9fff\u30a1-\u30fa]/.test(w) && w.length >= 2);
  const leadEarly = nounsEarly.find((w) => !/^(今日|明日|昨日|一緒)$/.test(w)) || "";
  if (best && !asking && leadEarly) {
    const act = /たい$|よう$|予定|行こ|しよ/.test(normalize(raw)) ? "plan"
      : /た$|終わ|できた/.test(normalize(raw)) ? "report"
      : /痛|辛|困|嫌|最悪|疲/.test(raw) ? "trouble" : "declare";
    const pool = REACT[act] || REACT.declare;
    claims.push([done(pool[Math.abs(turn) % pool.length].replace(/\{n\}/g, leadEarly)), "answer"]);
    const [body] = pickField(best.item, "general", raw);
    if (usable(body)) claims.push([done(String(body)), "add"]);
    topic = best.item.topic;
  } else if (best) {
    topic = best.item.topic;
    const [body, field] = pickField(best.item, qtype, raw);
    if (usable(body)) claims.push([done(String(body)), field]);
    // 理由・補足を 1 文（短い答えのときだけ足す）
    if (String(body || "").length < 46) {
      for (const pool of [best.item.why, best.item.facts, best.item.tips]) {
        const extra = (pool || []).map((x) => String(x).trim()).filter((x) => usable(x) && x !== body)[0];
        if (extra) { claims.push([done(extra), "add"]); break; }
      }
    }
    if (field === "how") {
      const steps = (best.item.how || []).map((x) => String(x).trim()).filter(Boolean);
      if (steps.length > 1) {
        claims.length = 0;
        claims.push([steps.slice(0, 4).map((s, i) => `${i + 1}. ${s}`).join("\n"), "how"]);
      }
    }
    const fu = (best.item.followups || []).filter(Boolean);
    if (fu.length && qtype !== "definition") claims.push([fu[(turn + (topic.length || 0)) % fu.length], "ask"]);
  }

  // 話題が引けないとき: 内容語で反応し、抜けている枠を 1 つだけ聞く（辞書引きはしない）
  const nouns = tokens(raw).filter((w) => /[\u4e00-\u9fff\u30a1-\u30fa]/.test(w) && w.length >= 2);
  const lead = nouns.find((w) => !/^(今日|明日|昨日|一緒)$/.test(w)) || topic || "";
  if (!claims.length && lead) {
    const act = /たい$|よう$|予定|行こ|しよ/.test(normalize(raw)) ? "plan"
      : /た$|終わ|できた/.test(normalize(raw)) ? "report"
      : /痛|辛|困|嫌|最悪|疲/.test(raw) ? "trouble"
      : /嬉|うれ|いいね|楽し|好き/.test(raw) ? "feel"
      : /[?？]$/
      ? "ask" : (/[。.]$/.test(raw) || raw.length > 8 ? "declare" : "declare");
    const pool = REACT[act] || REACT.declare;
    claims.push([done(pool[Math.abs(turn) % pool.length].replace(/\{n\}/g, lead)), "answer"]);
  }
  if (!claims.length) {
    claims.push(["その語については、いま読めている形から組み立てます。", "answer"]);
  }
  if (!best && lead) {
    const slot = SLOTS.find(([re]) => re.test(normalize(raw)));
    const ask = slot ? slot[Math.abs(turn) % 2 + 1] : "何について話していますか、語を一言もらえますか。";
    if (!claims.some((c) => /[？?]$/.test(c[0]))) claims.push([ask, "ask"]);
  }

  let body = claims.map(([line, kind], i) => (i === 0 ? line : rot(kind === "add" ? "add" : kind, turn + i) + line))
    .join("\n");
  body = styleTail(body, style.tone || (polite ? "" : raw));
  if (target) body = fitTo(body, target);
  const confidence = best ? Math.min(0.95, 0.5 + 0.3 * best.score) : 0.4;
  return {
    text: body.trim(),
    plan: best ? `knowledge:${best.item && pickField(best.item, qtype, raw)[1]}` : "statement",
    confidence: Number(confidence.toFixed(3)),
    knowledge: { topic: topic || null, coverage: best ? Number(best.coverage.toFixed(3)) : 0,
                 via: best ? "kb" : "local", qtype },
    sources: [],
  };
}

// --------------------------------------------------------------------------- //
// 指示の-lite 実装（要約・抽出）。Pages 版はここだけ簡易です。
// --------------------------------------------------------------------------- //
export function summarize(text, n = 3) {
  const body = NFKC(String(text || "")).replace(/\s+/g, " ").trim();
  const sents = body.split(/(?<=[。！？!?])/).map((s) => s.trim()).filter((s) => s.length >= 12);
  const picked = [];
  for (const s of sents) {
    if (picked.length >= n) break;
    if (!picked.some((p) => dice(p, s) > 0.6)) picked.push(s);
  }
  return (picked.length ? picked : sents.slice(0, n)).map((s) => `・${s.replace(/[。！？!?]+$/, "")}`).join("\n");
}

export function extract(text, fields = []) {
  const src = NFKC(String(text || "")).replace(/\s+/g, " ").trim();
  const out = {};
  for (const key of fields) {
    let val = pickValue(src, key);
    if (val === null) val = "";
    out[key] = String(val).replace(/^[「『"']+|[」』"'。]+$/g, "").trim();
  }
  return JSON.stringify(out, null, 2);
}

// 材料の中で *語が立つ位置* を見て取ります（キー名のラベル表記にも対応）。
const READERS = {
  // 助詞（から・へ・まで…）を語の中に含めないので、「東京から京都まで」が 2 つに正しく割れます
  origin: [/([^、。．\sからへにでとをがの]{1,12}?)から/, /(?:from|origin|出発地|出発)\D{0,4}([^、。\s]{1,12})/i],
  destination: [/([^、。．\sからへにでとをが]{1,12}?)(?:へ|まで|着)/,
                /(?:to|destination|到着地|行先)\D{0,4}([^、。\s]{1,12})/i],
  duration: [/((?:約|およそ)?\s*\d+\s*時間(?:\s*\d+\s*分)?)/, /((?:約|およそ)?\s*\d+\s*分)/,
             /(\d+\s*(?:h|hours?)\b)/i],
  fare: [/((?:約|およそ)?\s*[\d,，]+\s*(?:円|yen|JPY))/i, /(?:料金|運賃|費用)\D{0,3}([\d,，]+\s*円)/],
  date: /((?:令和|平成|\d{4})\s*年\s*\d{1,2}\s*月(?:\s*\d{1,2}\s*日)?)/,
  name: /([\u4e00-\u9ffa\u3041-\u30faA-Za-z]{1,10}?)(?:さん|氏|行|様|です|さんは)/,
};

function pickValue(src, key) {
  const k = String(key || "").trim();
  const pats = READERS[k] || [];
  for (const re of pats) {
    const m = src.match(re);
    if (m) {
      const v = String(m[1] != null ? m[1] : m[0]).trim();
      if (v && v.length <= 40) return v;
    }
  }
  // スキーマのキー名が本文にそのまま出ている形（key: value）にも対応
  const labeled = src.match(new RegExp(`${k.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}\\s*[:：]?\\s*([^\\n、,。;；]{1,40})`, "i"));
  if (labeled) return labeled[1].trim();
  return null;
}

export function respond(messages = [], { index, style = {}, turn = 0 } = {}) {
  const last = [...messages].reverse().find((m) => m && m.role === "user");
  const text = String((last && last.content) || "");
  const prompt = String((messages[messages.length - 1] || {}).content || text);
  if (/要約|まとめて|箇条書き/.test(prompt)) {
    const n = Number((prompt.match(/(\d+)\s*つ|\b(\d+)\s*(?:items|bullets)/) || [])[1] || 3);
    const body = prompt.replace(/^[\s\S]*?(文章|テキスト|本文)\s*[:：]?\s*/, "");
    return { text: summarize(body, n), plan: "instruction:summarize", confidence: 0.9, sources: [] };
  }
  if (/抽出|extract/i.test(prompt) && /[{\[]/.test(prompt)) {
    const fields = [...String(prompt).matchAll(/"([A-Za-z_][A-Za-z0-9_]*)"\s*:/g)].map((m) => m[1]);
    // *材料だけ* を取り出す（指示文を誤って材料として読むのが v3 の事故）
    const m = String(prompt).match(/(?:テキスト|文章|本文|text|body)\s*[:：]\s*([\s\S]+)$/i);
    const body = m ? m[1] : String(prompt);
    return { text: extract(body, fields.length ? fields : ["origin", "destination", "duration", "fare"]),
             plan: "instruction:extract", confidence: 0.92, sources: [] };
  }
  const out = answer(text, { index, turn: turn || messages.length, style });
  return { ...out, engine: "Snipher pages (lite)" };
}

export function status(index) {
  return {
    engine: "snipher-pages-lite",
    topics: index ? index.meta.topics : 0,
    facts: index ? index.meta.facts : 0,
    qa: index ? index.meta.qa : 0,
    ready: !!index,
  };
}

export async function loadIndex(base = "") {
  const res = await fetch(`${base}/kb.json`, { cache: "no-store" });
  if (!res.ok) throw new Error(`kb.json が読めません (${res.status})`);
  return new Index(await res.json());
}
