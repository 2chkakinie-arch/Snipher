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
  if (!claims.length && lead && !best && /[？?]|教えて|知りたい|ほしい|できますか|どうやって/.test(raw)) {
    // 知らない語を *聞かれた* ときは、相槌ではなく問いの形から方針を返します。
    claims.push([shapeLine(normalize(raw)), "answer"]);
  } else if (!claims.length && lead) {
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
    // 語が引けないときも *問いの形* から当面の方針を立てる（辞書引きも「無い」も言わない）
    const shape = shapeLine(normalize(raw));
    claims.push([shape, "answer"]);
    if (lead && !/^(何|どれ|いつ)/.test(normalize(raw))) {
      claims.push([`「${lead}」について、意味・使い方・数量のどれを欲しがっていますか。`, "ask"]);
    }
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

const SHAPES = [
  [/(手順|申請|手続き|やり方|使い方|方法|how\s*to)/i,
   "手続きを尋ねる形として組みます。決めるのは、いつまでに・誰に出すか・どの形で残すか、の三つです。"],
  [/(値段|いくらか|費用|コスト|料金)/,
   "費用の話として組みます。材料代と手間時間のどちらを先に押さえるかで答えの形が変わります。"],
  [/(違い|比較|vs|どっち)/i,
   "比較の形にします。速さ・手間・あとから拡張できるかの三本で揃えて比べるのが安全です。"],
  [/(おすすめ|選び方|向いて|どれがいい|価値|コスパ)/,
   "向きの話として組みます。扱う量と、壊れたときに立て直す時間を基準にすると迷いません。"],
  [/(エラー|動かない|失敗|直したい|bug)/i,
   "詰まっている話として受け取ります。最後に変わった一点を切り分けると早く減ります。"],
  [/(いくつ|何個|数量|どれくらい)/,
   "量の話を聞いている形にします。単位と、数える対象が決まれば答えられます。"],
];

export function shapeLine(text) {
  for (const [re, msg] of SHAPES) if (re.test(text)) return msg;
  return "どんな場面で使う語かを一言もらえれば、その場で同じ形に組みます。";
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
    steering: true,
    deliberative: true,
    infinite_knowledge: true,
    features: ["fast-path", "logit-steering", "parallel-deliberation", "realtime-web-wave"],
  };
}

// --------------------------------------------------------------------------- //
// v6: リアルタイム波（ステアリング / 並列熟考 / リアルタイムWeb）— Pages 版
// Python 版 snipher/lfm/steering.py・mind/deliberate.py・ground/realtime.py と同じ契約。
// --------------------------------------------------------------------------- //

// 挨拶・礼・謝罪・短い相槌は検索しない（高速経路を維持）。それ以外はすべて検索。
const SIMPLE_RE = /^(こんにちは|こんばんは|おはよう|はじめまして|やあ|もしもし|ありがとう|ありがと|感謝|すみません|ごめん|おつかれ|お疲れ|さようなら|さよなら|またね|バイバイ|おやすみ|はい|いいえ|うん|ええ|そう|なるほど|わかった|了解|おけ|ok|草|w+|笑)+[。！？!?.]*$/i;

export function shouldSearchRealtime(text) {
  const t = normalize(String(text || ""));
  if (!t || t.length < 2) return false;
  if (SIMPLE_RE.test(t)) return false;
  if (/^[ぁ-んァ-ヶー]{1,2}[？?。]*$/.test(t)) return false;
  return true;
}

// 並列熟考（ミニ版）: 問いの構造を分解し、結論を確率波の材料にする
export function deliberateMini(text, index) {
  const q = String(text || "").trim();
  const steps = [`問いを分解: ${q.slice(0, 60)}`];
  let conclusion = "相手の語を尊重し、具体例と次の1手を添える";
  let confidence = 0.62;
  const qt = questionType(q);
  if (qt === "definition") { steps.push("定義要求 → 周辺知識から多角的に整理"); conclusion = "周辺語から推論し、断定は避けて多面的に定義する"; confidence = 0.7; }
  else if (qt === "why") { steps.push("理由の問い → 因果を分解して順序を決める"); conclusion = "原因→過程→結果の順で、検証可能な事実から積み上げる"; confidence = 0.7; }
  else if (qt === "how") { steps.push("手順の問い → 最小の動作例から組み立てる"); conclusion = "最小の動作例から始め、端数ケースを追加する"; confidence = 0.72; }
  if (index) {
    const hits = index.search(q, 1);
    if (hits.length && hits[0].score >= 0.42) {
      steps.push(`知識ベースに関連記述あり: ${hits[0].item.topic}`);
      confidence += 0.08;
    } else steps.push("知識ベースに直接の記述なし、推論で補う");
  }
  steps.push("自己検証: 矛盾・飛躍がないか最終チェック");
  return { query: q.slice(0, 80), steps, conclusion, confidence: Math.min(0.95, confidence),
           keywords: tokens(q).slice(0, 8) };
}

// 生成を文単位で刻み、チャンク間にステアリング波を差し込む（出力は止めない）
export async function* respondStream(messages = [], {
  index, style = {}, turn = 0, pollSteer = null, search = null, chunkDelayMs = 14,
} = {}) {
  const last = [...messages].reverse().find((m) => m && m.role === "user");
  const text = String((last && last.content) || "");
  yield { type: "start", engine: "Snipher pages (lite · waves)" };

  // 1) 並列熟考（同期ミニ版 — 出力前に思考の骨格を作り、確率波の材料にする）
  const thought = deliberateMini(text, index);
  yield { type: "thought", state: "start" };
  yield { type: "thought", state: "done", ...thought };

  // 2) リアルタイムWeb（挨拶以外はすべて。出力中に検索し、証拠を確率波として干渉）
  let webSentences = 0;
  const webSources = [];
  if (search && shouldSearchRealtime(text)) {
    yield { type: "web", state: "start", query: text.slice(0, 80) };
    try {
      const res = await search(text);
      const sents = (res && res.sentences) || [];
      const srcs = (res && res.sources) || [];
      webSentences = sents.length;
      for (const s of srcs.slice(0, 4)) webSources.push({ url: s.url, title: s.title || "" });
      yield { type: "web", state: "done", sentences: webSentences, sources: webSources };
    } catch (e) {
      yield { type: "web", state: "done", sentences: 0, sources: [], error: String((e && e.message) || e) };
    }
  }

  // 3) 再帰的思考（モデルA: 思考 → モデルB: 下書き → モデルC: 校正）。
  //    校正で却下されたら再下書き。チャンク間でステアリング波を受け取り、
  //    波が来たら残りの文を *その場で組み直す*（生成は止めない）
  const rec = recurrentRespond(messages, { index, turn, style });
  yield { type: "draft", round: 1, text: rec.text };
  yield { type: "verify", accepted: rec.accepted, reasons: rec.reasons, rounds: rec.rounds };
  let out = { text: rec.text, plan: "recurrent", confidence: rec.confidence,
              engine: "Snipher pages (recurrent-v8)", sources: [] };
  let sents = String(out.text || "").split(/(?<=[。！？!?\n])/).filter(Boolean);
  let emitted = "";
  let steerWaves = 0;
  for (let i = 0; i < sents.length; i++) {
    if (pollSteer) {
      const waves = pollSteer() || [];
      for (const w of waves) {
        const steerText = String((w && w.text) || "").trim();
        if (!steerText) continue;
        steerWaves++;
        yield { type: "steer", text: steerText.slice(0, 80), signals: 1 };
        // 確率波として残りの生成に干渉: ステアの語を材料に残りを組み直す
        const merged = [...messages, { role: "user", content: steerText }];
        const rerolled = respond(merged, { index, turn, style });
        const rest = String(rerolled.text || "");
        if (rest && rest !== emitted) sents = [rest];
      }
    }
    const piece = sents[i] || "";
    emitted += piece;
    if (piece) yield { type: "delta", text: piece };
    if (chunkDelayMs > 0) await new Promise((r) => setTimeout(r, chunkDelayMs));
  }

  yield {
    type: "done", text: emitted || out.text || "",
    stats: {
      route: "pages", engine: out.engine || "snipher-pages-lite", plan: out.plan,
      confidence: out.confidence, knowledge: {},
      sources: webSources.slice(0, 8),
      waves: { deliberate: true, web_search: !!(search && shouldSearchRealtime(text)),
               web_hits: webSentences, steer_waves: steerWaves, thought: true,
               recurrent: true, rounds: rec.rounds, accepted: rec.accepted },
    },
  };
}

export async function loadIndex(base = "") {
  const res = await fetch(`${base}/kb.json`, { cache: "no-store" });
  if (!res.ok) throw new Error(`kb.json が読めません (${res.status})`);
  return new Index(await res.json());
}

// --------------------------------------------------------------------------- //
// v8: 再帰的思考（Draft-Verification）— Pages 版
//   モデルA（思考）→ モデルB（文章化）→ モデルC（校正・却下なら再下書き）
//   反復は n-gram で物理的に封印。generateNovel はテンプレート無しの文字レベル生成。
//   Python 版 snipher/mind/recurrent.py と同じ契約（thought / draft / verify / delta）。
// --------------------------------------------------------------------------- //

const DANGLE = /(を|が|は|に|で|と|も|へ|の|や|から|まで)$/;
const TEMPLATE_RE = /(の知識で答えます|としてお答えします|以下の通りです|ご質問ありがとうございます|お役に立てれば幸いです)/;

// モデルA: 意図を分析し、結論の箇条書きを作る
export function thinkV8(text, index) {
  const q = normalize(String(text || ""));
  const bullets = [];
  const qt = questionType(q);
  if (qt === "definition") bullets.push("問いは定義。周辺語から多角的に整理する");
  else if (qt === "why") bullets.push("問いは理由。原因→過程→結果の順に積む");
  else if (qt === "how") bullets.push("問いは手順。最小の動作例から組み立てる");
  else if (qt === "opinion") bullets.push("問いは感想。理由と具体例を添える");
  else bullets.push("平叙。相手の語を尊重し、次の1手を添える");
  let confidence = 0.6;
  if (index) {
    const hits = index.search(q, 3).filter((h) => h.score >= 0.42);
    for (const h of hits.slice(0, 3)) {
      bullets.push(`関連: ${h.item.topic}（一致度 ${h.score.toFixed(2)}）`);
    }
    if (hits.length) confidence = Math.min(0.9, 0.6 + 0.1 * hits.length);
    else bullets.push("直接の記述なし。推論で補い、断定を避ける");
  }
  bullets.push("自己検証: 反復・飛躍・文体混在がないか最終確認");
  return { bullets, conclusion: bullets[0] || "", confidence };
}

// モデルB: 箇条書きをもとに本文を組み立てる（Pages 版は証拠ベースの合成）
export function draftV8(messages, { index, turn = 1, style = {} } = {}) {
  return answer(String(([...messages].reverse().find((m) => m && m.role === "user") || {}).content || ""),
    { index, turn, style });
}

// モデルC: 校正・フォーマット。不要な反復・壊れた文末・文体混在・定型表現を検出する
export function verifyV8(draft) {
  const text = String(draft || "");
  const reasons = [];
  // 1) 行・文の重複（同一文の 2 回以上）
  const lines = text.split(/\n+/).map((s) => s.trim()).filter(Boolean);
  const uniqLines = new Set(lines);
  if (uniqLines.size !== lines.length) reasons.push("反復: 同一の文が複数回");
  // 2) 5-gram の周回
  const flat = text.replace(/[^0-9a-z\u3041-\u30fa\u4e00-\u9fff]/g, "");
  const seen = new Set();
  for (let i = 0; i + 5 <= flat.length; i++) {
    const g = flat.slice(i, i + 5);
    if (seen.has(g)) { reasons.push("反復: 5-gram の周回"); break; }
    seen.add(g);
  }
  // 3) 文体混在
  if (/です|ます/.test(text) && /(だ。|た。)/.test(text)) reasons.push("文体: です・ます と だ・た が混在");
  // 4) 助詞で投げっぱなしの文末
  for (const line of lines) if (DANGLE.test(line) && line.length > 1) { reasons.push(`文末: ${line.slice(-2)}`); break; }
  // 5) 定型表現
  if (TEMPLATE_RE.test(text)) reasons.push("定型表現: " + (text.match(TEMPLATE_RE) || [])[0]);
  // 修正: 重複行を除く（校正の実際の仕事）
  let fixed = [...uniqLines].join("\n");
  return { accepted: reasons.length === 0, reasons, fixed };
}

// 再帰ループ: 思考 → 下書き → 校正（→ 却下なら再下書き）
export function recurrentRespond(messages = [], { index, style = {}, turn = 0, maxRounds = 3 } = {}) {
  const last = [...messages].reverse().find((m) => m && m.role === "user");
  const text = String((last && last.content) || "");
  const thought = thinkV8(text, index);
  let draft = draftV8(messages, { index, turn, style });
  let verdict = verifyV8(draft.text);
  let rounds = 1;
  while (!verdict.accepted && rounds < maxRounds) {
    // 却下 → 温度を下げ、重複行を除いた「波」を乗せて再下書き
    const merged = [...messages, { role: "user", content: verdict.fixed || text }];
    draft = draftV8(merged, { index, turn: turn + rounds, style });
    verdict = verifyV8(draft.text);
    rounds++;
  }
  return {
    text: verdict.accepted ? draft.text : (verdict.fixed || draft.text),
    thought: thought.bullets.join(" / "),
    rounds,
    accepted: verdict.accepted,
    reasons: verdict.reasons,
    confidence: Number((thought.confidence || 0.6).toFixed(3)),
  };
}

// テンプレート無しの長文生成（文字レベル n-gram、パラメータは KB 本文の頻度だけ）
export function generateNovel(prompt = "", { index, maxChars = 2000, seed = 0 } = {}) {
  const corpus = [];
  const push = (s) => { const t = String(s || "").trim(); if (t.length >= 2) corpus.push(t); };
  if (index && index.items) {
    for (const it of index.items) {
      push(it.def); push(it.opinion);
      for (const f of it.facts || []) push(f);
      for (const f of it.why || []) push(f);
      for (const f of it.how || []) push(f);
      for (const f of it.tips || []) push(f);
      for (const [q, a] of it.qa || []) { push(q); push(a); }
    }
  }
  if (!corpus.length) corpus.push("ある小さな町のはずれに、古い時計台がありました。");
  // 3-gram 頻度表（context: 直前2文字 → 次文字の重み）
  const model = new Map();
  const starts = [];
  for (const s of corpus) {
    starts.push(s[0]);
    for (let i = 0; i + 2 < s.length; i++) {
      const ctx = s.slice(i, i + 2);
      const nxt = s[i + 2];
      if (!model.has(ctx)) model.set(ctx, new Map());
      const m = model.get(ctx);
      m.set(nxt, (m.get(nxt) || 0) + 1);
    }
  }
  let state = (Number(seed) >>> 0) || 1;
  const rnd = () => { state = (state * 1664525 + 1013904223) >>> 0; return state / 4294967296; };
  const pick = (m) => {
    let total = 0;
    for (const c of m.values()) total += c;
    let r = rnd() * total;
    for (const [ch, c] of m) { r -= c; if (r <= 0) return ch; }
    return [...m.keys()][0];
  };
  const ALPHABET = "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん、。";
  let out = prompt ? String(prompt) : "";
  if (!out) out = starts[Math.floor(rnd() * starts.length)] || "あ";
  // 既出 5-gram の集合（反復の物理的封印。この集合に無い文字しか追加しない）
  const gram5 = new Set();
  for (let i = 0; i + 5 <= out.length; i++) gram5.add(out.slice(i, i + 5));
  while (out.length < maxChars) {
    const ctx = out.slice(-2);
    let m = model.get(ctx);
    if (!m || !m.size) m = model.get(out.slice(-1));
    if (!m || !m.size) m = new Map([...ALPHABET].map((ch) => [ch, 1]));
    const tail4 = out.slice(-4);
    let cand = new Map();
    for (const [ch, c] of m) {
      if (tail4.length === 4 && gram5.has(tail4 + ch)) continue;  // 反復 → 禁止
      cand.set(ch, c);
    }
    if (!cand.size) {
      // 全候補が反復に当たる場合は、文字表から反復しない文字を選ぶ
      for (const ch of ALPHABET) {
        if (tail4.length === 4 && gram5.has(tail4 + ch)) continue;
        cand.set(ch, 1);
      }
    }
    if (!cand.size) break;
    out += pick(cand);
    if (out.length >= 5) gram5.add(out.slice(-5));
  }
  return out.slice(0, maxChars).trim();
}
