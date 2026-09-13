# 指示層（`snipher/instruction/`）— プロンプトを *仕事として* 実行する

「次のテキストから情報を抽出し、JSON 形式のみで出力してください」に対して
v3 までの Snipher は「テキストを読みました」という **語彙の反応** を返していました。
原因はモデルの大きさではなく、**指示文と材料を分けて読んでいなかった**ことです。
指示層はその 1 点を構造で直します。

```
1 通 ──► parse()   指示部 / 材料部 / 出力仕様（JSON スキーマ・件数・文字数・口調・役割）
     ──► execute() タスクを実行する（抽出・要約・コード・応答・変換・列挙・作文）
     ──► verify()  出力を仕様に照らす（JSON はパース、箇条書きは本数、コードは実行）
     ──► 外れていれば組み直す ──► Result（text / checks / confidence / meta）
```

指示でなければ `parse()` が `None` を返し、いつもの会話経路（`mind/think.py`）が
そのまま引き継ぎます。**閾値は 0.55**、誤検出は `tools/bench_instruction.py` が
20 通の会話で 0 であることを測ります。

---

## 1. 読み取り（`parser.py`）

判定は語の意味引きではなく *形* です。得点を足して閾値を超えたときだけ指示とみなします。

| 信号 | 例 | 得点 |
|---|---|---|
| 材料が差し出されている | `テキスト: 「…」` / ``` フェンス / `文章：` | 0.30（8 字以上）/ 0.22（ラベルあり） |
| 出力スキーマ | `{"origin": "出発地", …}` | 0.35 |
| 形式の指定 | `JSON 形式のみ` `CSV 形式` `表形式` `key: value 形式` | 0.18（+ strict 0.12） |
| 出力ルール（引用された語） | `必ず「α-β枝刈り」を含めて` `「です・ます」は使わない` | 0.16 |
| 命令の述語 | `〜してください` `〜せよ` | 0.12〜0.18 |
| 仕事を名指しする動詞 | `要約` `抽出` `作成` `列挙` `変換` | 0.14 |
| 件数 / 文字数 / 口調 / 役割 | `3つの箇条書き` `200文字程度` `〜だよ` `あなたは〜です` | 0.12 / 0.12 / 0.10 / 0.12 |
| コード（言語名 + 関数名） | `JavaScriptで uniqueSort(arr)` | 0.30 + 0.12 + 0.14 |

読み取る仕様は `FormatSpec` に入ります:
`kind`（json/csv/table/keyvalue）`strict` `only_output` `no_greeting` `no_explanation`
`schema_fields` `bullets` `numbered` `bullet_char` `lines` `max_chars` `target_chars`
`length_kind`（brief/normal）`tone`（friendly/polite/plain…）`register` `language`
`extra_rules`（`必ず「…」を含めて` `「…」は使わない` の硬い制約＝`Directive.rules`）。

欄名（スキーマ）は *材料と指示の両方* から読みます。優先順は
(1) JSON テンプレート、(2) `CSVフォーマット:` のヘッダ行、(3) `項目: 名前と値段` の欄名列、
(4) 材料のラベル（`氏名: 山田太郎`）です。`100文字以内 / 以下 / まで / を超えない` は
**上限**（`max_chars`）、`200文字程度` は目標（`target_chars`）として読み分けます。
`「です・ます」は使わない` のような *打ち消し* は敬体の指定とは読みません（常体として扱う）。

`Directive` は `task`（extract/summarize/code/answer/list/transform/write）と
`instruction`（指示部）`payload`（材料部）`question`（問い）`role`（役割）を持ちます。
**指示部と材料部が分離されている**のが要点で、材料の中の「テキスト」「文章」という語に
反応して読み上げることが構造上できません。

## 2. 実行

| モジュール | 仕事 | ねつ造を防ぐ仕掛け |
|---|---|---|
| `extract.py` | 材料から値を抜き、スキーマの欄に埋める。`label_values`（`氏名: 山田太郎`）と `extract_records`（`りんごは1個120円。みかんは…` → 2 行）で **材料の形に合わせて行を増やす** | 値は **材料の文字列そのもの**。距離・所要時間・料金は正規表現のパターンで拾い、欄に足りなければ `missing` として残す（埋めない）。欄名が無いときは `field1` をでっち上げず、材料のラベルを欄名にする |
| `summarize.py` | 文を単位に割り、内容語の重みで上位を points にする | `coverage`（材料の語をどれだけ使ったか）を返し、**材料に無い語で文を作らない**。点が足りなければ足りないまま返す |
| `code.py` | 操作の動詞から関数を組み、**実際に実行**する | `node` / `python3` / `go run` で走らせて結果を `notes` に残す。実行できない言語は構文の自己点検まで、と明記する |
| `answer.py` | 役割・口調・文字数を守って答える | 材料は 知識ベース → 実辞書 → ウェブ裏取り → **指示文が差し出した資料** → 数えられる事実 の順。推測で語義を作らない |
| `translate.py` | 英訳・和訳。文を節に割り、役割（は/が/を/に/で/と）を英語の語順に並べ替える | 対応表に無い語は語彙バンクの **読み（ローマ字）** で残し `notes` に書く。材料に無い事実を足さない |
| `style.py` | 口調（〜だよ/です・ます/常体）と長さの組み替え | 文末の語を実辞書で引いて辞書形に戻し、活用表で組み直す。語尾の文字差し替えはしない |

コードの実行例（`uniqueSort(arr)`）:

```javascript
function uniqueSort(arr) {
  const out = Array.from(new Set(arr));
  out.sort((a, b) => (a < b ? -1 : a > b ? 1 : 0));
  return out;
}
```

`node` で `uniqueSort([3,1,3,2]) === [1,2,3]` と `uniqueSort(["b","a","b"]) === ["a","b"]` を
実際に走らせてから返します（`ran=True, ok=True`、`notes` に「node で実行: 2/2 件が期待通り」）。
TypeScript の型注釈（`nums: number[]`）は `strip_ts_types()` で外してから実行します。

## 3. 検証（`run.py::verify`）

出力は返す前に機械的に検査され、記録が `Result.checks` に残ります。

| 検査 | 見ているもの |
|---|---|
| `non_empty` / `no_refusal` | 空でないか / 「できません」系の逃げが無い（`CAN_NOT_SAY` 15 語） |
| `json_schema` / `values_filled` / `grounded_in_payload` | パースできるか、欄が埋まったか、**値が材料の文字列か** |
| `strict_json_only` / `strict_no_preamble` | `JSON のみ` 指定のとき前置き・解説が無いか |
| `bullet_count` / `bullet_is_sentence` | 箇条書きの本数、各行が述語で終わるか |
| `max_chars` / `target_chars` | 文字数（上限 / 目標の 0.55〜1.55 倍。空白と改行は数えない） |
| `tone_friendly` / `tone_polite` | 指定の語尾（だよ・だね / です・ます）が入っているか |
| `code_fence` / `code_executed` / `function_name` | コードブロック、**実行が通ったか**、指定の関数名 |
| `coverage` | 要約が材料の語をどれだけ使ったか（0.6 未満は不合格） |
| `rule_require` / `rule_forbid` | 指示が `必ず含めて` と書いた語が入っているか、`使わない` と書いた語が無いか |
| `line_count` / `no_extra_prose` | 行数の指定、`解説は不要` のときコードブロックの外に文字が無いか |
| `well_formed` / `shape_ok` | 文章なら `composer.validate()`、データなら形の健全性 |

検査の前に `apply_rules()` が規則を出力に当てます。`必ず「α-β枝刈り」を含めて` に対して
語が無ければ最初の文の話題として入れ、その分だけ長くなるので **上限に収め直します**
（`「です・ます」は使わない` は `style.restyle(register="plain")` で常体に組み替える）。
落ちたら *組み直します*（要約は文字数予算を緩めて再構成、応答は材料を集め直す）。
それでも落ちたら `ok=False` のまま返し、`checks` に理由を残します。
**「できた」と嘘をつかない**ための仕組みです。

## 4. 経路（どこから入っても同じ結果）

| 入口 | 場所 | 挙動 |
|---|---|---|
| HTTP / SSE | `core.py::stream_reply` → `_instruction_reply` | 下書きより前。`assist.mode="instruction"`、`route="instant"`（ニューラルを 1 トークンも使わない） |
| 思考ループ | `mind/think.py::_run_instruction` | §0（挨拶）より前。`Rendered(authoritative=True, plan="instruction:<task>")` |
| 同期応答 | `composer.py::compose` | think 経由で同じ `Result` を `Reply` にする |
| タスク振り分け | `tasks.py::TaskRouter.classify` | `"instruction"`（コードの依頼は従来どおり `"code"`） |
| 直接 | `from snipher.instruction import run` | `run(text)` → `Result` / `None` |

指示経路は `_light_reply`（1180〜1220 行あたりで厳密出力を文章に整え直してしまう場所）を
**通りません**。JSON が「JSON です。」に化けないのはこのためです。

## 5. 実測

```bash
.venv/bin/python -m pytest -q tests/test_instruction.py     # 43 passed
.venv/bin/python tools/bench_instruction.py                 # 判定 PASS
```

`tools/bench_instruction.py --runs 2`（この環境、`SNIPHER_WEB=off`）:

```
指示追従率 100.0%（extract / summarize / code / answer / list / transform すべて 1.0）
1 指示: 中央値 5.05ms / p95 35.1ms / 最大 547ms（40.2 指示/秒）
タスク別中央値: extract 0.95ms / list 1.37ms / transform 2.04ms / answer 10.23ms
              summarize 16.8ms / code 26.68ms（実行込み）
誤検出 0/20（読み取り 中央値 0.071ms） / 禁止表現 0
```

指示のバッテリーは 20 種（40 実行）。CSV ヘッダからの欄読み、材料 2 件 → 表 2 行、
`key: value`（ラベルを欄名に）、`必ず「…」を含めて` + `100文字以内`、`「です・ます」は使わない`、
英訳 / 和訳、`コードのみ（解説は不要）` を含みます。中身の期待（必ず入る語 / 入ってはいけない語）も
`expect.contains` / `expect.absent` で機械判定します。

合格線は「全指示が仕様どおり・誤検出ゼロ・逃げゼロ」で、1 つでも外すと exit code 1 です。

## 6. 制約（正直なところ）

- 材料に無い事実を作りません。要約の点が足りない、抽出の欄が埋まらない、
  というのは *出力にそのまま出る*（`missing` / `ok=False` / `checks`）仕様です。
- ウェブ検索が使えない設定では、未知の語の語義を推測で埋めません。
  代わりに「語彙バンク 15 万語の見出しに無い」「検索は使える設定なので裏取りに出られる」と
  **数えられる事実**を返します（`answer.py::fallback_claims`）。
- 実行系の検査（`code_executed`）は `node` / `python3` / `go` が無い環境では
  構文の自己点検までに落ちます。そのときは `checks` にそう書いてあります。
- 翻訳は「文型 + 語彙対応」の訳です。対応表（時間・場所・食べ物・移動・日常の動詞 約 250 語）に
  無い語は読み（ローマ字）で残し、`notes` に *どの語が残ったか* を書きます。
  専門分野の長文を文学的に訳すことはしません（材料の語を落とさないことを優先）。
- 指示に名前が無い関数（「配列の合計を返す関数」）は操作から名前を付けます
  （JS `sumValues` / Python `sum_values` / Go `SumValues`）。`notes` にそう書いた上で実行します。
