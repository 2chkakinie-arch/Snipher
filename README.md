# Snipher

**日本語を 0 から組み立てるチャット AI。LFM2.5-1.2B-JP より軽く・速く・賢く。**

Snipher v2 は「定型文の引き当て」をやめました。返事は毎回、次の 4 層が
その場で組み立てます。

| 層 | 何をするか | 規模 | 実測 |
|---|---|---|---|
| **知識ベース v2** | 発話から話題を引き、**問いの型**（定義/理由/手順/時期/場所/値段/感想/困りごと）に合う欄を選ぶ | 199 話題・611 事実・426 問答 | 1.2 ms/発話 |
| **composer** | 引いた材料を日本語の応答に設計する（相槌 → 本文 → 続きの問い）。材料が無ければ **無いと正直に言う** | 8 計画・回転する枠 | 0.8 ms/応答 |
| **巨大 n-gram LM** | 組み立てた文を採点し、壊れた日本語・ループ・文体の混在を捨てる | 276,832 エントリ（5-gram） | 0.7 ms/文 |
| **内蔵ニューラルコア** | 材料が無いときに本文の生成に挑戦し、文末の助動詞を補う。KV キャッシュ付き | 5,629,824 params（d=384 / L=10） | 2.5 ms/文字 |

同梱する重みの合計は **7.7 MB**。LFM2.5-1.2B-JP は 1.17B パラメータ・
GGUF Q4_K_M で 731 MB / safetensors で 2.2 GB のダウンロードが必要です。
Snipher は **ダウンロード 0・依存は numpy のみ**で、同じマシンで
1 ターン数十ミリ秒（検索 → 組立 → 判定）から応答します。

- **嘘をつかない**: 知識ベースに材料が無ければ「分かりません、もう少し言葉をください」と言う。
  数字だけ・文字化けだけの入力にも「読めなかった」と返す（確度 0% の定型文は出さない）
- **壊れない**: 出した文は必ず `validate()` と n-gram LM の二重検査を通す
  （助詞で切れた文・ですです・ループ・文体混在は捨てて組み直す）
- **飽きない**: 同じ入力が続いても枠を回転させ、直前の応答と違う言い回しを選ぶ
- **全自動**: 常駐環境では LFM2.5-1.2B-JP のフルウェイトを自分で取得して
  さらに上の品質に昇格。サーバーレス（Vercel）では同梱の 4 層だけでフル機能
- シンプルな**ホワイトテーマ**のチャット UI（SSE ストリーミング・tok/s 表示）

```bash
pip install -r requirements.txt
uvicorn snipher.api:app --host 0.0.0.0 --port 8000     # → http://localhost:8000
python tools/bench.py                                  # 上の表を実測で作り直す
```

---

## 内部構造

```
ユーザー発話
   │
   ▼ ── Snipher Core ────────────────────────────────────────────────┐
   │ ① 解析（1ms）                                                    │
   │    ├─ 意図 / 気分 / 問いの型 / 内容語 / 数字・ASCII の有無          │
   │    └─ 確率的な下書き（独自確率式が各スロットの softmax 確度を記録）    │
   │                                                                  │
   │ ② 知識ベース v2（1.2ms）                                          │
   │    文字バイグラム + 最長一致の辞書 → BM25 → 問いの型に合う欄を選ぶ     │
   │    定義 / 理由 / 手順 / 時期 / 場所 / 値段 / 感想 / よくある問い       │
   │    材料が無ければ None（＝知らない）を返す                           │
   │                                                                  │
   │ ③ composer（0.8ms）… 文の設計図                                    │
   │    材料あり → 相槌 + 本文 + 続きの問い                              │
   │    材料なし → 話題を受け取る / 読めなかったと言う / 自分について答える   │
   │    8 計画: kb_answer・unknown_topic・opaque_input・statement・      │
   │            self・greeting・social・safety                         │
   │                                                                  │
   │ ④ 巨大 n-gram LM（0.7ms/文）… 流暢さの審判                          │
   │    276,832 エントリの 5-gram が perplexity を測り、                 │
   │    壊れた文・ループ・文体混在を落として候補を並べ替える                 │
   │                                                                  │
   │ ⑤ 内蔵ニューラルコア（2.5ms/文字・KV キャッシュ）                    │
   │    材料が無いときだけ本文の生成に挑戦し、LM と両方が自信を持った文だけ採用 │
   │    文末の助動詞が欠けていれば補う（complete）                         │
   │                                                                  │
   │ ⑥ 昇格（任意）                                                     │
   │    T1 ローカル フルウェイト  llama.cpp(GGUF 731MB) / torch(INT8)    │
   │    T2 リモート委譲          SNIPHER_LFM_REMOTE_URL のホストに依頼     │
   │    … ⑤ の確信度が閾値を切ったときだけ、裏で起動準備（応答は止めない）    │
   │                                                                  │
   │    どの経路の出力も必ず polisher → validate() → LM で整形・検査        │
   └──────────────────────────────────────────────────────────────────┘
```

- 材料がある応答はニューラル生成を待たない → **数十ミリ秒で返る**
- 材料が無いときだけニューラルコアが生成し、**LM と話題の関連性の二重ゲート**を通す
- それも通らなければ「分かりません、教えてください」→ **それっぽい誤情報を出さない**

### composer（`snipher/composer.py`）

「定型文を埋める」のではなく、**発話の形に応じて文を設計する**部品です。

| 計画 | 使う場面 | 例（入力 → 出力） |
|---|---|---|
| `kb_answer` | 知識ベースに材料がある | 花火とは → 定義 + 事実 + 続きの問い |
| `unknown_topic` | 語は読めたが材料が無い | ぬるぬる猿 → 「確かな情報を持っていません。何を知りたいですか」 |
| `opaque_input` | 数字だけ・文字化け・1 文字 | 67 → 「数字だけのようです。年齢や数量、計算の途中でしょうか」 |
| `statement` | 相手の報告・感想 | 暇だなあ → 相槌 + 問い |
| `self` | Snipher 自身への質問 | あなたは誰？ → 自己紹介 |
| `greeting` / `social` | 挨拶・礼・謝罪・褒め・別れ | ありがとう → 応答（相槌を二重に付けない） |
| `safety` | どれにも当てはまらない | 最短の正直な応答 |

組み立てた文は必ず `validate()` を通します（長さ・終点・禁じパターン・
助詞のぶら下がり・2-gram のループ・です/ます体の一貫性）。通らなければ
別の枠で組み直し、それでも駄目なら本文だけに縮めます。

### 巨大 n-gram LM（`snipher/lm.py`）

| 項目 | 内容 |
|---|---|
| 構造 | 文字 1〜5 gram・線形補間（stupid backoff 系） |
| エントリ | **276,832**（＝このモデルのパラメータ数。小型ニューラルネットの数十倍） |
| 重み | uint64 キー + uint16 量子化カウント + zlib → **2.3 MB** |
| 語彙 | 1,706 文字 |
| 学習データ | 人が書いた日本語（`kb.json` 2,672 文 + `dialogues.json` 220 組）+ 文法生成 9 万文 |
| 判定力 | 正しい日本語 ppl **2.4** / 文法破壊 ppl 22 / 文字シャッフル ppl **3,400** |
| 速さ | 1 文の採点 **0.7 ms**（numpy の searchsorted を次数ぶん呼ぶだけ） |
| 依存 | numpy のみ・ダウンロード不要 |

確信度はビルド時に実データから校正します（自然な文の ppl を `lo`、
文字をシャッフルした文の ppl を `hi` として対数スケールで 0.05〜0.95 に写像）。
だから「確度 95%」が根拠のある数字になります。

```bash
python tools/build_lm.py --grammar 60000      # 約 17 秒で lm.npz を作り直す
```

### 内蔵ニューラルコア（`snipher/neural/`）

| 項目 | 内容 |
|---|---|
| 構造 | ShortConv → SelfAttn → GLU の **10 ブロック** Hybrid（LFM2.5 と同じ系譜）/ RMSNorm / SiLU ゲート / RoPE |
| パラメータ | **5,629,824**（d=384 / L=10 / 8 heads・int8 量子化 + 行別スケール → 重み 5.1 MB） |
| 語彙 | 文字レベル 1,150（未知文字が原理的に出ない） |
| 学習データ | 人が書いた対話 220 組 ×8 + 知識ベース 4,812 文書 ×3 + 文法生成 28,047 文書（外部データ 0） |
| 推論 | **KV キャッシュ**付き逐次デコード（1 文字 2.5 ms / 396 文字每秒・キャッシュ無し比 4.0x） |
| 依存 | numpy のみ（torch / transformers / llama.cpp 不要） |
| ロード | 数十ミリ秒・ダウンロード不要（Vercel でも即動く） |

```bash
python tools/distill_neural.py --profile huge   # 本番スナップショット（約 60 分）
python tools/distill_neural.py --profile tiny   # CI/スモーク用（約 20 秒）
```

各 epoch の終わりにスナップショットを書くので、途中で止めても重みは使えます。

### バックエンド（自動選択）

| バックエンド | ランタイム | モデル | 特徴 |
|---|---|---|---|
| `gguf`（既定・最速） | llama-cpp-python / llama-server | 公式 GGUF Q4_K_M (731MB) | CPU で最速・省メモリ(~1.5GB)。byte-fallback のため未知文字が発生しない |
| `torch` | transformers + 動的 INT8 | model.safetensors (2.2GB) | 未知文字の埋め込み学習（予約トークン + 勾配更新）が使える |

両方インストールすれば GGUF が優先されます。未知文字学習を使う場合のみ
`SNIPHER_LFM_BACKEND=torch` で切り替えてください。

---

## クイックスタート

```bash
python3 -m venv .venv

# A) 推奨: llama.cpp バックエンド（CPU 最速・731MB の自動取得）
.venv/bin/pip install -r requirements.txt llama-cpp-python

# B) または torch バックエンド（2.2GB の自動取得・未知文字学習対応）
.venv/bin/pip install -r requirements.txt -r requirements-llm.txt

.venv/bin/uvicorn snipher.api:app --host 0.0.0.0 --port 8000
```

`http://localhost:8000` を開くだけ。**初回起動時にモデルの取得が自動で始まり**、
進捗が UI のバッジとプログレスバーに表示されます（操作は一切不要）。
取得済みのモデルは `var/models/` にキャッシュされ、次回からは即起動します。

- チャット UI: `http://localhost:8000/`（ホワイトテーマ）
- 旧 UI（超小型エンジン単体のデモ）: `/classic`
- Swagger: `/docs`

### 生成パラメータ

LFM2.5 公式推奨を既定値にしています: `temperature 0.1` / `top_k 50` /
`repetition_penalty 1.05`。UI の「設定」から変更できます。

---

## モデルの自動取得（アップロード不要）

起動時に次のソースを自動で試行します（Range レジューム・サイズ検証・アトミック保存）:

1. ローカルキャッシュ（`var/models/`）/ `SNIPHER_LFM_MODEL` / `SNIPHER_LFM_GGUF`
2. `SNIPHER_LFM_URLS` に指定した直接 URL（任意数）
3. 共有ミラー（ギガワタス共有ページ → 直リンクを HTML から自動解決）
4. HuggingFace 公式
   - GGUF: [`LiquidAI/LFM2.5-1.2B-JP-202606-GGUF`](https://huggingface.co/LiquidAI/LFM2.5-1.2B-JP-202606-GGUF)
   - native: [`LiquidAI/LFM2.5-1.2B-JP-202606`](https://huggingface.co/LiquidAI/LFM2.5-1.2B-JP-202606)
5. hf-mirror.com

失敗しても数分ごとに自動再試行し、UI の「再試行」ボタンからもトリガーできます。
取得中は Snipher-mini+（高速コア）が応答を続けるので、会話が止まることはありません。

CLI で事前取得することもできます（同じロジック）:

```bash
python tools/fetch_model.py                    # GGUF Q4_K_M（既定）
python tools/fetch_model.py --backend torch    # transformers 用一式
python tools/fetch_model.py --quant Q8_0       # 他の量子化
```

### 完全にオフラインの環境（最後の手段）

ネットワークが完全に遮断された環境向けに、UI の「モデル取得」ドロワー内の
「手動取り込み」にファイルをドロップする方法も残してあります
（`config.json` / `tokenizer.json` / `*.safetensors`、または `*.gguf` 1 ファイル）。
通常の環境では使う必要はありません。

---

## 知識ベース v2（どんな話題でも、問いの型に合う答えを引く土台）

`snipher/data/kb.json` は **人が書いた日本語そのもの**です。中身は
`tools/kb_data/*.py`（7 ドメイン・1 トピック 1 エントリの DSL）に書き、
`tools/build_kb.py` が検証してから 1 ファイルに束ねます（生成物は手で編集しない）。

    文字バイグラム + 最長一致の辞書 → BM25(k1=1.4, b=0.72)
      → 話題の決定（主題語 +5 / 先頭語 +3 / 文末の名詞 +4 / 共有 alias は減点）
      → 問いの型に合う欄を選ぶ → 確度ゲート（coverage と alias ヒットで採用/不採用）

v1 との違いは **1 トピックが問いの型ごとの答えを持つ**ことです:

| 欄 | 答える問い | 例（観葉植物） |
|---|---|---|
| `def` | Xとは / Xって何 | 「光合成で養分を作り、室内で育てる植物です。」 |
| `why` | なぜ / どうして | 「水の多すぎか光の不足がほとんどです。」 |
| `how` | 作り方 / やり方 | 手順を番号付きで並べる |
| `when` `where` `who` `cost` | いつ / どこ / 誰 / いくら | 時期・場所・人・値段 |
| `tips` | コツ / 困りごと | 「葉が黄色いときは置き場所を見直します。」 |
| `opinion` | 好き？ / おすすめは？ | 一人称の感想（「私は〜だと思います」） |
| `qa` | 型の分からない具体質問 | 「葉が黄色い」→ 対処をそのまま返す |
| `followups` | （応答の末尾に 1 つ） | 「植物を育てていますか。」 |

現状: **199 話題 / 611 事実 / 426 問答 / 395 手順 / 53 理由 / 196 感想**
（食事・自然・文化・技術・生活・動物・雑談の 7 ドメイン）。検索は 1 発話 **1.2 ms**、
材料が無ければ `None`（＝知らない）を返すので、composer は正直に「分かりません」と言えます。

```bash
python tools/build_kb.py                       # 検証 + 生成（1 秒）
curl -s "localhost:8000/api/kb?q=観葉植物の葉が黄色い" | jq '.answer'
```

`kb.json` は **学習データも兼ねます**: 内蔵ニューラルコアと n-gram LM の両方が
この文を教師にするので、知識を増やすほど文章も自然になります（好循環）。

## 語彙テーブル（パラメータの本体）

ルール/確率経路の品質はテーブル規模で決まる。`tools/build_lexicon.py` が
唯一の編集窓口（生成物 `snipher/data/*.json` は直接触らない）:

| テーブル | エントリ | | テーブル | エントリ |
|---|---|---|---|---|
| 名詞 | 770 | | 助動詞 | 51 |
| 動詞 | 226 | | 副詞 | 93 |
| 形容詞 | 164 | | 接続詞 | 35 |
| 助詞 | 60 | | 文型パターン | 26 |
| 意図（intents） | 71 | | コーパス文 | 51 |

ビルド時に自動で **活用の整合性検証**（五段の語幹行、形容詞の活用級、
キー/スロット名のホワイトリスト、topic_affinity の実在確認）を通すので、
壊れた語彙が本番データに入ることはありません（`検証 OK` が出たら成功）。

## 未知文字の学習（LFM2.5 のパラメータを適用）

- **自動**: 入力に `𠮷` や絵文字などの未知文字があると自動検出し、
  LFM2.5 の**事前学習済み断片埋め込みの合成**で予約トークンとして即学習します
  （torch バックエンド。UI 操作不要・応答をブロックしません）
- **深学習（任意）**: 例文から埋め込み行のみを数ステップ勾配更新し、
  ディスクに永続化（再起動後も有効）。UI の「未知文字の学習」ドロワーから実行
- **GGUF バックエンド**: byte-fallback トークナイザのため未知文字は原理的に
  発生しません（すべての文字・絵文字がそのまま読み書きできます）

## テンプレートがない時の生成

chat template は 3 段のフォールバックで常に生成できます:

1. tokenizer / GGUF メタデータ同梱のネイティブテンプレート
2. Snipher 内蔵の ChatML テンプレート（LFM2 系 `<|im_start|>` 形式）
3. テンプレートなし（素の completion。UI の「テンプレート: なし」で選択可）

---

## 独自確率式（高速コア）

```
S(w) = alpha * logP_freq      … 頻度事前分布(コーパス統計)
     + beta  * logP_trans     … 品詞遷移(マルコフ)
     + gamma * Q_role         … 格・役割適合(動詞が要求する助詞)
     + delta * Q_inflect      … 活用整合(活用形と助動詞の一致)
     + epsilon * Q_register   … 文体整合(です/ます体の一貫性)
     + zeta   * Q_topic       … トピック一貫性(意味クラスの一致)

P(w) = softmax( S(w) / temperature )
```

重みは `snipher/data/config.json` の 6 個のスカラーのみ。下書きの各スロット決定の
確率から confidence を計算し、ニューラルコアに渡すかどうかの内部判定に使います。

---

## REST API

| メソッド | パス | 説明 |
|---------|------|------|
| GET | `/` | ホワイトテーマのチャット UI |
| GET | `/health` | ヘルスチェック |
| GET | `/info` | モデル情報・パラメータ総数（テーブル/知識ベース/蒸留コアの内訳つき） |
| POST | `/analyze` | `{"text": "..."}` を解析（文構造・助動詞・要点） |
| POST | `/generate` | 確率的な日本語文の生成 |
| GET | `/api/status` | Snipher Core の状態（バックエンド・自動取得・学習済み語彙） |
| POST | `/api/chat` | SSE ストリーミング応答。`mode: auto/fast/lfm/light` |
| POST | `/api/complete` | 助動詞の補い（断片文の補完: ルール + 必要なら LFM2.5） |
| GET | `/api/neural` | 内蔵ニューラルコア（蒸留スナップショット）の状態 |
| POST | `/api/neural/probe` | 蒸留コアに生成・補完・採点させてみる |
| POST/GET | `/api/neural/rebuild` | 蒸留コアを自分で再ビルド（常駐環境のみ・progress 取得可） |
| GET | `/api/kb` | 知識ベース検索（`?q=...`、`answer` は採用された返信材料） |
| GET | `/api/model/remote` | LFM2.5 リモート委譲の疎通確認 |
| GET | `/api/model/acquire` | 自動取得ジョブの進捗 |
| POST | `/api/model/fetch` | 自動取得の再試行 |
| GET | `/api/model/import` | 手動取り込みディレクトリの状態 |
| POST | `/api/model/upload` | （最後の手段）モデルファイルの手動取り込み |
| POST | `/api/model/load` | 取り込み済み/指定ローカルモデルで再ロード |
| POST | `/api/vocab/check` | テキスト中の未知文字をスキャン |
| POST | `/api/learn` | 未知文字の学習（`mode: instant/deep`・torch バックエンド） |
| GET/DELETE | `/api/learn/status` `/api/learn` | 深学習ジョブ状態 / 学習済み語彙の全消去 |

```bash
# チャット（SSE）
curl -N -X POST http://localhost:8000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"今日はいい天気だね"}],"mode":"auto"}'

# 知識ベース検索
curl -s "localhost:8000/api/kb?q=雨はなぜ降るの"

# 内蔵ニューラルコア: 状態 / 生成テスト / 再蒸留（サーバーが自分で作り直す）
curl -s localhost:8000/api/neural
curl -s -X POST localhost:8000/api/neural/probe -H 'content-type: application/json' \
     -d '{"text":"今日は天気"}'
curl -s -X POST localhost:8000/api/neural/rebuild -H 'content-type: application/json' -d '{"profile":"tiny"}'

# LFM2.5 リモート委譲の疎通確認
curl -s localhost:8000/api/model/remote

# 助動詞の補い
curl -X POST http://localhost:8000/api/complete \
  -H 'Content-Type: application/json' \
  -d '{"text":"私は猫が好き"}'
```

---

## 環境変数

| 変数 | 既定 | 説明 |
|---|---|---|
| `SNIPHER_LFM_BACKEND` | `auto` | `auto/gguf/torch/off` |
| `SNIPHER_LFM_MODEL` | 公式モデルID | ローカルディレクトリ or HF モデルID |
| `SNIPHER_LFM_GGUF` | – | GGUF ファイルの直接指定 |
| `SNIPHER_LFM_GGUF_QUANT` | `Q4_K_M` | 自動取得する GGUF の量子化 |
| `SNIPHER_LFM_URLS` | – | 追加の直接ダウンロード URL（カンマ/改線区切り） |
| `SNIPHER_LFM_SHARE_PAGE` | ギガワタス共有 | 共有ミラーのページ URL |
| `SNIPHER_LFM_CACHE_DIR` | `var/models` | 取得したモデルのキャッシュ先 |
| `SNIPHER_LFM_AUTO_FETCH` | `1` | 起動時の自動取得 |
| `SNIPHER_LFM_QUANTIZE` | `1` | torch バックエンドの INT8 量子化 |
| `SNIPHER_ASSIST_THRESHOLD` | `0.35` | これ未満の確度ならニューラルコアが生成 |
| `SNIPHER_LIGHT_CORE` | `auto` | 内蔵蒸留コア: `auto`（重みが有れば使用）/ `on` / `off` |
| `SNIPHER_LM` | `auto` | 巨大 n-gram LM: `auto`（lm.npz が有れば使用）/ `on` / `off` |
| `SNIPHER_LIGHT_MAX_CHARS` | `64` | 蒸留コア 1 応答の生成上限文字数 |
| `SNIPHER_LIGHT_GATE` | `0.34` | この確信度を下回ったときだけフルウェイト起動を裏で準備 |
| `SNIPHER_LFM_REMOTE_URL` | 空 | LFM2.5 を常駐させたホスト（別の Snipher でも可）へ生成を委譲 |
| `SNIPHER_LFM_REMOTE_TOKEN` | 空 | リモート委譲の Bearer トークン |
| `SNIPHER_LLM_THREADS` | 自動 | llama.cpp のスレッド数 |
| `SNIPHER_LLAMA_SERVER` | 自動探索 | llama-server バイナリのパス |

---

## テスト

```bash
.venv/bin/python -m pytest tests/ -q          # 192 passed, 3 skipped（約 14 秒）
```

| ファイル | 守っている契約 |
|---|---|
| `test_knowledge_v2.py` | 問いの型に合う欄を返す / 知らないことは `None` / 1 文字の話題名も索引される |
| `test_composer.py` | 材料が無ければ正直に言う / 定型文のゴミを出さない / 繰り返しても文が変わる / `validate()` が壊れた日本語を落とす |
| `test_lm.py` | 自然な文とシャッフル文を perplexity で区別する / 保存・読み込みでスコアが変わらない / 確信度の校正が単調 |
| `test_light_core.py` | 経路の契約（instant / knowledge / light / neural / fallback）・KV キャッシュが一括計算と一致する |
| `test_core.py` `test_core_api.py` `test_engine.py` | イベント契約・REST API・パラメータ総数の内訳 |

torch/transformers が無い環境では LFM2.5 系のテストだけ自動的にスキップされます
（**Snipher 本体の 4 層はすべてテストされます**）。
本物の GGUF がある環境では実推論テストも実行できます:

```bash
SNIPHER_TEST_GGUF=/path/to/LFM2.5-1.2B-JP-202606-Q4_K_M.gguf SNIPHER_TEST_GGUF_RUN=1 \
  pytest tests/test_gguf_backend.py -q
```

開発/CI 用の小型モデル（同アーキテクチャ）で全コードパスを検証することもできます:

```bash
python tools/make_test_model.py --out var/tiny-lfm2
SNIPHER_LFM_MODEL=var/tiny-lfm2 uvicorn snipher.api:app
```

---

## デプロイ

- **自分の PC / VPS（推奨）**: 上のクイックスタートの通り。GGUF バックエンドなら
  2GB 程度の RAM で動きます
- **Render**: `render.yaml` が自動検出されます（無料プランの 512MB では
  ニューラルコアは動かないため、高速コアのフォールバックで応答します。
  スタンダード以上 + `pip install llama-cpp-python` をビルドコマンドに追加すれば
  フル動作）
- **Vercel**: `pyproject.toml` の `[tool.vercel]` によりサーバーレス起動。
  731MB のモデルは読み込まない（読み込めるサイズではない）ので、
  **同梱の 4 層（知識ベース 384 KiB + n-gram LM 2.3 MB + ニューラルコア 5.1 MB = 7.7 MB）**
  でフル機能動作する。
  `vercel.json` が `maxDuration=60 / memory=1024` とバンドル除外を設定済みで、
  サーバーレス環境では `SNIPHER_LFM_AUTO_FETCH` が自動で `off`
  （無駄なダウンロードを試みない）。より賢くしたければ
  `SNIPHER_LFM_REMOTE_URL` に LFM2.5 を常駐させた VPS の URL を 1 行入れるだけ。
- **Docker / VPS**: `pip install -r requirements.txt -r requirements-llm.txt`
  なら起動時にフルウェイトを自動取得。常駐なので T1 が生き、同じ UI のまま品質が上がる

---

## リポジトリ構成

```
snipher/
  core.py         # ★ Snipher Core: 解析 → 知識 → composer → LM 判定 → ニューラル → 昇格
  composer.py     # ★ 文の設計図（8 計画・枠の回転・validate）… v2 の中心
  lm.py           # ★ 巨大 n-gram 言語モデル（276,832 エントリ・流暢さの審判）
  knowledge.py    # ★ 知識ベース v2（BM25 + 最長一致辞書 + 問いの型の振り分け）
  api.py          # FastAPI（SSE チャット・自動取得 API・学習 API）
  engine.py       # 公開ファサード（解析/生成/info のパラメータ内訳）
  lexicon.py      # テーブルロード
  morphology.py   # 活用処理
  parser.py       # 構文解析(文構造/助動詞/要点)
  probability.py  # 独自確率式(softmax 確度)
  generator.py    # 確率的生成(スロット確度の記録)
  polisher.py     # 助動詞の補い・文体修復(ルール)
  responder.py    # 意図分類 + 対話テーブル応答
  data/           # 同梱する重みと知識（すべて生成物＋人が書いた日本語）
    kb.json         #   知識ベース v2（tools/build_kb.py → 199 話題 / 384 KiB）
    lm.npz          #   n-gram LM の重み（tools/build_lm.py → 2.3 MB）
    dialogues.json  #   人が書いた対話 220 組（学習の錨・手で編集する）
    *.json          #   語彙テーブル（tools/build_lexicon.py から生成）
    neural/core.npz #   内蔵ニューラルコアの重み（int8・5.1 MB・tools/distill_neural.py）
  neural/         # ★ LFM2.5 を蒸留した内蔵ニューラルコア（NumPy のみ）
    nn.py           # ShortConv/注意 Hybrid + 手書き backward + DecodeCache（KV キャッシュ）
    tokenizer.py    # 文字レベル語彙（未知文字が出ない）
    corpus.py       # 人が書いた対話 + kb.json + 文法生成から教師文を自動構築
    train.py        # Adam + warmup/cosine（epoch ごとのスナップショット対応）
    store.py        # int8 量子化 + zlib ヘッダの npz コンテナ
    core.py         # generate / reply / complete / score（KV キャッシュ使用）
    cache.py        # プロセス共通インスタンス（遅延ロード・スレッドセーフ）
  lfm/            # LFM2.5-1.2B-JP ニューラルコア（任意の昇格先）
    acquire.py      # モデルの全自動取得(マルチソース・レジューム)
    gguf_backend.py # llama.cpp バックエンド(CPU 最速)
    engine.py       # torch バックエンド(INT8 量子化・ストリーミング)
    learner.py      # 未知文字の学習(埋め込み合成 + 勾配更新)
    template.py     # テンプレートの 3 段フォールバック
    assist.py       # 高速コアの下書き/確信度(HybridAssist)
    config.py       # 設定(環境変数)
    vocab.py        # 未知文字スキャン
  web/chat.html   # ホワイトテーマのチャット UI
tools/
  kb_data/           # ★ 知識ベースの中身（人が書いた日本語・7 ドメイン 176 エントリ）
    __init__.py        #   T() DSL + validate()
    food.py nature.py culture.py tech.py life.py animals.py talk.py
  build_kb.py        # ★ kb.json の生成（v2 を優先し、v1 テーブルと 1 つに束ねる）
  build_lm.py        # ★ n-gram LM の学習 + 判別力の検証（約 17 秒）
  distill_neural.py  # ★ 内蔵ニューラルコアの蒸留ビルド（--profile tiny/base/big/huge）
  bench.py           # ★ 実測ベンチマーク（README の表はここから作る）
  build_lexicon.py   # 語彙テーブル生成 + 文法/活用検証
  fetch_model.py     # 事前取得 CLI(自動取得と同じロジック)
  make_test_model.py # 開発用小型モデル生成
tests/            # 単体テスト 192（知識ベース v2 / composer / LM / KV キャッシュ / 経路契約）
docs/             # 設計書(design.md / lfm.md)
demo.py           # 高速コアの CLI デモ
```

### 作り直し方（すべてローカル・外部データ 0）

```bash
python tools/build_kb.py          # 知識ベース   … 約 1 秒
python tools/build_lm.py          # n-gram LM    … 約 17 秒
python tools/distill_neural.py    # ニューラルコア … 約 60 分（epoch ごとに保存）
python tools/bench.py             # 実測して表を作る
```

---

## ライセンス

MIT（Snipher 本体）。LFM2.5-1.2B-JP の重みは Liquid AI のライセンスに従います
（HuggingFace リポジトリの LICENSE を参照）。
