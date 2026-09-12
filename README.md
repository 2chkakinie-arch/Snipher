# Snipher

**LFM2.5-1.2B-JP を「内部構造」として動かす、高速で賢い日本語チャット AI**

Snipher は 2 つの AI を並べたハイブリッドではありません。Liquid AI の
**LFM2.5-1.2B-JP**（1.17B パラメータ / 32K context / 日英対応）の仕事を
**Snipher 本体の内側**に統合した設計です: 確率的に不安な部分の文章生成・
助動詞の補い・未知文字の学習・テンプレートがない時の生成をニューラルコアが担い、
挨拶などの確実な定形応答だけを数ミリ秒の高速コア（対話テーブル + 独自確率式 +
polisher）が即答するので、**今の速度を保ったまま賢く**なります。

そして重要なのは、そのニューラルコアが **3 階層の自動選択**になっていること。
Vercel のようなサーバーレスでは 731MB(GGUF) / 2.2GB(safetensors) を読む物理的な
余地が無い（関数バンドル 500MB・永続ディスク無し・コールドスタート）ので、
**LFM2.5 と同じアーキテクチャを蒸留した 92 万パラメータのスナップショット
（NumPy・約 0.9MB）をリポジトリに同梱**して動かします。ダウンロードも
アップロードも不要、ロードは数十ミリ秒です。

- **全自動**: 常駐環境（VPS / Render / Docker）では起動時にフルウェイトを
  自分で取得（レジューム対応・マルチソース）・ロード・量子化。
  サーバーレスでは同梱スナップショット＋BM25 知識ベースで即稼働。
- **速度維持**: 即答経路はニューラルコアを 1 トークンも消費しない（1〜15ms）。
  生成経路も 1 文字 2〜3ms（NumPy）で、SSE ストリーム表示。
- **どんな会話でも**: 語彙テーブル（名詞 770 / 動詞 226 / 形容詞 164 …）と
  知識ベース（66 トピック / 187 事実）を土台に、事実検索＋生成で応答。
- **壊れない**: フルウェイト取得失敗・依存なし・オフライン、いずれでも
  高速コアが安全な応答を返す。
- シンプルな**ホワイトテーマ**のチャット UI（SSE ストリーミング・tok/s 表示）

---

## 内部構造

```
ユーザー発話
   │
   ▼ ── Snipher Core ─────────────────────────────────────────────┐
   │ ① 高速コア（数ミリ秒・常に動作）                                │
   │    ├─ 意図判定        挨拶/感謝/謝罪…の確実な定形 → ⚡ そのまま即答 │
   │    ├─ 確率的な下書き   独自確率式が各スロットの softmax 確度を記録   │
   │    └─ 助動詞の補い     文末・助動詞・文体の欠落をルールで即時修復    │
   │                                                              │
   │ ② 知識ベース（BM25・約 0.1ms）                                 │
   │    └─ snipher/data/kb.json を検索し事実を引用（確度が低ければ不採用） │
   │                                                              │
   │ ③ ニューラルコア（LFM2.5 の仕事）・3 階層を自動選択                │
   │    T1 ローカル フルウェイト  llama.cpp(GGUF 731MB) / torch(INT8)  │
   │       … 常駐環境では起動時に全自動で取得・量子化・ロード            │
   │    T2 リモート委譲         SNIPHER_LFM_REMOTE_URL のホストに生成を依頼│
   │       … Vercel から VPS 常駐の LFM2.5 をそのまま使える            │
   │    T3 内蔵蒸留コア         LFM2.5 Hybrid と同じ構造の小型 LM        │
   │       … 語彙テーブル＋kb.json から自動蒸留 / int8 で同梱（0.9MB）    │
   │                                                              │
   │    共通の役割: 不安な部分の生成 / 助動詞の補い / perplexity でゲート  │
   │    どの階層の出力も必ず polisher を通して整形（暴走しても壊れない）     │
   └──────────────────────────────────────────────────────────────┘
```

- 確実な定形応答はニューラルコアを 1 トークンも消費せず即答 → **従来の速度を維持**
- 確率的に不安な応答だけニューラルコアが生成 → **賢さは LFM2.5 レベルに段階昇格**
- 内蔵蒸留コアの perplexity 確信度が閾値を切ったときだけ、フルウェイトの
  起動を裏で進める（応答は待たせない）
- 生成が失敗・空でも下書き/骨子で必ず応答 → **壊れない**

### 内蔵ニューラルコア（`snipher/neural/`）

| 項目 | 内容 |
|---|---|
| 構造 | ShortConv → SelfAttn → GLU の 6 ブロック Hybrid（LFM2 と同じ系譜）/ RMSNorm / SiLU ゲート |
| パラメータ | 920,640（int8 量子化 + 行別スケール → 重み 876 KiB） |
| 語彙 | 文字レベル 720（未知文字が原理的に出ない） |
| 学習データ | 語彙テーブル＋kb.json から自動生成した 28,572 文書（外部データ 0・手作業 0） |
| 品質 | 文字 top-1 精度 **0.810** / perplexity **2.16**（val・10 epoch 約 18 分） |
| 依存 | numpy のみ（torch / transformers / llama.cpp 不要） |
| ロード | 数十ミリ秒・ダウンロード不要（Vercel でも即動く） |

語彙や知識を増やしたら、スナップショットも自分で作り直せる（全部ローカル実行）:

```bash
python tools/build_lexicon.py            # 辞書テーブルをマージ + 文法検証
python tools/build_kb.py                 # 知識ベース kb.json を再生成
python tools/distill_neural.py           # 内蔵コアを再蒸留（10 epoch・約 13 分）
python tools/distill_neural.py --profile tiny   # 数秒の検証用ビルド
```

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

## 知識ベース（どんな話題でも引ける土台）

`snipher/data/kb.json` は「会話で使い回せる事実の集合」です。
`tools/build_kb.py` から生成し（手で編集しない）、検索は 1 クエリ **約 0.1ms**。

    文字バイグラム + ASCII トークン化 → BM25(k1=1.4, b=0.72) → 確度ゲート

- 発話を tokenize して上位トピックを引き、**クエリ覆盖率（coverage）と
  alias ヒットで採用/不採用を判定**（BM25 の相対スコアはゲートに使わない）
- 採用された事実はそのまま応答に使い、内蔵ニューラルコアには
  「事実を伝えた後の会話の延续」だけを作らせる（ハルシネーションを減らす）
- `kb.json` は学習データも兼ねる: 内蔵ニューラルコアの蒸留時は
  この事実文がそのまま教師になる（知識を増やす → モデルも強くなる好循環）

現状: **66 トピック / 187 事実 / 107 の質問 / 110 の応答**（天気・睡眠・防災・
AI・日本語・猫・コーヒー…）。`python tools/build_kb.py` に項目を足せば、
辞書テーブルと同じく再生成だけで知識が増えます。

```bash
curl -s "localhost:8000/api/kb?q=なぜ雨が降るの" | jq '.answer'
```

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
| `SNIPHER_LIGHT_MAX_CHARS` | `64` | 蒸留コア 1 応答の生成上限文字数 |
| `SNIPHER_LIGHT_GATE` | `0.34` | この確信度を下回ったときだけフルウェイト起動を裏で準備 |
| `SNIPHER_LFM_REMOTE_URL` | 空 | LFM2.5 を常駐させたホスト（別の Snipher でも可）へ生成を委譲 |
| `SNIPHER_LFM_REMOTE_TOKEN` | 空 | リモート委譲の Bearer トークン |
| `SNIPHER_LLM_THREADS` | 自動 | llama.cpp のスレッド数 |
| `SNIPHER_LLAMA_SERVER` | 自動探索 | llama-server バイナリのパス |

---

## テスト

```bash
.venv/bin/python -m pytest tests/ -q
```

torch/transformers が無い環境ではニューラル系のテストは自動的にスキップされます。
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
  **同梱の内蔵蒸留コア（0.9MB）＋BM25 知識ベース＋高速コア**でフル機能動作する。
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
  core.py         # ★ Snipher Core: 高速コア + LFM2.5 ニューラルコアの統括
  api.py          # FastAPI（SSE チャット・自動取得 API・学習 API）
  engine.py       # 高速コアの公開ファサード（解析/生成）
  lexicon.py      # テーブルロード
  morphology.py   # 活用処理
  parser.py       # 構文解析(文構造/助動詞/要点)
  probability.py  # 独自確率式(softmax 確度)
  generator.py    # 確率的生成(スロット確度の記録)
  polisher.py     # 助動詞の補い・文体修復(ルール)
  responder.py    # 意図分類 + 対話テーブル応答
  knowledge.py    # ★ BM25 知識ベース（検索 0.1ms・確度ゲート付き）
  data/           # 高速コアの全知識(JSON テーブル)
    kb.json       #   知識ベース（tools/build_kb.py から生成）
    neural/core.npz  # 内蔵ニューラルコアの重み（int8・0.9MB・tools/distill_neural.py）
  neural/         # ★ LFM2.5 を蒸留した内蔵ニューラルコア（NumPy のみ）
    tokenizer.py    # 文字レベル語彙（未知文字が出ない）
    nn.py           # ShortConv/注意 Hybrid + 手書き backward（勾配検証済み）
    corpus.py       # 語彙テーブル+kb.json から教師文を自動生成（品質フィルタ付き）
    train.py        # Adam + warmup/cosine の学習ループ
    store.py        # int8 量子化 + zlib ヘッダの npz コンテナ
    core.py         # generate / reply / complete（助動詞の補い）/ score
    cache.py        # プロセス共通インスタンス（遅延ロード・スレッドセーフ）
  lfm/            # LFM2.5-1.2B-JP ニューラルコア
    acquire.py      # ★ モデルの全自動取得(マルチソース・レジューム)
    gguf_backend.py # ★ llama.cpp バックエンド(CPU 最速)
    engine.py       # torch バックエンド(INT8 量子化・ストリーミング)
    learner.py      # 未知文字の学習(埋め込み合成 + 勾配更新)
    template.py     # テンプレートの 3 段フォールバック
    assist.py       # 高速コアの下書き/確信度(HybridAssist)
    config.py       # 設定(環境変数)
    vocab.py        # 未知文字スキャン
  web/chat.html   # ホワイトテーマのチャット UI
tools/
  build_lexicon.py   # ★ 語彙テーブル生成 + 文法/活用検証（data/*.json は生成物）
  build_kb.py        # ★ 知識ベース kb.json の生成
  distill_neural.py  # ★ 内蔵ニューラルコアの蒸留ビルド（--profile tiny/base/big）
  fetch_model.py     # 事前取得 CLI(自動取得と同じロジック)
  make_test_model.py # 開発用小型モデル生成
tests/            # 単体テスト(90+)
docs/             # 設計書(design.md / lfm.md)
demo.py           # 高速コアの CLI デモ
```

---

## ライセンス

MIT（Snipher 本体）。LFM2.5-1.2B-JP の重みは Liquid AI のライセンスに従います
（HuggingFace リポジトリの LICENSE を参照）。
