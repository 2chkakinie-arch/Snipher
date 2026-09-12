# Snipher

**LFM2.5-1.2B-JP を「内部構造」として動かす、高速で賢い日本語チャット AI**

Snipher は 2 つの AI を並べたハイブリッドではありません。
Liquid AI の **LFM2.5-1.2B-JP**（1.17B パラメータ / 32K context / 日英対応）が
**Snipher 本体のニューラルコア**として内蔵され、確率的に不安な部分の文章生成・
助動詞の補い・未知文字の学習・テンプレートがない時の生成を担います。
挨拶などの確実な定形応答だけが数ミリ秒の高速コア（対話テーブル + 独自確率式 +
polisher）で即答されるため、**速度を維持したまま LFM2.5 レベルの日常会話**ができます。

- **全自動**: 起動するとモデルを自分で取得（レジューム対応・マルチソース）・
  ロード・量子化します。**ファイルのアップロードなどは一切不要**
- **CPU 最速**: llama.cpp + 公式 GGUF（Q4_K_M / 731MB）を自動選択。
  torch しか無い環境では動的 INT8 量子化で動作
- **シンプルなホワイトテーマ**のチャット UI（SSE ストリーミング・tok/s 表示）
- 依存の重い部分（torch / llama-cpp-python）はオプション。
  どちらも無い環境では超小型エンジン（502+ パラメータ）が自動フォールバック

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
   │ ② ニューラルコア = LFM2.5-1.2B-JP（内部構造）                   │
   │    ├─ 不安な部分の生成  下書きの確度が低い/質問/雑談 →              │
   │    │                  会話履歴ごと LFM2.5 が本文をストリーム生成    │
   │    ├─ 助動詞の補い     LFM 出力にも polisher を適用（二重に補完）    │
   │    ├─ 未知文字の学習   𠮷 や絵文字を学習済み埋め込みの合成で          │
   │    │                  1 トークン化（入力時に自動検出・自動学習）      │
   │    └─ テンプレート     native → 内蔵 ChatML → なし（素の生成）      │
   │                                                              │
   │ ③ モデル自動取得（起動時・全自動）                               │
   │    キャッシュ → SNIPHER_LFM_URLS → 共有ミラー → HuggingFace 公式   │
   │    → hf-mirror（Range レジューム・破損検証・失敗時は自動再試行）      │
   └──────────────────────────────────────────────────────────────┘
```

- 確実な定形応答は LFM を 1 トークンも消費せず即答 → **従来の速度を維持**
- 確率的に不安な応答だけ LFM2.5 が生成 → **賢さは LFM2.5 レベル**
- LFM が失敗・未ロードでも下書き/骨子で必ず応答 → **壊れない**

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
| GET | `/info` | 高速コアのモデル情報・パラメータ数 |
| POST | `/analyze` | `{"text": "..."}` を解析（文構造・助動詞・要点） |
| POST | `/generate` | 確率的な日本語文の生成 |
| GET | `/api/status` | Snipher Core の状態（バックエンド・自動取得・学習済み語彙） |
| POST | `/api/chat` | SSE ストリーミング応答。`mode: auto/fast/lfm` |
| POST | `/api/complete` | 助動詞の補い（断片文の補完: ルール + 必要なら LFM2.5） |
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
- **Vercel**: `pyproject.toml` の `[tool.vercel]` によりサーバーレス起動
  （エフェメラル環境のため高速コア中心の動作になります）

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
  data/           # 高速コアの全知識(JSON テーブル)
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
  fetch_model.py     # 事前取得 CLI(自動取得と同じロジック)
  make_test_model.py # 開発用小型モデル生成
tests/            # 単体テスト(82+)
docs/             # 設計書(design.md / lfm.md)
demo.py           # 高速コアの CLI デモ
```

---

## ライセンス

MIT（Snipher 本体）。LFM2.5-1.2B-JP の重みは Liquid AI のライセンスに従います
（HuggingFace リポジトリの LICENSE を参照）。
