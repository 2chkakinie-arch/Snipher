# Snipher

**超小型・確率的日本語AI（日本語のみ対応）+ LFM2.5-1.2B-JP ハイブリッドチャット**

動詞・助動詞・名詞・助詞・文構造を「あらかじめ決めておく」ことで、
量子化された大規模言語モデル（例: 1.2B）よりも**圧倒的に少ないパラメータ**で、
日本語の構文解析と高速な文章生成を実現する試みです。
さらにオプションで Liquid AI の **LFM2.5-1.2B-JP** をサーバー側で動かし、
**確率的に不安な返答だけをニューラル補正するハイブリッドモード**で、
速度を維持したまま賢い日常会話ができます。

- 全パラメータ数: **502+**（テーブル 496 + 確率式の重み 6 + 対話応答テーブル）
- 依存: `fastapi` / `uvicorn` / `pydantic` / `python-multipart`。GPU 不要
- 日本語の文構造・助動詞・要点を if 構文ベースのルールで解析
- 独自確率式による確率的な日本語文生成 + **意図別の対話応答テーブル**
- **助動詞の補い**: 文末・助動詞・文体の欠落をルールで自動補正（polisher）

設計の詳細は [docs/design.md](docs/design.md) を参照してください。

---

## 独自確率式

```
S(w) = alpha * logP_freq      … 頻度事前分布(コーパス統計)
     + beta  * logP_trans     … 品詞遷移(マルコフ)
     + gamma * Q_role         … 格・役割適合(動詞が要求する助詞)
     + delta * Q_inflect      … 活用整合(活用形と助動詞の一致)
     + epsilon * Q_register   … 文体整合(です/ます体の一貫性)
     + zeta   * Q_topic       … トピック一貫性(意味クラスの一致)

P(w) = softmax( S(w) / temperature )
```

重みは `snipher/data/config.json` の6個のスカラーのみ。

---

## ⚡ ハイブリッドモード: 速さを保ったまま賢く

Snipher-mini（超小型エンジン）が**常に一瞬で下書きを作り**、
その確度に応じてサーバー側の LFM2.5-1.2B-JP が介入します:

```
ユーザー発話
   ↓ 数ミリ秒
Snipher-mini: 意図判定(対話テーブル) + 確率的文生成 + 助動詞の補い(polisher)
   ↓
下書きの各スロット決定に確率(softmax確率)が付く → 重み平均 = confidence
   ├─ confidence ≥ 閾値(0.35) → そのまま返答（LFM 未使用・最速）
   └─ confidence < 閾値        → LFM2.5 に「下書きを自然な返答に書き直させる」
                                  （出力は短い 64 トークン上限なので高速）
```

- **確率的に不安な部分だけ**ニューラル生成に任せるので、全体の速度はほぼ現状維持
- 助動詞・文体の欠落（「〜だ。」「〜ますです」「飲むます」等）は LFM が無くても
  polisher がルールで補う
- UI には「⚡ 高速経路」か「✨ LFM 補正」か、下書きの確度、補正件数を表示
- 環境変数 `SNIPHER_ASSIST_THRESHOLD`（既定 0.35）で介入の度合いを調整

---

## 🚀 LFM2.5-1.2B-JP チャット（ニューラルエンジン）

Liquid AI の **LFM2.5-1.2B-JP**（1.17B / 32K context）の学習済みパラメータを載せて、
**CPU でも高速な日本語の日常会話**ができるホワイトテーマのチャット UI を追加。

- **高速化**: 動的 INT8 量子化 + 短いコンテキスト + SSE ストリーミング。
  応答速度（tok/s）は UI に実測値で表示
- **未知文字の学習**: `𠮷` や `🚀` などトークナイザが保持できない文字を、
  **事前学習済み断片埋め込みの合成**で即時に学習（さらに例文から
  埋め込みのみの勾配更新で深学習も可能）。学習結果は永続化され、
  モデルはその文字を 1 意味トークンとして読み・書きできる
- **テンプレートがない時の生成**: ネイティブ chat template →
  内蔵 ChatML テンプレート → テンプレートなし（素の生成）の
  3 段フォールバックで常に応答可能
- **フォールバック**: torch 未インストールやモデル未取得の環境では
  Snipher-mini+（対話テーブル + polisher）が自動で応答

```bash
# ニューラルエンジン込みで起動（初回に ~2.4GB を HuggingFace から取得）
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt -r requirements-llm.txt
.venv/bin/uvicorn snipher.api:app --host 0.0.0.0 --port 8000
```

`http://localhost:8000` でチャット UI が開く。オフライン環境では
`SNIPHER_LFM_MODEL=/path/to/model` でローカルのモデルを指定できる。
設計の詳細は [docs/lfm.md](docs/lfm.md) を参照。

---

## 📥 HuggingFace に接続できない環境でのモデル取り込み

このサンドボックスのように `huggingface.co` への外向き接続が遮断されている
環境では、transformers の自動ダウンロードが失敗します（起動時に 5 秒で
検出して軽量モードに落ちます）。対応は 2 つ:

### 方法 1: ブラウザからアップロード（UI 統合）

1. 自分の PC で [LiquidAI/LFM2.5-1.2B-JP-202606](https://huggingface.co/LiquidAI/LFM2.5-1.2B-JP-202606)
   から `config.json` / `generation_config.json` / `tokenizer.json` /
   `tokenizer_config.json` / `model.safetensors`（+`chat_template.jinja`）をダウンロード
2. チャット UI の「モデル管理（オフライン環境への取り込み）」を開いてドロップ
3. 「アップロード済みモデルをロード」→ エンジンがホットリロードされ、
   LFM2.5 + ハイブリッド補正が有効になる

必要ファイルの判定は `GET /api/model/import` が行い、アップロードは
`POST /api/model/upload`、ロードは `POST /api/model/load` が担当します。

### 方法 2: ミラー対応のフェッチツール（自分のマシンで実行）

```bash
python tools/fetch_model.py --dest var/models/LFM2.5-1.2B-JP
# → HuggingFace → hf-mirror.com の順に自動試行。レジューム対応・標準ライブラリのみ
SNIPHER_LFM_MODEL=var/models/LFM2.5-1.2B-JP uvicorn snipher.api:app --host 0.0.0.0 --port 8000
```

### 開発/CI 用の小型モデル

本物の 2.4GB を持たない環境では、同アーキテクチャの小型モデルで全コードパス
（テンプレート / 未知文字学習 / ストリーミング / ハイブリッド補正）を検証できます:

```bash
python tools/make_test_model.py --out var/tiny-lfm2
SNIPHER_LFM_MODEL=var/tiny-lfm2 uvicorn snipher.api:app   # 動作確認用
```


---

## クイックスタート（超小型エンジンのみ）

```bash
# 依存インストール
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# API サーバ起動
.venv/bin/uvicorn snipher.api:app --host 0.0.0.0 --port 8000
```

ブラウザで `http://localhost:8000` を開くとチャット UI、
`http://localhost:8000/classic` に旧 UI、
`http://localhost:8000/docs` で Swagger UI が使えます。

### CLI デモ

```bash
.venv/bin/python demo.py
```

### Python API

```python
from snipher import SnipherEngine

engine = SnipherEngine(seed=42)

# 解析: 文構造・助動詞・要点を抽出
result = engine.analyze("私は猫が好きです。")
print(result["structure"], result["key_points"], result["topic"])

# 生成: 確率的に日本語文を出力
out = engine.generate(prompt="旅行", register="polite", n=3)
for s in out["sentences"]:
    print(s["text"])

# モデル情報
print(engine.info())
```

---

## REST API

| メソッド | パス | 説明 |
|---------|------|------|
| GET | `/` | ホワイトテーマのチャット UI |
| GET | `/info` | モデル情報・パラメータ数 |
| GET | `/health` | ヘルスチェック |
| POST | `/analyze` | `{"text": "..."}` を解析 |
| POST | `/generate` | `{"prompt": "...", "register": "polite|casual", "tense": "nonpast|past", "n": 1, "seed": 1}` で生成 |
| GET | `/api/status` | エンジン状態 + ハイブリッド補正の設定 |
| POST | `/api/chat` | SSE ストリーミング応答（`hybrid:false` で LFM 直接・`true` でハイブリッド強制） |
| GET | `/api/model/import` | ローカル取り込みの状態（不足ファイル判定） |
| POST | `/api/model/upload` | モデルファイルをアップロード（`activate=1` で即ホットロード） |
| POST | `/api/model/load` | アップロード済み/指定ローカルモデルで再ロード |
| POST | `/api/vocab/check` | テキスト中の未知文字をスキャン |
| POST | `/api/learn` | 未知文字を学習（`mode: instant` / `deep`） |
| GET/DELETE | `/api/learn/status` `/api/learn` | 深学習ジョブ状態 / 学習済み語彙の全消去 |

```bash
curl -X POST http://localhost:8000/analyze \
  -H 'Content-Type: application/json' \
  -d '{"text":"昨日は映画を見ました。"}'

curl -X POST http://localhost:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"猫","register":"polite","n":2}'
```

---

## テスト

```bash
.venv/bin/python -m pytest tests/ -q
```

---

## デプロイ

### Vercel

1. リポジトリを GitHub に push
2. Vercel でこのリポジトリをインポート
3. `pyproject.toml` の `[tool.vercel] entrypoint = "snipher.api:app"` により、
   FastAPI アプリが Python ランタイムの Serverless Function として自動認識される
   (追加のルーティング設定は不要。全リクエストがアプリに転送される)
4. 依存パッケージは `pyproject.toml` の `[project.dependencies]` にも定義してあるため、
   Vercel の Python ランタイムが `fastapi` / `uvicorn` / `pydantic` を確実にインストールできる

### Render

1. Render で「New Web Service」→ このリポジトリを接続
2. `render.yaml` が自動検出される（`pip install -r requirements.txt` → `uvicorn snipher.api:app`）

---

## リポジトリ構成

```
snipher/          # 本体パッケージ
  data/           # モデルの全知識(JSON テーブル = パラメータ)
    responses.json  # 意図別の対話応答テーブル(ハイブリッドの下書き源)
  lexicon.py      # テーブルロード
  morphology.py   # 活用処理
  parser.py       # 構文解析(文構造/助動詞/要点)
  probability.py  # 独自確率式(候補の softmax 確率も返す)
  generator.py    # 確率的生成(スロット決定の確度を記録)
  polisher.py     # 助動詞の補い・文体修復(ルールのみ・数ミリ秒)
  responder.py    # 意図分類 + 対話テーブル応答
  engine.py       # 統括 API
  lfm/            # LFM2.5-1.2B-JP ニューラルエンジン
    assist.py       # ハイブリッド補正(下書き→確度→LFM 書き直し)
  web/chat.html   # ホワイトテーマのチャット UI
  api.py          # FastAPI
tools/
  fetch_model.py    # HuggingFace/ミラーからモデル取得(レジューム対応)
  make_test_model.py # 開発用小型モデル生成
tests/            # 単体テスト
docs/             # 設計書(design.md / lfm.md)
demo.py           # CLI デモ
vercel.json       # Vercel 設定
render.yaml       # Render 設定
```
