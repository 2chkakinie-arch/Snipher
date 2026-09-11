# Snipher

**超小型・確率的日本語AI（日本語のみ対応）+ LFM2.5-1.2B-JP チャット**

動詞・助動詞・名詞・助詞・文構造を「あらかじめ決めておく」ことで、
量子化された大規模言語モデル（例: 1.2B）よりも**圧倒的に少ないパラメータ**で、
日本語の構文解析と高速な文章生成を実現する試みです。
さらにオプションで Liquid AI の **LFM2.5-1.2B-JP** を載せ、
未知文字の学習とテンプレートフォールバック付きの高速な日常会話ができます。

- 全パラメータ数: **476**（テーブル 470 + 確率式の重み 6）
- 依存: `fastapi` / `uvicorn` / `pydantic` のみ。GPU 不要、モデルファイル不要
- 日本語の文構造・助動詞・要点を if 構文ベースのルールで解析
- 独自確率式による確率的な日本語文生成

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
  従来の Snipher-mini（476 パラメータ）が自動で応答

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
| GET | `/` | 日本語 UI |
| GET | `/info` | モデル情報・パラメータ数 |
| GET | `/health` | ヘルスチェック |
| POST | `/analyze` | `{"text": "..."}` を解析 |
| POST | `/generate` | `{"prompt": "...", "register": "polite|casual", "tense": "nonpast|past", "n": 1, "seed": 1}` で生成 |

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
  lexicon.py      # テーブルロード
  morphology.py   # 活用処理
  parser.py       # 構文解析(文構造/助動詞/要点)
  probability.py  # 独自確率式
  generator.py    # 確率的生成
  engine.py       # 統括 API
  api.py          # FastAPI
tests/            # 単体テスト
docs/design.md    # 設計書
demo.py           # CLI デモ
vercel.json       # Vercel 設定
render.yaml       # Render 設定
```
