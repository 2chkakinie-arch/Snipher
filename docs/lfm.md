# LFM2.5-1.2B-JP ニューラルチャット設計

Snipher に Liquid AI の **LFM2.5-1.2B-JP**（1.17B / LFM2 hybrid アーキテクチャ /
32K context）の学習済みパラメータを載せ、CPU でも高速な日本語の日常会話を
実現するニューラルエンジンの設計メモ。

従来の超小型エンジン（~500 パラメータ・ルールベース + 対話テーブル）は、
ニューラルエンジンが使えない環境（torch 未インストール / モデル未ダウンロード /
オフライン）でのフォールバックとしてそのまま残してある。

---

## 0. ハイブリッド補正（速度を維持したまま賢く）

「LFM に全部を作らせる」のではなく、**Snipher-mini が常に下書きを作り、
確率的に不安な部分だけ LFM が書き直す**二段構え:

```
ユーザー発話
  │  <数ミリ秒>
  ▼
Snipher-mini 下書き
  ├─ 意図判定     ... responses.json の対話テーブル(挨拶/感謝/質問…)
  ├─ 確率的補完   ... generator が話題に合わせた1文を生成
  │                  (各スロット決定の softmax 確率を記録)
  └─ 助動詞の補い ... polisher が文末・助動詞・文体をルール修復
  │
  ▼
confidence = スロット確率の重み平均(述語 1.5 / 名詞 1.0)
  ├─ ≥ SNIPHER_ASSIST_THRESHOLD(既定 0.35)
  │     → そのまま返答。LFM は 1 トークンも消費しない(最速経路)
  └─ < 閾値
        → LFM2.5 に書き直し依頼(既定 64 トークン上限)
          「相手の発話 + 下書き → 自然な返答(1〜2文)」
          出力が空でも下書きにフォールバックするので壊れない
```

ポイント:

- 下書きは常に完成しているため、LFM が何らかの理由で失敗しても応答は出る
- 補正対象が短いので、応答全体を LFM で作るより速い
- LFM が無い環境では、不確実な生成文を安全な骨子(base_text)に退避させ、
  テーブル由来の確実な文だけで返す(品質の下限を守る)
- 助動詞の補いは LFM の有無にかかわらずルールで実行(マイクロ秒単位)

## 1. オフライン環境でのモデル取り込み

`huggingface.co` への外向き接続が遮断された環境（CI サンドボックス等）では
自動ダウンロードが失敗する。`LfmEngine._probe_hf` が 5 秒の事前確認を行い、
失敗なら軽量モードへ素早く落ちる（ハングしない）。モデルの入手手段は 2 つ:

1. **ブラウザからのアップロード（UI 統合）**
   - 「モデル管理」パネルにモデルファイルをドロップ → `POST /api/model/upload`
   - `config.json` + トークナイザ + `*.safetensors` が揃うと complete 判定
     （`GET /api/model/import` が不足ファイルを列挙）
   - 「ロード」→ `POST /api/model/load` → `LfmEngine.reload()` がホットスワップ
2. **`tools/fetch_model.py`（自分のマシンで実行）**
   - HuggingFace → hf-mirror.com の順に自動試行、Range リクエストでレジューム対応
   - 標準ライブラリのみで動作
   - 取得後 `SNIPHER_LFM_MODEL=<dir>` で起動、または同じく UI からロード

---

## 全体構成

```
snipher/
├── polisher.py       # 助動詞の補い・文体修復(ルールのみ・LFM 不要)
├── responder.py      # 意図分類 + 対話テーブル応答(下書き生成)
├── lfm/
│   ├── config.py     # 環境変数 SNIPHER_LFM_* の設定
│   ├── engine.py     # ロード / INT8 量子化 / ストリーミング生成 / 学習ジョブ / ホットスワップ
│   ├── assist.py     # ハイブリッド補正(下書き → 確度判定 → LFM 書き直し)
│   ├── template.py   # チャットテンプレート管理（ネイティブ → 内蔵 → なし の3段）
│   ├── vocab.py      # 未知文字の検出（UNK / バイト断片 / 分割）
│   └── learner.py    # 未知文字の学習（即時合成 + 埋め込み勾配更新）と永続化
├── web/chat.html     # ホワイトテーマのチャット UI
└── api.py            # /api/chat (SSE) /api/model/* /api/learn /api/status など
```

## 1. 高速化のポイント

| 工夫 | 効果 |
| --- | --- |
| 動的 INT8 量子化（`torch.ao.quantization.quantize_dynamic`、Linear のみ） | 重み ~2.4GB → ~1.2GB。CPU の int8 GEMM（oneDNN/FBGEMM）で高速化 |
| 量子化しないモジュール（Embedding / norm / conv）だけ fp32 に寄せる | 動的量子化 Linear は fp32 入力を要求するための整合 |
| `TextIteratorStreamer` によるトークン単位の SSE ストリーミング | 初トークンまでの体感遅延を削減 |
| プロンプト予算（既定 1024 トークン）+ 古い履歴から削るローリング窓 | prefill コストを一定に保つ |
| 応答長の既定 128 トークン / temperature 0.3 / top_k 50 / repetition_penalty 1.05 | 日常会話に適した短い返答と安定性 |
| 予約トークン（既定 256 個）のうち未割当の id を logits で `-inf` に抑制 | 語彙拡張の副作用（空きスロットからの無意味なトークン排出）を防止 |

速度の目安: 2 vCPU・メモリ 4GB クラスで数十トークン/秒の応答が目標。
実測値は UI の各応答に `tokens_per_second` として表示される。

## 2. テンプレートがない時の生成（3 段フォールバック）

LFM2.5-1.2B-JP-202606 は `chat_template.jinja` を同梱するが、
tokenizer によってはテンプレートを持たない（または壊れている）ことがある。

1. **native**: `tokenizer.apply_chat_template()` が使えるならそのまま使う
2. **builtin**: 内蔵の ChatML テンプレート（`<|startoftext|>` + `<|im_start|>role\n...<|im_end|>`。
   LiquidAI 純正の jinja と同じワイヤ形式を素の Python で再現、jinja 依存なし）
3. **raw**: テンプレートなし生成。制御トークンを一切付けずテキストを素で続唱
   （UI の「テンプレート: なし」で強制できる。BOS のみ付与）

どの段で失敗しても次の段に落ちるため、**常に何らかの生成が可能**。

## 3. 未知文字の学習

トークナイザ（語彙 65,536）でも保持できない文字（希少漢字 `𠮷`、絵文字 `🚀`、
新語など）を、**事前学習済みパラメータを再利用して**会話に参加させる。

### 検出（vocab.py）
1 文字ずつ単独トークナイズし、次を「未知」と判定する:
- UNK トークンに落ちる
- バイトフォールバック断片（`<0xNN>` 等）に分解される
- 2 個以上のサブワードに分割される（fragmented）

### レベル 1: 即時学習（数百ミリ秒・再起動不要）
- 語彙末尾に用意した**予約スロット**（65536 番以降）に新トークンを割り当てる
- その埋め込み行を、**トークナイザがその文字を分解した既知断片の
  学習済み埋め込みの平均**で初期化する（=`compose_row`）
  - 例: `𠮷` → 4 バイト断片の埋め込みの平均 → 「吉の異体字」っぽい方向ベクトル
- untied の場合は出力側 lm_head 行にも書き込む。tied では入力側のみ
  （量子化済み lm_head は再ロード時に反映）
- `var/learned_vocab/` に JSON（文字 → id）+ safetensors（行ベクトル）として永続化

### レベル 2: 深学習（例文から勾配で数ステップ・CPU でも数十秒）
- エンコード時に**断片列 → 予約トークン**へ写像（`map_ids`）し、
  モデルが学習文字を 1 意味トークンとして読めるようにする
- チャット用の量子化モデルを一度解放し、bf16 マスターをロード
- **新トークンの行だけ**を `nn.Parameter` として切り出し、例文
  （未指定なら自動テンプレート「私は{x}が好きです。」等）で
  causal LM の損失を数ステップ（既定 12 step / AdamW / lr 3e-3）最小化
  - 損失は学習文字の近傍 3 トークン以内の位置に限定し、汎用挙動を壊さない
  - 本体の重みは完全凍結（破壊的ファインチューニングではない）
- 学習後、行を永続化して量子化モデルを再構築（tied lm_head にも学習結果が反映され、
  モデルがその文字を**出力**できるようになる）

### 永続化とライフサイクル
- `var/learned_vocab/learned_vocab.json` + `learned_rows.safetensors`
- 起動時: resize → ゼロ初期化 → 保存済み行を復元 → 量子化
- 深学習は非同期ジョブ（`POST /api/learn {"mode":"deep"}` → `GET /api/learn/status`）
- `DELETE /api/learn` で全消去

## 4. フォールバック階層（ニューラルエンジンが無い環境）

```
torch/transformers あり + モデルあり → LFM2.5-1.2B-JP（INT8）
torch なし / モデル未取得 / 読込中 / 学習中 → Snipher-mini（476 パラメータ）
```

UI のバッジに現在のエンジンが表示される。`GET /api/status` で機械可読な状態を取得。

## 5. モデルの入手について

- 既定では `LiquidAI/LFM2.5-1.2B-JP-202606` を初回起動時に HuggingFace から
  自動ダウンロードする（~2.4GB / LFM Open License v1.0）
- オフライン環境では `SNIPHER_LFM_MODEL` にローカルのモデルディレクトリを指定
- 開発/CI 用に、同アーキテクチャの小型モデル（~0.4M パラメータ）を
  `python tools/make_test_model.py --out var/tiny-lfm2` で生成でき、
  テンプレート/学習/ストリーミングの全コードパスを検証できる
  （テスト `tests/test_lfm.py` はこの小型モデルで動く）

## 6. 環境変数一覧

| 変数 | 既定 | 説明 |
| --- | --- | --- |
| `SNIPHER_LFM_MODEL` | `LiquidAI/LFM2.5-1.2B-JP-202606` | モデルID またはローカルディレクトリ |
| `SNIPHER_LFM_AUTOSTART` | `1` | 起動時にバックグラウンドロード |
| `SNIPHER_LFM_QUANTIZE` | `1` | 動的 INT8 量子化 |
| `SNIPHER_LFM_RESERVED` | `256` | 未知文字学習用の予約トークン数 |
| `SNIPHER_LFM_PROMPT_BUDGET` | `1024` | プロンプト予算（トークン） |
| `SNIPHER_LFM_MAX_NEW_TOKENS` | `128` | 既定の最大応答トークン |
| `SNIPHER_LFM_TEMPERATURE` | `0.3` | 既定温度 |
| `SNIPHER_LFM_TOP_K` | `50` | 既定 top_k |
| `SNIPHER_LFM_REPETITION_PENALTY` | `1.05` | 繰り返しペナルティ |
| `SNIPHER_LFM_LEARN_STEPS` | `12` | 深学習の既定ステップ数 |
| `SNIPHER_LFM_LEARN_LR` | `3e-3` | 深学習の学習率 |
| `SNIPHER_LFM_STORE_DIR` | `var/learned_vocab` | 学習済み語彙の保存先 |
| `SNIPHER_LFM_UPLOAD_DIR` | `var/models/upload` | ブラウザからのモデル取り込み先 |
| `SNIPHER_LFM_SKIP_NET_CHECK` | `0` | `1` で HuggingFace 事前接続確認をスキップ |
| `SNIPHER_ASSIST_ENABLED` | `1` | ハイブリッド補正の有効化 |
| `SNIPHER_ASSIST_THRESHOLD` | `0.35` | 下書き確度がこの値未満なら LFM 補正 |
| `SNIPHER_ASSIST_MAX_NEW_TOKENS` | `64` | LFM 補正の出力トークン上限 |

## 7. API

| メソッド | パス | 説明 |
| --- | --- | --- |
| GET | `/` | ホワイトテーマのチャット UI |
| GET | `/classic` | 旧 UI（超小型エンジンのデモ） |
| GET | `/api/status` | エンジン状態・学習済み文字・テンプレート情報・ハイブリッド設定 |
| POST | `/api/chat` | SSE ストリーミング応答（`hybrid:false` で LFM 直接・`use_template:false` でテンプレートなし生成） |
| GET | `/api/model/import` | ローカル取り込み状態(不足ファイルの列挙) |
| POST | `/api/model/upload` | モデルファイルのアップロード(multipart、`activate=1` で即ロード) |
| POST | `/api/model/load` | アップロード済み/指定ローカルモデルでホットリロード |
| POST | `/api/vocab/check` | テキスト中の未知文字をスキャン |
| POST | `/api/learn` | 未知文字を学習（`mode: instant` / `deep`） |
| GET | `/api/learn/status` | 深学習ジョブの状態 |
| DELETE | `/api/learn` | 学習済み語彙を全消去 |
| GET/POST | `/info` `/analyze` `/generate` | 従来どおり（超小型エンジン） |
