# Snipher Core 設計 — LFM2.5-1.2B-JP を内部構造として動かす

Liquid AI の **LFM2.5-1.2B-JP-202606**（1.17B / LFM2.5 ハイブリッドアーキテクチャ /
16 層 = 10 conv + 6 GQA / 32K context / 語彙 65,536）を、Snipher という
1 つのエンジンの**ニューラルコア**として内蔵する設計メモ。

「別製品のハイブリッド」ではなく、Snipher の内部パイプラインの一段として
LFM2.5 が組み込まれている。高速コア（従来の ~500 パラメータのテーブル +
確率式 + polisher）は、意図判定・確信度の算出・助動詞のルール補正・
フォールバック応答という**内部の役割**を担い続ける。

---

## 1. 内部パイプライン（snipher/core.py）

```
ユーザー発話
  │  <数ミリ秒>
  ▼
高速コア（常に動作）
  ├─ 未知文字の自動検出      … 入力に 𠮷/絵文字 → 自動学習をバックグラウンド起動
  ├─ 意図判定               … responses.json の対話テーブル
  ├─ 確率的な下書き + 確信度 … generator の softmax 確度 → confidence
  └─ 助動詞の補い(ルール)    … polisher（文末・助動詞・文体の修復）
  │
  ▼ 経路判定（route_of）
  ├─ intent ∈ 定形(挨拶/感謝/…) かつ confidence ≥ 閾値
  │     → ⚡ instant: そのまま即答。ニューラルコアは 1 トークンも消費しない
  ├─ それ以外（質問・雑談・低確信度）
  │     → ✨ neural: LFM2.5-1.2B-JP が会話履歴ごと本文をストリーム生成。
  │        下書きは「内部ヒント」として system プロンプトに同梱し、
  │        出力は polisher（助動詞の補い）を通して返す
  └─ ニューラルコア未準備/失敗
        → fallback: 下書き（低確信度なら安全な骨子 base_text）で必ず応答。
          自動取得の進捗を stats に載せる
```

ポイント:

- **速度維持**: 確実な定形応答はミリ秒のまま。LFM が動いても SSE ストリーミング
  なので体感遅延は小さい
- **壊れない**: ニューラル出力が空/例外でも下書きにフォールバック
- **助動詞の補い**は二重: 下書き段階（ルール）と LFM 出力の後処理（ルール）。
  さらに `POST /api/complete` では断片文を LFM に補完させる（神経系の補い）

## 2. バックエンド（自動選択）

| | gguf（既定・最速） | torch |
|---|---|---|
| ランタイム | llama-cpp-python（無ければ llama-server バイナリを PATH / var/bin / SNIPHER_LLAMA_SERVER から自動探索しサブプロセス起動） | transformers + 動的 INT8 量子化 |
| モデル | 公式 GGUF（Q4_K_M 731MB 既定。Q4_0/Q5_K_M/Q6_K/Q8_0/F16 を選択可） | model.safetensors 2.2GB + config/tokenizer 一式 |
| RAM 目安 | ~1.5GB | ~4GB（量子化時ピーク） |
| 未知文字学習 | 不要（byte-fallback で未知文字が発生しない） | 予約トークン + 埋め込み合成（即時）/ 勾配更新（深学習） |
| テンプレート | GGUF メタデータの chat template → 内蔵 ChatML → raw | tokenizer の native → 内蔵 ChatML → raw |

LFM2.5 は llama.cpp の `lfm2` アーキテクチャで動く（conv + GQA のハイブリッド）。
実測: 2 コア CPU で LFM2.5-2.6B Q4_0 が ~8-10 tok/s（ロード 1.5 秒）だったため、
1.2B-JP Q4_K_M なら一般的な PC で 20-40 tok/s 程度が見込める。

## 3. モデルの全自動取得（snipher/lfm/acquire.py）

ユーザーにアップロードや手動ダウンロードをさせない。起動時に:

1. ローカル（キャッシュ `var/models/` / `SNIPHER_LFM_MODEL` / `SNIPHER_LFM_GGUF` /
   手動取り込みディレクトリ）を確認
2. 無ければバックエンドに応じたプランでダウンロード:
   - `SNIPHER_LFM_URLS` の直接 URL
   - 共有ミラー（ギガワタスの共有ページ → HTML から直リンクを自動解決 →
     HEAD で検証してから使用。期限切れは自動スキップ）
   - HuggingFace 公式 → hf-mirror.com
3. Range リクエストでレジューム、`.part` → アトミック rename、最小サイズ検証
4. 接続レベルで失敗したホストは死亡扱いにして以降のファイルを高速スキップ
5. 失敗時は `SNIPHER_LFM_FETCH_RETRY`（既定 300 秒）ごとに自動再試行。
   UI の「再試行」/ `POST /api/model/fetch` からもトリガー可能
6. gguf の取得が失敗しても torch（逆も）に自動で切り替えて再試行する

進捗（ファイル・%・速度・ETA・ソース・ログ）は `GET /api/status` の
`acquire` と `GET /api/model/acquire` で公開し、UI のプログレスバーが描画する。
取得中も高速コアが応答を続ける。

## 4. 未知文字の学習（torch バックエンド・LFM2.5 のパラメータを適用）

- **自動**: チャット入力をスキャンし、未知文字を**応答をブロックせずに**
  バックグラウンドで即時学習する（learn_instant: トークナイザが文字を分解した
  既知断片の事前学習済み埋め込みの平均で予約トークン行を初期化）
- **深学習（任意）**: 例文から埋め込み行のみを少数ステップ勾配更新
  （本体の重みは凍結）。`var/learned_vocab/` に永続化され、再起動後も有効。
  学習済み文字はプロンプト内で 1 意味トークンに写像され、
  未割当の予約トークンは生成時にロジット抑制される
- **GGUF バックエンド**: byte-fallback BPE のため未知文字は原理的に発生しない
  （`/api/vocab/check` は unknown 0 + 説明文を返す）

## 5. テンプレートの 3 段フォールバック（snipher/lfm/template.py）

1. ネイティブ（tokenizer.chat_template / GGUF メタデータ）
2. 内蔵 ChatML（LFM2 系の <|im_start|>/<|im_end|> 形式を jinja 無しで描画）
3. テンプレートなし（raw: 役割ラベルも制御トークンも付けず素で続唱）

LFM2.5 公式のワイヤ形式:

```
<|startoftext|><|im_start|>system
…<|im_end|>
<|im_start|>user
日本の首都は？<|im_end|>
<|im_start|>assistant
```

## 6. 状態機械（boot）

```
idle → booting → fetching → loading → ready
                 │           │
                 └── failed ─┴→ (クールダウン) → 自動再試行 / deps_missing
```

- fetching/loading 中も `/api/chat` は高速コアで応答し続ける
- `POST /api/model/fetch`（UI の再試行）と `/api/model/load`（手動取り込み後の
  ホットスワップ）で状態をリセットして再 boot できる
- テスト/ホット再設定のため、環境変数（SNIPHER_LFM_MODEL 等）の変化を
  検出して自動的に設定を作り直す

## 7. UI（snipher/web/chat.html・ホワイトテーマ）

- ヘッダー: エンジンバッジ（⚡ LFM2.5-1.2B-JP (GGUF Q4_K_M · llama.cpp) 等）
- 自動取得バー: ダウンロード中は %・速度・ETA・ソースを表示（操作不要）
- 吹き出し: 送信=右（グレー）/ 応答=左（白・枠線）。応答中に
  「✨ 確率的に不安な応答 → 内部の LFM2.5 が生成」等の内部経路ノートを表示
- メタ行: 経路（⚡/✨）・助動詞補正件数・tok/s・intent・下書き確度
- 未知文字チップ: 自動学習中は「自動学習中…」、完了後は「学習済み」
- ドロワー: 設定（モード/温度/最大トークン/テンプレート）、コアの状態、
  未知文字の深学習、モデル取得（手動取り込みは「最後の手段」として折りたたみ）
