# Cloudflare Pages へのデプロイ

Snipher は本体（Python + FastAPI）と、Pages で動く **静的 UI + Pages Function** の 2 構成です。
Pages は Python プロセスを載せられないので、同じ設計（索引 → 証拠 → 文の組み立て）を
JS に移植した `engine.mjs` を同梱します。本体 API を生やしておくと同じ画面のまま全機能（ネット裏取り・
コード生成・会話状態）が使えるので、`SNIPHER_API_ORIGIN` を設定します。

## ビルド（リポジトリ内で完結）

```bash
python tools/build_kb.py        # snipher/data/kb.json
python tools/build_public.py    # public/ と functions/ を生成
python tools/build_public.py --check   # 成果物が最新か（CI で使う）
```

`public/` と `functions/` はビルド成果物としてリポジトリに置いてあります（Pages は
`functions/api/[[path]].js` のパスでルートを確定するため、デプロイ前に存在する必要があります）。

## Pages へのデプロイ

```bash
npm i -D wrangler
npx wrangler pages deploy public --project-name=snipher
```

ダッシュボードから繋ぐ場合の設定:

| 項目 | 値 |
| --- | --- |
| Framework preset | None |
| Build command | `python3 tools/build_public.py` |
| Build output directory | `public` |
| Functions directory | `functions`（リポジトリ置きなら自動） |
| 環境変数 | `SNIPHER_API_ORIGIN`（任意: 本体 API の URL） |

`SNIPHER_API_ORIGIN` を設定すると `/api/*` は本体へ素通しされ、落ちているときだけ
Pages 同梱の JS エンジンが答えます（画面がエラーにならない）。

## ローカルで Pages の挙動を確認する

```bash
python tools/build_public.py
node tools/preview_pages.mjs          # http://127.0.0.1:8788
# 本体 API も同時に使うとき:
SNIPHER_API_ORIGIN=http://127.0.0.1:8000 node tools/preview_pages.mjs
```

## 検証

```bash
node --test tests/js/engine.test.mjs          # JS エンジン単体（12 検査）
python -m pytest tests/test_cloudflare_pages.py
```

`tests/test_cloudflare_pages.py` は、①ビルド成果物の鮮度 ②Pages Function が UI の契約
（`/api/status` と `/api/chat` の SSE: start / delta / done）を満たすか ③SSR 的な `localhost`
直叩きをしていないか ④node があれば JS の実測、まで見ます。

## Pages 側の制限と設計判断

- **Python は動かない** — 重量モデル（`lfm/` の 731MB 級）は載せられません。Pages 版は
  `public/kb.json`（JS が読む欄だけを抜いた 367 KiB 相当）で検索し、文はその場で組み立てます。
- **1 アセット・バンドルの上限** — `kb.json` は `tools/build_public.py` が 4 MiB を超えたら
  失敗させるので、デプロイできないサイズの成果物を出しません。
- **キャッシュ** — ハッシュ付きアセットは不変、`index.html` は `no-store`（`public/_headers`）。
  古いキャッシュで画面が壊れる事故を防ぎます。
- **Vercel** — `vercel.json` は残してありますが、Pages を正とします（本体 API 側で使ってください）。
