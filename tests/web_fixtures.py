"""ローカルな SERP / 記事フィクスチャ（テスト専用の mini web）。

実ネットワークに出ずに「検索 → html-fetch → 証拠文」の全経路を測るため、
Bing と同じ class 構造（b_algo / b_ad / b_ans）の HTML をたたく 1 台の
http.server を生やす。広告ブロックと chrome 文が *除外されること* を
確かめるのが主眼なので、意図的にゴミを紛れらせてある。
"""

from __future__ import annotations

import http.server
import socketserver
import threading

import pytest

ARTICLE = """<!doctype html><html><body>
<article>
<h1>GLM5.3</h1>
<p>GLM5.3 は 2026 年 4 月に公開された会話専用の基盤モデルです。</p>
<p>語彙と文法を同時に学ぶ設計で、短い指示でも手順を組み直して答えを作ります。</p>
<p>このページはデモ用の固定フィクスチャです。</p>
<nav>本サイトはリワード配布中です。サインインしてください。表現の自由についてのご案内。</nav>
</article></body></html>"""

NO_CONTENT = """<!doctype html><html><body><div id="wall">
<p>このドメインは利用できません。表示できません。</p></div></body></html>"""

SERP = """<!doctype html><html><body><ol id="b_results">
<li class="b_algo"><h2><a href="{base}/glm">GLM5.3 とは - 例示事典</a></h2>
<p>GLM5.3 は会話用の基盤モデルで、2026 年 4 月に公開されました。</p></li>
<li class="b_ad"><h2><a href="{base}/ad">GLM5.3 リワードでもらえます</a></h2>
<p>サインインしてアカウントを選択すると無料でもらえます。</p></li>
<li class="b_ans"><h2><a href="{base}/ans">検索の仕組み</a></h2>
<p>当社は表現の自由を基本的人権として重視しています。</p></li>
<li class="b_algo"><h2><a href="{base}/nothing">GLM5.3 に関する別のページ</a></h2>
<p>このページは本文がありません。</p></li>
</ol>
<footer>リワードのポイント残高: 0。Cookie の設定はここから。</footer></body></html>"""


class FixtureHandler(http.server.BaseHTTPRequestHandler):
    routes: dict = {}

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?")[0]
        got = self.routes.get(path, ("Not Found", "text/plain"))
        body, ctype = got if isinstance(got, tuple) else (got, "text/html")
        raw = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(200)
        self.send_header("Content-Type", f"{ctype}; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *a) -> None:  # テスト出力を汚さない
        return


@pytest.fixture(scope="module")
def server():
    with socketserver.TCPServer(("127.0.0.1", 0), FixtureHandler) as httpd:
        port = httpd.server_address[1]
        th = threading.Thread(target=httpd.serve_forever, daemon=True)
        th.start()
        try:
            yield f"http://127.0.0.1:{port}"
        finally:
            httpd.shutdown()
            th.join(timeout=3)


@pytest.fixture()
def engine(server):
    """フィクスチャサーバに向く ResearchEngine（ローカルなので SSRF 許可）。"""
    from snipher.research import EdgeSearchProvider, HtmlFetcher, ResearchEngine

    base = server
    FixtureHandler.routes = {
        "/search": SERP.format(base=base),
        "/glm": ARTICLE,
        "/ad": "ad page",
        "/ans": "search engine chrome",
        "/nothing": NO_CONTENT,
    }
    fetcher = HtmlFetcher(timeout=5.0, allow_private=True)
    eng = ResearchEngine(
        fetcher=fetcher,
        providers=[EdgeSearchProvider(fetcher, endpoint=base + "/search?q={query}&count={count}")],
    )
    eng.base_url = base
    return eng


@pytest.fixture()
def grounding(engine):
    """同じフィクスチャを裏取りに使う WebGrounding。"""
    from snipher.ground.web import WebGrounding

    return WebGrounding(engine, enabled=True, fetch_pages=2, limit=6)
