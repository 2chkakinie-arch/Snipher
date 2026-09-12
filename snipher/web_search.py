"""後方互換の公開入口。

ウェブ検索機能の実装は :mod:`snipher.research` に集約しています。短い名前を
使いたい利用者や既存の統合コード向けに、Edge HTML 検索と html-fetch を
ここからも import できます。
"""

from .research import (
    DuckDuckGoProvider,
    EdgeScraper,
    EdgeSearchProvider,
    FetchResult,
    HtmlFetch,
    HtmlFetcher,
    ResearchEngine,
    ResearchPolicy,
    ResearchResult,
    SearchResult,
    edge_search,
    html_fetch,
    validate_url,
)

__all__ = [
    "DuckDuckGoProvider", "EdgeScraper", "EdgeSearchProvider", "FetchResult",
    "HtmlFetch", "HtmlFetcher", "ResearchEngine", "ResearchPolicy", "ResearchResult",
    "SearchResult", "edge_search", "html_fetch", "validate_url",
]
