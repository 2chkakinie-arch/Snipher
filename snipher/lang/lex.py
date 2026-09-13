"""語彙バンク（wordbank）— 12 万語の「言葉の在庫」とその索引。

`tools/build_wordbank.py` が Janome のシステム辞書（Apache-2.0 / IPADIC 系）から作って
同梱する。無い環境でも同梱の小さい語彙テーブルで動く（graceful degradation）。

この一层で、Snipher は次の仕事を *タスク専用の分岐なし* にこなせる:

* 表記 → 読み・品詞・辞書形の照会（「三毛猫の読みは?」）
* 読みでの前方一致／後方一致／含む／同音／韻（しりとり・ケツのカ行・文末合わせ）
* 拍数の正確な数え上げ（「5 音の名词を 3 つ」）
* 未知の文の分かち書き（最長一致 + 成本）と内容語の抽出
"""

from __future__ import annotations

import bisect
import gzip
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .phonetics import chain_key, kana_to_ro, mora_count, normalize, to_hiragana

_KANA_ONLY = re.compile(r"[ぁ-んァ-ヶー]+$")

DATA = Path(__file__).resolve().parents[1] / "data"

_CONTENT_POS = ("名詞", "動詞", "形容詞", "副詞", "接頭詞", "連体詞")
_NOUN_POS = ("名詞",)


def _rank_word(w: "Word") -> tuple[int, int, int]:
    """一覧の並び順: 和語・漢字語をカタカナ語より前に、次に頻出度、長さ。"""
    kana = w.kana()
    kata = all("\u30a1" <= c <= "\u30f6" or c in "ー" for c in kana) if kana else True
    score = 1 if kata else 0
    return (score, w.cost, len(w.surface))


@dataclass(frozen=True)
class Word:
    """語彙バンクの 1 エントリ。"""

    surface: str
    reading: str          # ひらがな読み（無い場合は表記から推定）
    pos: str              # 品詞（名詞/名詞/普通名詞 のように細目つき）
    dictform: str = ""    # 辞書形・活用語彙では基本形
    cost: int = 6000      # 小さいほど頻出（成本）

    @property
    def pos_major(self) -> str:
        return self.pos.split("/")[0] if self.pos else ""

    @property
    def is_noun(self) -> bool:
        return self.pos_major in _NOUN_POS

    @property
    def is_content(self) -> bool:
        return self.pos_major in _CONTENT_POS

    @property
    def morae(self) -> int:
        return mora_count(self.reading or self.surface)

    def kana(self) -> str:
        return to_hiragana(self.reading or self.surface)

    def romaji(self) -> str:
        return kana_to_ro(self.kana())

    def as_dict(self) -> dict:
        return {"surface": self.surface, "reading": self.reading, "pos": self.pos,
                "dictform": self.dictform, "morae": self.morae, "romaji": self.romaji()}


class WordBank:
    """読み込みは 1 回だけ。索引（前方一致用）は辞書読み込み時に作る。"""

    _shared: "WordBank | None" = None

    def __init__(self, path: str | Path | None = None):
        self._by_surface: dict[str, Word] = {}
        self._readings: list[tuple[str, str]] = []      # (読み, 表記) 読み順
        self._game: list[tuple[str, str, str]] = []      # (読み, 表記, 品詞) 遊び対象
        self._suffix_index: list[tuple[str, str]] | None = None
        self.source = ""
        self.loaded_from: str | None = None
        self._load(Path(path) if path else DATA / "wordbank.bin.gz")
        self._load_game(DATA / "wordbank_game.bin.gz")

    # ------------------------------------------------------------------ #
    @classmethod
    def shared(cls) -> "WordBank":
        if cls._shared is None:
            cls._shared = WordBank()
        return cls._shared

    @classmethod
    def reset(cls) -> None:
        cls._shared = None

    # ------------------------------------------------------------------ #
    def _load(self, path: Path) -> None:
        if path.exists():
            try:
                with gzip.open(path, "rt", encoding="utf-8") as fh:
                    for line in fh:
                        parts = line.rstrip("\n").split("\t")
                        if len(parts) < 4:
                            continue
                        surface, reading, pos, dictform = parts[0], parts[1], parts[2], parts[3]
                        cost = int(parts[4]) if len(parts) > 4 and parts[4].isdigit() else 6000
                        w = Word(surface, reading, pos, dictform, cost)
                        if surface not in self._by_surface:
                            self._by_surface[surface] = w
                if self._by_surface:
                    self.loaded_from = str(path)
                    self.source = ("Janome system dictionary (Apache-2.0; dict data: IPADIC)"
                                   if path.parent.name == "data" else "custom")
            except Exception:  # noqa: BLE001
                self._by_surface = {}
        if not self._by_surface:
            self._load_fallback()
        # 読み順の索引（前方一致用）
        self._readings = sorted(
            ((w.reading or w.surface, w.surface) for w in self._by_surface.values()),
            key=lambda t: (t[0], len(t[0])))

    def _load_fallback(self) -> None:
        """wordbank が無い環境用の最小在庫（同梱の語彙テーブルから作る）。"""
        for fname, pos in (("nouns.json", "名詞"), ("verbs.json", "動詞"),
                           ("adjectives.json", "形容詞")):
            p = DATA / fname
            if not p.exists():
                continue
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            for key in (data if isinstance(data, dict) else []):
                if not isinstance(key, str) or not key:
                    continue
                self._by_surface.setdefault(key, Word(key, to_hiragana(key), pos, key, 5000))
        self.source = "bundled lexicon tables (fallback)"

    def _load_game(self, path: Path) -> None:
        if path.exists():
            try:
                with gzip.open(path, "rt", encoding="utf-8") as fh:
                    for line in fh:
                        parts = line.rstrip("\n").split("\t")
                        if len(parts) >= 2:
                            self._game.append((parts[0], parts[1], parts[2] if len(parts) > 2 else ""))
            except Exception:  # noqa: BLE001
                self._game = []
        if not self._game:
            self._game = sorted(
                ((w.reading or w.surface, w.surface, w.pos) for w in self._by_surface.values()
                 if w.is_noun and (w.reading or "").strip() and not w.reading.endswith(("ん", "ー"))
                 and all("ぁ" <= c <= "ん" for c in (w.reading or ""))),
                key=lambda t: (t[0], len(t[0])))

    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self._by_surface)

    def stats(self) -> dict:
        return {"words": len(self._by_surface), "indexed_readings": len(self._readings),
                "game_words": len(self._game), "source": self.source,
                "params_estimate": len(self._by_surface) * 4 + len(self._game)}

    def entry(self, surface: str) -> Word | None:
        if not surface:
            return None
        s = normalize(surface)
        w = self._by_surface.get(s) or self._by_surface.get(s.lower())
        if w is None and len(s) > 1:
            w = self._by_surface.get(to_hiragana(s))
        return w

    def has(self, surface: str) -> bool:
        return self.entry(surface) is not None

    def reading(self, surface: str) -> str:
        w = self.entry(surface)
        return w.reading if w else to_hiragana(normalize(surface))

    def pos(self, surface: str) -> str:
        w = self.entry(surface)
        return w.pos if w else ""

    # ------------------------------------------------------------------ #
    # 索引クエリ（一般の言葉遊び・辞書照会・部分一致を同じ実装で支える）
    # ------------------------------------------------------------------ #
    def _iter_readings(self, prefix: str) -> list[tuple[str, str]]:
        if not prefix:
            return []
        lo = bisect.bisect_left(self._readings, (prefix, ""))
        out: list[tuple[str, str]] = []
        for i in range(lo, len(self._readings)):
            r, s = self._readings[i]
            if not r.startswith(prefix):
                break
            out.append((r, s))
        return out

    def by_reading_prefix(self, prefix: str, *, limit: int = 12,
                          pos: tuple[str, ...] | None = None,
                          max_len: int = 10, min_len: int = 1) -> list[Word]:
        prefix = to_hiragana(normalize(prefix))
        out: list[Word] = []
        for _r, surface in self._iter_readings(prefix):
            w = self._by_surface.get(surface)
            if w is None:
                continue
            ln = len(w.kana())
            if ln < min_len or ln > max_len:
                continue
            if pos and w.pos_major not in pos:
                continue
            out.append(w)
            if len(out) >= limit * 3:
                break
        out.sort(key=_rank_word)
        return out[:limit]

    def suffix_index(self) -> list[tuple[str, str]]:
        """後方一致・韻に使います。初回のみ構築。"""
        if self._suffix_index is None:
            self._suffix_index = sorted(
                ((w.kana()[::-1], w.surface) for w in self._by_surface.values()
                 if w.reading), key=lambda t: (t[0], len(t[0])))
        return self._suffix_index

    def by_reading_suffix(self, suffix: str, *, limit: int = 12,
                          pos: tuple[str, ...] | None = None) -> list[Word]:
        s = to_hiragana(normalize(suffix))[::-1]
        idx = self.suffix_index()
        lo = bisect.bisect_left(idx, (s, ""))
        out: list[Word] = []
        for i in range(lo, len(idx)):
            rev, surface = idx[i]
            if not rev.startswith(s):
                break
            w = self._by_surface.get(surface)
            if w is None or (pos and w.pos_major not in pos):
                continue
            out.append(w)
            if len(out) >= limit * 3:
                break
        out.sort(key=_rank_word)
        return out[:limit]

    def containing(self, sub: str, *, limit: int = 20, pos: tuple[str, ...] | None = None) -> list[Word]:
        sub = to_hiragana(normalize(sub))
        if len(sub) < 1:
            return []
        out: list[Word] = []
        for w in self._by_surface.values():
            if not w.reading:
                continue
            if sub in w.kana() or sub in w.surface:
                if pos and w.pos_major not in pos:
                    continue
                out.append(w)
                if len(out) >= limit * 4:
                    break
        out.sort(key=_rank_word)
        return out[:limit]

    def by_morae(self, n: int, *, pos: tuple[str, ...] = ("名詞",), limit: int = 20) -> list[Word]:
        """拍数で探す（「5 音の名前」のような指定）。game 索引を使うので速い。"""
        out: list[Word] = []
        for reading, surface, p in self._game:
            if pos and p.split("/")[0] not in pos:
                continue
            if mora_count(reading) == n:
                w = self._by_surface.get(surface)
                if w is not None:
                    out.append(w)
                    if len(out) >= limit * 5:
                        break
        out.sort(key=_rank_word)
        return out[:limit]

    # ------------------------------------------------------------------ #
    # 連鎖（しりとり系）: 「前の語の送り音 → 次の語」を一般的な索引クエリとして実装
    # ------------------------------------------------------------------ #
    def chain_candidates(self, prev: str, *, limit: int = 24,
                         exclude: tuple[str, ...] = (), forbid_ends: tuple[str, ...] = ("ん",),
                         allow_small_kana: bool = False, pos: tuple[str, ...] = ("名詞",),
                         max_len: int = 8) -> list[Word]:
        key = chain_key(prev)
        if not key:
            return []
        heads = [key]
        if len(key) == 2:                       # 拗音（しゃ → しゃ / し の両方で受ける）
            heads.append(key[0])
        if key in "がぎぐげござじずぜぞだぢづでどばびぶべぼぱぴぷぺぽ":
            heads.append({"が": "か", "ぎ": "き", "ぐ": "く", "げ": "け", "ご": "こ",
                          "ざ": "さ", "じ": "し", "ず": "す", "ぜ": "せ", "ぞ": "そ",
                          "だ": "た", "ぢ": "ち", "づ": "つ", "で": "て", "ど": "と",
                          "ば": "は", "び": "ひ", "ぶ": "ふ", "べ": "へ", "ぼ": "ほ",
                          "ぱ": "は", "ぴ": "ひ", "ぷ": "ふ", "ぺ": "へ", "ぽ": "ほ"}[key])
        out: list[Word] = []
        seen: set[str] = set()
        for head in heads:
            for reading, surface, p in self._game:
                if not reading.startswith(head):
                    continue
                if surface in exclude or surface in seen:
                    continue
                if p.split("/")[0] not in pos:
                    continue
                if any(reading.endswith(x) for x in forbid_ends):
                    continue
                if len(reading) > max_len:
                    continue
                if not allow_small_kana and re.search(r"[ぁぃぅぇぉゃゅょ]", reading[2:]):
                    continue
                w = self._by_surface.get(surface)
                if w is None:
                    continue
                seen.add(surface)
                out.append(w)
                if len(out) >= limit:
                    break
            if out:
                break
        return out

    # ------------------------------------------------------------------ #
    # 分かち書き（最長一致 + 成本）
    # ------------------------------------------------------------------ #
    def segment(self, text: str, *, max_word: int = 12) -> list[tuple[str, str]]:
        """[(表層形, 品詞)] を返す。辞書に無い語は 1 文字ずつ unknown として出す。"""
        t = normalize(text)
        out: list[tuple[str, str]] = []
        i, n = 0, len(t)
        while i < n:
            if t[i].isspace():
                i += 1
                continue
            if re.match(r"[0-9０-９.]+(?:%|割)?", t[i:]):
                m = re.match(r"[0-9０-９.]+(?:%|割)?", t[i:])
                assert m
                out.append((normalize(m.group()), "数詞"))
                i += len(m.group())
                continue
            if re.match(r"[A-Za-z][A-Za-z0-9_+#.\-]*", t[i:]):
                m = re.match(r"[A-Za-z][A-Za-z0-9_+#.\-]*", t[i:])
                assert m
                out.append((m.group(), "補助記号" if len(m.group()) <= 1 else "名詞"))
                i += len(m.group())
                continue
            best: tuple[int, int, str] | None = None      # (語の終わり, cost, pos)
            # 1 文字語（猫・魚・火・が・を）も拾うため、下界は i（j=i+1 まで試す）。
            for j in range(min(n, i + max_word), i, -1):
                cand = t[i:j]
                w = self._by_surface.get(cand)
                if w is not None:
                    if best is None or j > best[0] or (j == best[0] and w.cost < best[1]):
                        best = (j, w.cost, w.pos)
            if best:
                out.append((t[i:best[0]], best[2]))
                i = best[0]
            else:
                ch = t[i]
                cls = "記号" if not re.match(r"[\wぁ-んァ-ヶ一-龯]", ch) else "未知語"
                out.append((ch, cls))
                i += 1
        return self._absorb_inflections(out)

    def _absorb_inflections(self, out: list[tuple[str, str]]) -> list[tuple[str, str]]:
        """辞書に見出しの無い活用形（「食」+「べた」）を、活用エンジンで 1 語に戻す。

        分かち書きの質は「誰が何をした」の抽出と、文体ぞろえの両方に効く。
        未知語のかたまりを後ろに 1〜3 個つないで見て、辞書形に引ける語なら
        その品詞で 1 語として数える。
        """
        if not out:
            return out
        try:
            from .morph import lemma_of
        except Exception:  # noqa: BLE001
            return out
        merged: list[tuple[str, str]] = []
        i, n = 0, len(out)
        while i < n:
            surf, pos = out[i]
            major = str(pos).split("/")[0]
            unknown = not pos or pos.startswith("未知語")
            # 漢字 1-2 字 + かな続き（食+べた、走+った）は活用の途中取り違えやすい。
            # 「辞書形に引ける & 動詞・形容詞に落ちる」場合だけ接収するので、
            # 名詞（猫・魚）を壊すことはない。
            next_is_kana = (i + 1 < n and bool(_KANA_ONLY.match(out[i + 1][0] or "")))
            if not unknown and not (major == "名詞" and len(surf) <= 2 and next_is_kana):
                merged.append((surf, pos))
                i += 1
                continue
            hit: tuple[str, str] | None = None
            span = 1
            for take in (3, 2, 1):        # 長い取り方を優先（食べた＝食べる+た）
                if i + take > n:
                    continue
                cand = "".join(s for s, _p in out[i:i + take])
                if len(cand) < 2 or len(cand) > 12 or cand == surf:
                    continue
                try:
                    lem = lemma_of(cand)
                except Exception:  # noqa: BLE001
                    lem = ""
                w = self.entry(lem) if lem else None
                if w is not None and str(w.pos).split("/")[0] in ("動詞", "形容詞"):
                    # 辞書形に引けるだけでは足りない。その辞書形を実際に活用させて
                    # 目の前の表記が再現できなければ、それはこの語の活用形ではない
                    # （「いんて」→「いむ」テ形「いんで」✗ のような見立てを弾く）。
                    from .morph import fits_inflection

                    if fits_inflection(lem, cand):
                        hit, span = (cand, w.pos), take
                        break
            if hit:
                merged.append(hit)
                i += span
            else:
                merged.append((surf, pos))
                i += 1
        return merged

    def content_words(self, text: str) -> list[Word]:
        """発話から内容語（名詞・動詞・形容詞・副詞）を引く。未知語は長さで拾う。"""
        out: list[Word] = []
        for surface, pos in self.segment(text):
            major = pos.split("/")[0]
            if major in _CONTENT_POS or major in ("数詞", "未知語"):
                w = self._by_surface.get(surface)
                if w is None:
                    w = Word(surface, to_hiragana(surface), pos or "未知語", surface, 9000)
                if w.surface not in {x.surface for x in out}:
                    out.append(w)
        return out

    def known(self, text: str) -> bool:
        """text 全体（または読み）が語彙にあるか。未知語判定に使う。"""
        s = normalize(text)
        if s in self._by_surface:
            return True
        kana = to_hiragana(s)
        if kana in self._by_surface:
            return True
        return any(w.surface == s or w.reading == kana for w in self._by_surface.values()
                   if len(s) <= 3)


_EN: tuple[str, ...] | None = None


def english_words() -> tuple[str, ...]:
    """`tools/build_words_en.py` が作る欧文語彙（遅延ロード・無ければ空）。"""
    global _EN
    if _EN is None:
        path = DATA / "words_en.txt.gz"
        words: list[str] = []
        if path.exists():
            try:
                with gzip.open(path, "rt", encoding="utf-8") as fh:
                    words = [ln.strip() for ln in fh if ln.strip()]
            except Exception:  # noqa: BLE001
                words = []
        _EN = tuple(words)
    return _EN


def _bisect_word(words: tuple[str, ...], prefix: str) -> int:
    lo, hi = 0, len(words)
    while lo < hi:
        mid = (lo + hi) // 2
        if words[mid] < prefix:
            lo = mid + 1
        else:
            hi = mid
    return lo


def english_by_length(n: int, *, limit: int = 12, startswith: str = "", endswith: str = "",
                      letters: str = "", banned: str = "") -> list[str]:
    """欧文の語彙クエリ（長さ・頭/尾の綴り・使える文字・含めてはいけない文字）。"""
    pool = english_words()
    if not pool or n <= 0:
        return []
    startswith, endswith = startswith.lower(), endswith.lower()
    allowed = set(letters.lower()) if letters else None
    forbid = set(banned.lower())
    out: list[str] = []
    for w in pool[_bisect_word(pool, startswith):] if startswith else pool:
        if len(w) < n:
            continue
        if len(w) > n:
            if startswith:
                break
            continue
        if startswith and not w.startswith(startswith):
            continue
        if endswith and not w.endswith(endswith):
            continue
        if allowed is not None and any(c not in allowed for c in w if c.isalpha()):
            continue
        if forbid & set(w):
            continue
        out.append(w)
        if len(out) >= limit:
            break
    return out


__all__ = ["Word", "WordBank", "bank", "lookup", "english_words", "english_by_length"]


_BANK: WordBank | None = None


def bank() -> WordBank:
    global _BANK
    if _BANK is None:
        _BANK = WordBank.shared()
    return _BANK


@lru_cache(maxsize=4096)
def lookup(surface: str) -> dict | None:
    w = bank().entry(surface)
    return w.as_dict() if w else None



