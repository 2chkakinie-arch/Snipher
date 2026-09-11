"""Snipher — 超小型・確率的日本語AI。

アイデア:
    動詞・形容詞・名詞・助詞・助動詞・文型を「あらかじめ決めておく」ことで、
    量子化された大規模言語モデル(1.2Bなど)と比べてパラメータ数を桁違いに減らし、
    日本語の構文解析 + 独自の確率計算式による高速な文章生成を実現する。
"""

__version__ = "0.1.0"

# LFM2.5 1.2B JP を解析して導出した独自確率式の重み(スカラー6個)。
# 詳細は snipher/probability.py の docstring を参照。
DEFAULT_WEIGHTS = {
    "alpha": 1.2,    # 頻度事前分布(コーパス由来の共起頻度)
    "beta": 1.0,     # 品詞遷移(マルコフ遷移行列)
    "gamma": 1.0,    # 格・役割適合(動詞が要求する助詞との整合)
    "delta": 0.8,    # 活用整合(活用形と後続助動詞の一致)
    "epsilon": 0.6,  # 文体整合(です/ます体 vs だ/である体)
    "zeta": 1.5,     # トピック一貫性(話題の意味クラスの連続性)
}

from .lexicon import Lexicon  # noqa: E402
from .morphology import Morphology  # noqa: E402
from .parser import Parser  # noqa: E402
from .probability import ProbabilityModel  # noqa: E402
from .generator import Generator  # noqa: E402
from .polisher import Polisher  # noqa: E402
from .responder import Responder  # noqa: E402
from .engine import SnipherEngine  # noqa: E402

__all__ = [
    "Lexicon",
    "Morphology",
    "Parser",
    "ProbabilityModel",
    "Generator",
    "Polisher",
    "Responder",
    "SnipherEngine",
    "DEFAULT_WEIGHTS",
]
