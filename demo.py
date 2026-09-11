#!/usr/bin/env python3
"""Snipher のデモスクリプト。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from snipher import SnipherEngine


def main() -> None:
    engine = SnipherEngine(seed=42)
    info = engine.info()
    print("=" * 62)
    print(f"Snipher v{info['version']}  ({info['architecture']})")
    print(f"対応言語: {info['language']}")
    print(f"パラメータ数: {info['total_parameters']} "
          f"(テーブル {info['table_entries']} + 確率式の重み {info['probability_weights']})")
    print(f"確率式: {info['formula']}")
    print("=" * 62)

    print("\n[解析デモ]")
    for text in ("私は猫が好きです。", "昨日は映画を見ました。", "仕事が忙しいので休みたいです。"):
        a = engine.analyze(text)
        print(f"\n  {text}")
        print(f"    文型: {a['structure']['name']}")
        print(f"    話題: {a['topic']}")
        print(f"    助動詞: {[t['surface'] for t in a['tokens'] if t['pos'] == '助動詞']}")
        print(f"    要点: {[(p['kind'], p.get('value') or p.get('role')) for p in a['key_points']]}")

    print("\n[生成デモ]")
    for kwargs in (
        {"prompt": "猫について", "register": "polite"},
        {"prompt": "朝の習慣", "register": "polite"},
        {"prompt": "旅行", "register": "casual", "tense": "past"},
        {"register": "polite", "n": 3},
    ):
        out = engine.generate(**kwargs)
        if "sentences" in out:
            for s in out["sentences"]:
                print(f"  ({s['pattern_name']}) {s['text']}")
        else:
            print(f"  ({out['pattern_name']}) {out['text']}")


if __name__ == "__main__":
    main()
