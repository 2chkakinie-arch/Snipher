"""LFM2 アーキテクチャの小型テストモデルを作る。

本物の LFM2.5-1.2B-JP は ~2.4GB で HuggingFace からダウンロードする必要があるが、
オフライン環境や CI ではここで作る小さな同アーキテクチャモデルで
コードパス（テンプレート / 未知文字学習 / ストリーミング生成）を検証できる。

使い方:
    python tools/make_test_model.py --out var/tiny-lfm2
"""

from __future__ import annotations

import argparse
from pathlib import Path

CORPUS = """
こんにちは。今日はいい天気ですね。
そうですね。散歩に行きましょうか。
いいですね。どこか良い場所はありますか。
公園がおすすめです。桜がきれいですよ。
それは楽しみですね。何時ごろに出かけますか。
午前十時ごろはどうですか。
了解です。楽しみにしています。
今日の昼ごはんは何にしますか。
うどんにしようと思います。あなたはどうですか。
私はお弁当を作りました。一緒に食べましょう。
ありがとうございます。とても嬉しいです。
最近は仕事が忙しいですか。
少し忙しいですが、頑張っています。
無理をしないでくださいね。体に気をつけてください。
ありがとうございます。おかげさまで元気です。
週末は何をしますか。
映画を見に行く予定です。一緒にどうですか。
ぜひお願いします。楽しみです。
このあいだ買った本は面白かったですか。
はい、とても勉強になりました。
今度貸してもらえますか。
もちろんです。明日持ってきますね。
今日は寒くなりましたね。
そうですね。暖かくして帰ってください。
また明日会いましょう。お疲れさまでした。
お疲れさまでした。気をつけて帰ってください。
""".strip().splitlines()


def build_tokenizer(vocab_size: int = 1600):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers

    specials = [
        "<unk>",            # 0
        "<|pad|>",          # 1
        "<|startoftext|>",  # 2
        "<|endoftext|>",    # 3
        "<|im_start|>",     # 4
        "<|im_end|>",       # 5
    ]
    backend = Tokenizer(models.BPE(unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    backend.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=specials,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    backend.train_from_iterator(CORPUS, trainer)
    backend.post_processor = processors.TemplateProcessing(
        single="<|startoftext|> $A",
        pair="<|startoftext|> $A $B",
        special_tokens=[("<|startoftext|>", 2)],
    )
    from transformers import PreTrainedTokenizerFast

    tok = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        pad_token="<|pad|>",
        bos_token="<|startoftext|>",
        eos_token="<|im_end|>",
    )
    return tok


def build_model(out: Path, vocab_size: int = 1600, hidden: int = 64) -> dict:
    """トークナイザ + 小型 LFM2 モデル + ネイティブテンプレート版を書き出す。"""
    import torch
    from transformers import Lfm2Config, Lfm2ForCausalLM

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    tok = build_tokenizer(vocab_size)
    real_vocab = len(tok)  # バックエンドが実際に作った語彙数
    tok.save_pretrained(out)

    # ネイティブ chat_template 同梱版（template 経路のテスト用）
    native = out.parent / (out.name + "-native")
    native.mkdir(parents=True, exist_ok=True)
    tok_native = build_tokenizer(vocab_size)
    tok_native.chat_template = (
        "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
    )
    tok_native.save_pretrained(native)

    cfg = Lfm2Config(
        vocab_size=real_vocab,
        hidden_size=hidden,
        intermediate_size=176,
        num_hidden_layers=4,
        layer_types=["conv", "conv", "full_attention", "conv"],
        num_attention_heads=4,
        num_key_value_heads=2,
        conv_L_cache=3,
        tie_word_embeddings=True,
        bos_token_id=2,
        eos_token_id=5,
        pad_token_id=1,
        max_position_embeddings=4096,
    )
    model = Lfm2ForCausalLM(cfg)
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(0.5)  # ランダム初期化の分散を少し抑える
    model.save_pretrained(out)
    return {"model_dir": str(out), "native_dir": str(native), "vocab": real_vocab}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="var/tiny-lfm2")
    ap.add_argument("--vocab-size", type=int, default=1600)
    args = ap.parse_args()
    info = build_model(Path(args.out), vocab_size=args.vocab_size)
    print("tiny LFM2 model written:", info)
