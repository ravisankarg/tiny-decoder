import argparse
import os

import sentencepiece as spm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_txt", required=True)
    parser.add_argument("--vocab_size", type=int, default=16000)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    model_prefix = os.path.join(args.output_dir, "tokenizer")
    model_path = model_prefix + ".model"
    if os.path.exists(model_path) and not args.overwrite:
        print(f"Tokenizer already exists at {model_path}; skipping")
        print("=== DONE: build_tokenizer.py ===")
        return
    if args.overwrite:
        for path in [model_prefix + ".model", model_prefix + ".vocab"]:
            if os.path.exists(path):
                os.remove(path)

    spm.SentencePieceTrainer.train(
        input=args.corpus_txt,
        model_prefix=model_prefix,
        vocab_size=args.vocab_size,
        model_type="bpe",
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        user_defined_symbols="[OCR],[INSTR],[AUTO]",
        character_coverage=1.0,
        input_sentence_size=10000000,
        shuffle_input_sentence=True,
        hard_vocab_limit=False,
    )
    print("=== DONE: build_tokenizer.py ===")


if __name__ == "__main__":
    main()
