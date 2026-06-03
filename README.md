# Tiniest Decoder

This repo is an experiment to build the smallest useful decoder-only language model that can run locally on a small GPU, then gradually add narrow task skills such as OCR-style field extraction, instruction parsing, and autocomplete.

The main goal is not to train a large general assistant. The goal is to learn how far a very small decoder can go when the data pipeline, objective, tokenizer, model size, and finetuning stages are designed carefully.

## Current Direction

We are using an iterative approach:

1. Build a tiny decoder and train a base language model.
2. Test whether it can do basic next-token continuation.
3. Only after the base LM is coherent, add task-specific training.
4. Keep technical notes from each iteration so future runs do not repeat the same mistakes.

The current reset is focused on base LM quality first. OCR, instruction parsing, and autocomplete will come later after the plain language model behaves sensibly.

The main experiment design is to keep the dataset and training settings constant, then compare how different tiny parameter counts learn across the same epochs, steps, and token budget. Each run writes metadata and CSV metrics so model-size comparisons are explicit.

## Iteration 1: Failed Mixed-Stage Training

The first iteration trained a small decoder with about 20M parameters:

- 8 layers
- 384 hidden size
- 6 attention heads
- 16k SentencePiece vocabulary
- About 20.3M parameters

Training was split into three stages:

- Stage 1: plain language modeling on a mixed corpus
- Stage 2: supervised OCR/instruction datasets
- Stage 3: OCR/instruction/autocomplete mixture

The model trained without crashing, but the final checkpoint was not useful as a language model. In Gradio, plain continuation often repeated the last token or fell into subword loops such as repeated fragments. Sampling helped slightly, but did not fix the underlying model quality.

## Why Iteration 1 Failed

### Padding Was Trained As A Target

Stage 1 padded short examples with token `0`, but labels also used token `0`. That means the loss taught the model to predict `<pad>` after many short examples.

For language modeling, padding positions should be ignored with label `-100`, so they do not contribute to cross-entropy loss.

Further reading:

- PyTorch `CrossEntropyLoss` and `ignore_index`: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html

### The Base LM Data Was Too Fragmented

Stage 1 mostly used short independent lines: user questions, search queries, captions, and OCR-like text. Most rows were much shorter than the 256-token training length.

This made the model see many tiny fragments instead of dense, continuous language. A decoder-only LM learns better when text is packed into full token chunks.

Further reading:

- Hugging Face causal language modeling guide: https://huggingface.co/docs/transformers/tasks/language_modeling
- Token packing discussion in language modeling: https://huggingface.co/docs/trl/sft_trainer#packing

### Stage 3 Overwrote Base LM Behavior

Stage 3 was dominated by short task rows, especially MS MARCO next-word autocomplete examples. That objective is not the same as broad language modeling.

A tiny model can forget base language behavior quickly if finetuning is mostly task-format outputs. This is a form of catastrophic forgetting.

Further reading:

- Catastrophic forgetting overview: https://en.wikipedia.org/wiki/Catastrophic_interference

### Label Smoothing Was Not Helpful Here

The first run used label smoothing during LM training. For a very small decoder learning from scratch, this can make next-token learning less sharp. Iteration 2 uses `label_smoothing=0.0` for base LM.

Further reading:

- Label smoothing paper: https://arxiv.org/abs/1512.00567

### The Model Was Probably Too Small For The Data Mix

20M parameters may be enough for narrow autocomplete if the data is clean, but it was too small for a noisy mixture of questions, captions, OCR text, and supervised task formats. Iteration 2 increases capacity while still targeting a small local GPU.

## Iteration 2: Base LM First

Iteration 2 changes the strategy:

- Train only a base language model first.
- Do not train OCR, instruction, or autocomplete tasks yet.
- Use packed token chunks instead of short padded examples.
- Ignore padding labels correctly.
- Increase the model from about 20M to about 40M parameters.
- Test plain continuation before any task finetuning.

New model config:

- 10 layers
- 512 hidden size
- 8 attention heads
- 2048 FFN size
- 16k vocabulary
- 256 token context for base LM
- About 39.7M parameters

Additional comparison configs are included for smaller runs:

- `model_config_10m.json`: about 8.8M parameters
- `model_config.json`: about 20.3M parameters
- `model_config_30m.json`: about 31.3M parameters
- `model_config_40m.json`: about 39.7M parameters

New files/scripts:

- `model_config_40m.json`
- `data/lm_base/`
- `scripts/prepare_lm_base.sh`
- `scripts/start_lm_base_background.sh`
- `scripts/start_lm_experiment_background.sh`
- `scripts/compare_lm_experiments.py`

The packed LM dataset is built from `data/corpus.txt` into:

- `data/lm_base/train`
- `data/lm_base/val`

The improved prose LM path also adds app-description sources so the base model sees smartphone and application-domain language before task finetuning. Current public sources include Google Play app metadata from MobileRec and Mac App Store description text. These are used as prose documents, not as scraped live store pages.

Training output goes to:

- `checkpoints/<experiment_name>/`
- `logs/<experiment_name>.out`

Each experiment stores:

- `metadata.json`: parameter count, dataset fingerprint, batch settings, learning rate, token budget, and notes
- `training_log.csv`: step, epoch, tokens seen, train loss, validation loss, and learning rate
- checkpoints and exported safetensors for the best model

The LM experiment launcher chooses physical batch size from model size so smaller models use more of the available GPU memory. Gradient accumulation is adjusted to keep the effective tokens per update close across runs:

- ~9M params: batch 32, accumulation 2
- ~20M params: batch 16, accumulation 4
- ~31M params: batch 10, accumulation 6
- ~40M params: batch 8, accumulation 8

These defaults target a 4 GB laptop GPU with FP16 and gradient checkpointing. They can be overridden with `LM_BATCH_SIZE` and `LM_GRAD_ACCUM_STEPS` when a specific GPU has more or less usable memory.

## What Must Pass Before Task Training

Before adding OCR/instruction/autocomplete again, the base LM should produce at least somewhat coherent continuations for prompts like:

- `the best place to visit in`
- `a young boy walked into`
- `machine learning is`
- `the reason this happened was`

If the base LM still repeats tokens or produces nonsense, task training should not start.

## Next Planned Iterations

After Iteration 2 produces a usable base LM:

1. Add instruction/OCR training with a small learning rate.
2. Mix LM data during task finetuning to reduce forgetting.
3. Keep task ratios explicit, for example mostly LM with smaller task batches.
4. Evaluate plain LM quality after each task stage.
5. Only keep checkpoints that preserve base continuation quality.

## Key Lesson

For tiny decoder models, training discipline matters more than adding more task data. A small model needs a clean base LM objective first. If the base distribution is weak or corrupted by padding/task formats, later finetuning will not fix it; it usually makes the model worse.
