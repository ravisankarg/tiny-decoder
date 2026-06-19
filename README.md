# Tiniest Decoder

This repo is an ablation study for tiny decoder-only language models. The experiment keeps the data, tokenizer, environment, optimizer settings, context length, and effective token budget as constant as possible, then compares how different decoder sizes learn.

The main goal is not to train a large general assistant. The goal is to learn how far very small decoders can go when the data pipeline, objective, tokenizer, model size, and finetuning stages are designed carefully.

The current study compares approximately 9M, 20M, 31M, and 40M parameter decoder configs on the same base LM corpus, then tests whether a cleaner 40M-only corpus improves the strongest config. Once the base LM behavior is coherent, later iterations will add narrow task skills such as OCR-style field extraction, instruction parsing, and autocomplete.

## Ablation Study Goal

The core question is:

> With the same data and same training environment, how much does decoder size alone change learning quality?

To make that comparison meaningful, each model-size run should keep these fixed:

- same tokenizer
- same LM dataset
- same sequence length
- same cumulative tokens seen or same number of epochs
- same validation split
- same learning-rate schedule
- same effective tokens per optimizer update
- same GPU class and mixed-precision settings

Only the model config and the physical batch/accumulation split should change. The physical batch size changes because larger models need more VRAM, but gradient accumulation is adjusted to keep the effective token budget per update constant.

## Token Accounting

This repo reports **tokens seen** as the cumulative training token slots processed by the optimizer.

For example:

```text
tokens seen = training steps * effective tokens per update
```

This is different from dataset size. If a train split has 40M token slots and the model trains for 3 epochs, the training run has seen about 120M token slots. That cumulative number is the usual number to report when describing how many tokens a model was trained on.

When needed, the README distinguishes:

- **dataset size**: token slots available in the train split
- **tokens seen**: cumulative token slots processed during training
- **epochs**: how many passes through the train split produced that token budget

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

The full model-size comparison processed about 4.071B cumulative training token slots over three epochs. The follow-up `lm_40m_v2` run processed 3.457B token slots on `lm_prose_v2` and reached much lower validation loss, but the generated text is still repetitive, so the result should be treated as a better base-LM baseline rather than a finished language model.

The expanded prose corpus can stream additional public text from:

- Wikimedia English Wikipedia
- FineWeb-Edu sample text
- app-description sources already used for smartphone/application-domain prose
- MS MARCO passages and Dolly prose snippets

The wrapper for this larger corpus is:

- `scripts/prepare_lm_prose_1b.sh`

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
- ~31M params: batch 8, accumulation 8
- ~40M params: batch 4, accumulation 16

These defaults are intentionally conservative for a 4 GB RTX 3050 Laptop GPU with BF16 mixed precision and gradient checkpointing. They can be overridden with `LM_BATCH_SIZE` and `LM_GRAD_ACCUM_STEPS` when a specific GPU has more or less usable memory.

Training launchers default to `--precision bf16` because BF16 uses the same 16-bit activation memory as FP16 but has a wider exponent range and usually does not need gradient scaling. If a GPU does not support BF16, use `TRAIN_PRECISION=fp16` or `LM_PRECISION=fp16` as a fallback.

For ablation runs, `--overwrite` means a fresh run. The launcher clears the same-name checkpoint/log output before starting and does not resume `latest.pt` or `best.pt`. Without `--overwrite`, it will resume an existing same-name checkpoint when available.

## Iteration 3: 2026-06-15 Full Stage 1 Language Ablation

All four Stage 1/base-LM model configs have now been trained on the same `lm_prose` corpus. In the training metadata this run is recorded as `stage=0`, but conceptually it is the Stage 1 base-language phase: no OCR, instruction, or autocomplete finetuning has been applied.

Common training setup:

- Dataset: `lm_prose`
- Train rows: 5,309,400 packed 256-token chunks
- Validation rows: 279,934 packed 256-token chunks
- Epochs: 3
- Steps: 248,500
- Tokens/update: 16,384
- Tokens seen per model: 4,071,424,000 token slots
- Optimizer settings: `lr=0.0002`, `min_lr=0.00001`, `warmup_steps=1000`, `weight_decay=0.01`
- Precision: BF16
- Label smoothing: `0.0`
- Output dirs: `checkpoints/lm_10m`, `checkpoints/lm_20m`, `checkpoints/lm_30m`, `checkpoints/lm_40m`

![Validation loss vs tokens seen](docs/plots/lm_val_loss_vs_tokens.png)

![Validation loss vs epoch](docs/plots/lm_val_loss_vs_epoch.png)

### Model Configs And Training Results

| Experiment | Config | Layers | Hidden | Heads | FFN | Parameters | Batch | Accum | Final train loss | Final val loss | Final ppl |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `lm_10m` | `model_config_10m.json` | 6 | 256 | 4 | 1,024 | 8,835,072 | 32 | 2 | 3.2237 | 4.2208 | 68.09 |
| `lm_20m` | `model_config.json` | 8 | 384 | 6 | 1,536 | 20,340,480 | 16 | 4 | 2.8819 | 3.8677 | 47.83 |
| `lm_30m` | `model_config_30m.json` | 10 | 448 | 7 | 1,792 | 31,311,616 | 8 | 8 | 2.7437 | 3.7047 | 40.64 |
| `lm_40m` | `model_config_40m.json` | 10 | 512 | 8 | 2,048 | 39,716,864 | 4 | 16 | 2.6767 | 3.6010 | 36.63 |
| `lm_40m_v2` | `model_config_40m.json` | 10 | 512 | 8 | 2,048 | 39,716,864 | 16 | 8 | 3.1276 | 3.1331 | 22.95 |

Best checkpoint metrics:

| Experiment | Best step | Best val loss | Best ppl | Final step | Tokens/param | Train time | Tokens/sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `lm_10m` | 243,500 | 4.2178 | 67.88 | 248,500 | 460.83 | 1,687.7 min | 40,208 |
| `lm_20m` | 248,500 | 3.8677 | 47.83 | 248,500 | 200.16 | 3,165.3 min | 21,438 |
| `lm_30m` | 248,000 | 3.7042 | 40.62 | 248,500 | 130.03 | 4,930.0 min | 13,764 |
| `lm_40m` | 247,000 | 3.5988 | 36.55 | 248,500 | 102.51 | 5,978.6 min | 11,350 |
| `lm_40m_v2` | 105,000 | 3.1329 | 22.94 | 105,500 | 87.04 | 4,233.4 min | 13,610 |

### Language Capability Level

The strongest objective metric is validation loss on held-out packed prose. In the original same-corpus ablation, model size helped consistently: `lm_40m` beat `lm_30m`, `lm_20m`, and `lm_10m`. The follow-up `lm_40m_v2` run on `lm_prose_v2` is now the best objective checkpoint overall, with best validation loss `3.1329` and best perplexity `22.94`.

Qualitative fixed-seed continuation checks on 2026-06-15 showed:

- `lm_10m`: can form English sentences, but loops simple phrases and generic words. Capability level: very weak base LM.
- `lm_20m`: more sentence-like and sometimes topical, but still confuses domains and invents odd facts. Capability level: weak but usable for pipeline checks.
- `lm_30m`: better grammar and topical consistency, but still repeats app-description style and shallow clauses. Capability level: early base LM.
- `lm_40m`: best original same-corpus validation loss and most readable continuations, but still repetitive, generic, and not reliable.
- `lm_40m_v2`: much better validation loss on the cleaner `lm_prose_v2` corpus, but greedy continuations still collapse into repeated phrases. Capability level: best current base LM, still not ready for task finetuning.

Representative samples from `lm_40m`:

```text
The best place to visit in => the world. The best place to visit is where you can easily find and view more than one hundred people who can be visiting.
A smartphone camera app can => be installed by using its iPhone or iPad. A mobile phone is a device that uses an iOS device...
Machine learning is useful because => you are not only an engineer, but also a computer scientist. Features: - Easy to use, easy to use...
```

Representative samples from `lm_40m_v2` on 2026-06-19:

```text
The best place to visit in => the world is the Mediterranean Sea. The Mediterranean Sea is a great place to visit in the world...
A smartphone camera app can => be used to monitor the temperature of the air. The camera can be used to monitor the temperature of the air...
Machine learning is useful because => it is a powerful tool for learning. It is a powerful tool for learning...
```

### Learnings

- Capacity matters in this setup. The original 40M model had the best same-corpus validation loss and about 46 percent lower perplexity than the 10M model.
- Data quality matters at least as much now. The 40M v2 run cut best perplexity from `36.55` to `22.94` versus the older 40M baseline, a 36.9 percent reduction.
- The larger models are much slower, but the quality gain is real. `lm_40m` ran at about 11.3k tokens/sec versus about 40.2k tokens/sec for `lm_10m`.
- The 10M model received the most tokens per parameter, but it still had the worst validation loss. More tokens per parameter did not overcome limited capacity.
- The current dataset teaches fluent surface text but appears too repetitive and app-description-heavy. Generated text often falls into generic feature-list language.
- Validation loss alone is not enough. The 40M v2 model has the best loss, but samples still show repetition and weak semantics.

### Mistakes To Avoid Repeating

- Do not start Stage 2/Stage 3 task finetuning yet. The base LM is readable but not strong enough; task finetuning would likely overwrite or narrow it.
- Do not judge language capability only from loss. Always run a small fixed prompt set after each run.
- Do not keep scaling steps on the same data if samples plateau into the same generic style. That will mostly strengthen the dataset bias.
- Do not compare train loss across old runs unless the logging denominator and dataset are known to match. Current comparison should use validation loss and samples.

### Recommendation

Use `lm_40m_v2` as the new base-LM baseline, but do not start OCR/instruction/autocomplete finetuning yet:

1. Evaluate more prompts with greedy, top-k, and nucleus sampling to separate model repetition from decoding repetition.
2. Inspect `lm_prose_v2` for repeated boilerplate that may still be teaching phrase loops.
3. Keep 256 context for the next controlled cleanup run so it remains comparable.
4. Track both validation loss/perplexity and a fixed qualitative prompt set.
5. Only begin task finetuning after the 40M base model can continue simple prompts without obvious loops.

The practical answer: `lm_prose_v2` helped a lot objectively, but the model is still a base-LM checkpoint, not a task-ready assistant backbone.

Plots can be regenerated from local checkpoint CSVs with:

```bash
venv/bin/python scripts/plot_lm_ablation.py
```

## What Must Pass Before Task Training

Before adding OCR/instruction/autocomplete again, the base LM should produce at least somewhat coherent continuations for prompts like:

- `the best place to visit in`
- `a young boy walked into`
- `machine learning is`
- `the reason this happened was`

If the base LM still repeats tokens or produces nonsense, task training should not start.

## Next Planned Iterations

After the 2026-06-19 `lm_40m_v2` result, the next work should reduce repetition before any task-specific training:

1. Run a broader fixed-prompt eval with multiple decoding modes.
2. Clean remaining repeated boilerplate from `lm_prose_v2` if the prompt eval points back to corpus style.
3. Keep `lm_40m_v2` as the baseline checkpoint for any next corpus cleanup.
4. Add instruction/OCR/autocomplete training later with a small learning rate and a continued LM-data mix to reduce forgetting.

## Key Lesson

For tiny decoder models, training discipline matters more than adding more task data. A small model needs a clean base LM objective first. If the base distribution is weak or corrupted by padding/task formats, later finetuning will not fix it; it usually makes the model worse.
