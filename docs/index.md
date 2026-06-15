# Tiniest Decoder Project Notes

This project is an ablation study for very small decoder-only language models. The goal is to learn how far tiny local decoders can go when the tokenizer, data pipeline, context length, optimizer settings, and training budget are controlled carefully.

The project is not trying to build a large general assistant. It is trying to answer a smaller engineering question:

> With the same data and same training setup, how much does decoder size change base language capability?

The current model family compares roughly 9M, 20M, 31M, and 40M parameter decoders trained from scratch on packed prose.

## Project Goal

The long-term target is a small decoder that can run locally and later support narrow skills such as:

- plain language continuation
- smartphone/application-domain text understanding
- OCR-style field extraction
- instruction parsing
- autocomplete

The current priority is base language modeling. Task-specific training is intentionally paused until the base model can produce stable, readable continuations without obvious repetition loops.

## Architecture Family

All models use the same general decoder design:

- decoder-only transformer
- SentencePiece tokenizer
- 16k vocabulary
- rotary position embeddings
- causal language modeling objective
- packed 256-token training chunks for base LM
- BF16 training where supported
- gradient checkpointing for small-GPU training

Current comparison configs:

| Experiment | Config | Layers | Hidden | Heads | FFN | Parameters |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `lm_10m` | `model_config_10m.json` | 6 | 256 | 4 | 1,024 | 8,835,072 |
| `lm_20m` | `model_config.json` | 8 | 384 | 6 | 1,536 | 20,340,480 |
| `lm_30m` | `model_config_30m.json` | 10 | 448 | 7 | 1,792 | 31,311,616 |
| `lm_40m` | `model_config_40m.json` | 10 | 512 | 8 | 2,048 | 39,716,864 |

## Iteration History

### Iteration 1: Mixed-Stage Training

The first useful run trained a roughly 20M parameter decoder through multiple stages:

- Stage 1: plain language modeling on mixed text
- Stage 2: supervised OCR/instruction data
- Stage 3: OCR/instruction/autocomplete mixture

The training completed, but the model was not useful as a language model. In interactive tests it repeated tokens, fell into subword loops, and failed to continue normal prose reliably.

Main failure causes:

- Padding token `0` was trained as a target instead of being ignored with `-100`.
- The base LM corpus was mostly short fragments rather than dense packed prose.
- Task finetuning came before the base LM was coherent.
- Stage 3 task data likely overwrote weak base-language behavior.
- Label smoothing made next-token learning less sharp for this tiny scratch-trained model.

The lesson from Iteration 1 was simple: do not add task behavior before the base language model is stable.

### Iteration 2: Base LM First

The second strategy reset the project around base LM quality:

- train only the language model first
- pack text into full token chunks
- ignore padding labels correctly
- remove label smoothing for base LM
- compare multiple model sizes under the same token budget
- delay OCR, instruction, and autocomplete training

This iteration added the controlled ablation setup and the 40M config. It also established the rule that only the model config and physical batch/accumulation split should change between ablation runs.

### Iteration 3: 2026-06-15 Stage 1 Language Ablation

All four model configs were trained on the same `lm_prose` corpus.

Common setup:

- Dataset: `lm_prose`
- Train rows: 5,309,400 packed 256-token chunks
- Validation rows: 279,934 packed 256-token chunks
- Epochs: 3
- Steps: 248,500
- Tokens/update: 16,384
- Tokens seen per model: 4,071,424,000 token slots
- Precision: BF16
- Label smoothing: `0.0`
- Learning rate: `0.0002`
- Minimum learning rate: `0.00001`
- Warmup: 1,000 steps
- Weight decay: `0.01`

## Current Results

The main objective metric is validation loss on held-out packed prose. Perplexity is computed from validation loss.

| Experiment | Final train loss | Final val loss | Final ppl | Best val loss | Best ppl | Tokens/sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `lm_10m` | 3.2237 | 4.2208 | 68.09 | 4.2178 | 67.88 | 40,208 |
| `lm_20m` | 2.8819 | 3.8677 | 47.83 | 3.8677 | 47.83 | 21,438 |
| `lm_30m` | 2.7437 | 3.7047 | 40.64 | 3.7042 | 40.62 | 13,764 |
| `lm_40m` | 2.6767 | 3.6010 | 36.63 | 3.5988 | 36.55 | 11,350 |

![Validation loss vs tokens seen](plots/lm_val_loss_vs_tokens.png)

![Validation loss vs epoch](plots/lm_val_loss_vs_epoch.png)

## Where We Stand

The ablation is working. Larger models consistently produce better validation loss under the same training setup.

Current quality ranking:

1. `lm_40m`: best objective metrics and most readable samples.
2. `lm_30m`: second-best, noticeably better than 20M.
3. `lm_20m`: usable for pipeline checks but weak as a language model.
4. `lm_10m`: can form English, but repeats heavily and lacks capacity.

The best current model is `lm_40m`, but it is not ready for task finetuning. It can produce readable continuations, yet the text is still generic, shallow, repetitive, and biased toward app-description style.

Representative `lm_40m` behavior:

```text
The best place to visit in => the world. The best place to visit is where you can easily find and view more than one hundred people who can be visiting.
A smartphone camera app can => be installed by using its iPhone or iPad. A mobile phone is a device that uses an iOS device...
Machine learning is useful because => you are not only an engineer, but also a computer scientist. Features: - Easy to use, easy to use...
```

## Key Learnings

Capacity matters. The 40M model has the best validation loss and about 46 percent lower final perplexity than the 10M model.

Tokens per parameter are not enough by themselves. The 10M model saw the most tokens per parameter but still had the worst validation loss.

Validation loss is necessary but not sufficient. The 40M model is objectively best, but qualitative samples show that the language capability is still early.

Data quality is now the bottleneck. More epochs on the same `lm_prose` data will likely strengthen the same generic app-description bias.

Task finetuning should wait. OCR, instruction parsing, and autocomplete should not resume until the base LM can continue simple prose without loops.

## Mistakes To Avoid

- Do not train padding as a target.
- Do not train on mostly short, padded fragments for base LM.
- Do not mix task finetuning into a weak base LM.
- Do not judge a model only by training loss.
- Do not continue scaling the same data if qualitative samples have plateaued into repetitive style.
- Do not compare experiments unless tokenizer, data, context length, tokens/update, and validation split are controlled.

## Recommended Next Step

Continue with the 40M config, but train it on cleaner and broader data.

The next useful experiment should be:

1. Build `lm_prose_v2`.
2. Reduce repeated app-store style text.
3. Add broader high-quality prose.
4. Keep packed 256-token chunks for comparability.
5. Train `lm_40m` first.
6. Compare against the 2026-06-15 `lm_40m` baseline using validation loss, perplexity, and fixed prompt samples.
7. Only run 20M/30M controls after the 40M v2 result proves the data improved language behavior.

The current project state is a good controlled baseline, not a finished decoder. The next improvement should come from corpus quality and balance before architecture changes or task finetuning.
