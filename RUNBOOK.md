### 1. One-time setup

```bash
git clone <your-repo>
cd <project-dir>
bash setup_env.sh
source venv/bin/activate
```

### 2. Full training walkthrough — run these commands in this exact order

```bash
# Step 1: extract corpus text for tokenizer training
# Includes natural-language captions from data/flickr/*.parquet and
# CodeX user questions only. Assistant/code answers are excluded.
python prepare_datasets.py --stage 1_text --output_dir data/ --overwrite

# Step 2: train the tokenizer
python build_tokenizer.py \
  --corpus_txt data/corpus.txt \
  --vocab_size 16000 \
  --output_dir tokenizer/ \
  --overwrite

# Step 3: tokenise Stage 1 data
python prepare_datasets.py \
  --stage 1 \
  --output_dir data/ \
  --tokenizer_path tokenizer/tokenizer.model \
  --overwrite

# Step 4: train Stage 1 (14–18 hours)
python train.py \
  --stage 1 \
  --data_dir data/ \
  --config_path model_config.json \
  --output_dir checkpoints/stage1/ \
  --fp16

# Step 5: prepare Stage 2 data
python prepare_datasets.py \
  --stage 2 \
  --output_dir data/ \
  --tokenizer_path tokenizer/tokenizer.model

# Step 6: train Stage 2 (10–14 hours)
python train.py \
  --stage 2 \
  --data_dir data/ \
  --config_path model_config.json \
  --output_dir checkpoints/stage2/ \
  --checkpoint checkpoints/stage1/best.pt \
  --fp16

# Step 7: prepare Stage 3 data
python prepare_datasets.py \
  --stage 3 \
  --output_dir data/ \
  --tokenizer_path tokenizer/tokenizer.model

# Step 8: train Stage 3 (8–12 hours)
python train.py \
  --stage 3 \
  --data_dir data/ \
  --config_path model_config.json \
  --output_dir checkpoints/stage3/ \
  --checkpoint checkpoints/stage2/best.pt \
  --tokenizer_path tokenizer/tokenizer.model \
  --fp16
```

Current local state after recovery:

```text
Tokenizer: built at tokenizer/tokenizer.model
Stage 1 corpus: 2,452,480 lines
  - CodeX user questions only: 2,185,671
  - Flickr captions: 158,915
  - MS MARCO queries: 102,023
  - ATIS utterances: 5,871
Stage 1 data: train=2,329,856 val=122,624, seq_len=256
Stage 2 data: prepared, including corrected FUNSD rows
Stage 3 data: prepared, including corrected FUNSD rows
Checkpoints: none yet
```

### 2a. Durable launch and monitor commands

Run a 5-step smoke test first:

```bash
bash scripts/start_stage1_background.sh --max_steps 5 --no_compile --overwrite
bash scripts/monitor_training.sh 1
```

If the smoke run is clean, start the real Stage 1 run:

```bash
bash scripts/start_stage1_background.sh --overwrite
```

Monitor progress:

```bash
bash scripts/monitor_training.sh 1
tail -f logs/stage1.out
tail -f checkpoints/stage1/training_log.csv
```

Resume Stage 1 after interruption:

```bash
bash scripts/start_stage1_background.sh --checkpoint checkpoints/stage1/latest.pt
```

After Stage 1 writes `checkpoints/stage1/best.pt`, run:

```bash
bash scripts/start_stage2_background.sh
bash scripts/monitor_training.sh 2
```

After Stage 2 writes `checkpoints/stage2/best.pt`, run:

```bash
bash scripts/start_stage3_background.sh
bash scripts/monitor_training.sh 3
```

### 3. Resume a stage from a checkpoint

```bash
# Example: resume Stage 2 from step 5000
python train.py \
  --stage 2 \
  --data_dir data/ \
  --config_path model_config.json \
  --output_dir checkpoints/stage2/ \
  --checkpoint checkpoints/stage2/step_5000.pt \
  --fp16
```

### 4. Evaluate a checkpoint

```bash
python evaluate.py \
  --checkpoint checkpoints/stage3/best.safetensors \
  --data_dir data/ \
  --tokenizer_path tokenizer/tokenizer.model \
  --task all
```

### 5. Export INT8 safetensors

```bash
python export_safetensors.py \
  --checkpoint checkpoints/stage3/best.pt \
  --output_path model/model.safetensors \
  --config_path model_config.json
```

### 6. Run inference demo

```bash
# OCR task
python inference_demo.py \
  --model_path model/model.safetensors \
  --tokenizer_path tokenizer/tokenizer.model \
  --task ocr \
  --input "B1GBASKET 12/01/2024 Sub T0tal 1720 GST 120 T0TAL 1840"

# INSTR task
python inference_demo.py \
  --model_path model/model.safetensors \
  --tokenizer_path tokenizer/tokenizer.model \
  --task instr \
  --input "show me photos from last december"

# AUTO task
python inference_demo.py \
  --model_path model/model.safetensors \
  --tokenizer_path tokenizer/tokenizer.model \
  --task auto \
  --input "beach sunset"
```

### 7. OOM troubleshooting (RTX 3050 4 GB VRAM)

If you see CUDA out of memory errors:

First fallback — reduce batch size and increase accumulation:

```bash
--batch_size 2 --grad_accum_steps 16
```

Second fallback (Stage 3 only) — reduce sequence length:

```bash
--max_seq_len 512
```

Mandatory — these flags must always be set:

```bash
--fp16
```

Gradient checkpointing is always ON in `train.py`.

If still OOM after both fallbacks:

Edit `model_config.json`: change `hidden_dim` from `384` to `320`. Retrain from Stage 1 with the new config.

### 8. Disk space by stage

```text
corpus.txt:              ~150 MB
Stage 1 Arrow data:      ~800 MB
Stage 2 Arrow data:      ~300 MB
Stage 3 Arrow data:    ~1,200 MB
Stage 1 checkpoints:     ~400 MB
Stage 2 checkpoints:     ~400 MB
Stage 3 checkpoints:     ~400 MB
INT8 safetensors export:  ~22 MB
Total (approximate):   ~3,800 MB (~3.8 GB)
```
