# Assignment 3: End-to-End Hugging Face Model Training & Docker Deployment

## Model Selection

**Model:** `distilbert-base-cased` (DistilBERT)

DistilBERT was selected because:
- It is a distilled version of BERT, retaining 97% of BERT's language understanding while being 60% faster and 40% smaller
- The cased variant preserves capitalization information, which can be useful for genre classification of book reviews
- It is well-suited for fine-tuning on classification tasks with the Hugging Face Trainer API
- Practical for training in resource-constrained environments (Docker containers, CPU-only machines)

## Dataset

Goodreads book reviews from the [UCSD Book Graph](https://mengtingwan.github.io/data/goodreads.html), classified into 8 genres:
- poetry, children, comics & graphic, fantasy & paranormal, history & biography, mystery/thriller/crime, romance, young adult

**Data split:** 200 training + 50 test samples per genre (1600 train / 400 test total)

Note: The original notebook uses 800/200 per genre (6400/1600 total) and achieves ~59% accuracy with 3 epochs on GPU. Due to CPU-only training constraints, a smaller subset was used here. The scripts support configurable data sizes via `data.py` parameters — increase `train_size`, `sample_per_genre`, `head`, and `sample_size` for better accuracy with GPU/more time.

## Training Summary

- **Epochs:** 2
- **Batch size:** 16 (train), 16 (eval)
- **Learning rate:** 5e-5 with 50 warmup steps
- **Weight decay:** 0.01
- **Training loss:** 1.92 (epoch 1) → 1.32 (epoch 2)
- **Trainer API:** Hugging Face `Trainer` with evaluation at every 100 steps

## Evaluation Comparison

| Source | Accuracy | F1 (weighted) | Loss |
|--------|----------|---------------|------|
| DistilBERT (local) | 0.470 | 0.445 | 1.478 |
| DistilBERT (from HF Hub) | 0.518 | 0.508 | 1.363 |
| Original notebook (GPU, full data) | ~0.590 | ~0.590 | ~1.280 |

The local and hub evaluations produce slightly different metrics due to random sampling of test data each run. Both confirm the model has learned meaningful genre distinctions well above the random baseline of 12.5% (8 classes). The difference from the original notebook is due to smaller training set size and fewer epochs.

## Hugging Face Model

**Model link:** [https://huggingface.co/Tron2703/distilbert-goodreads-genre](https://huggingface.co/Tron2703/distilbert-goodreads-genre)

## Project Structure

```
├── ML_DL_Ops_Ass_3_Fine_Tuning_Classification.ipynb  # Original notebook
├── data.py              # Data loading, preprocessing, dataset class
├── utils.py             # Metrics computation and result saving
├── train.py             # Training script with HF Trainer API
├── eval.py              # Evaluation script (local or HF Hub model)
├── Dockerfile           # Training Docker image
├── Dockerfile.eval      # Production evaluation-only Docker image
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

## Docker Setup

Two Dockerfiles are provided:

### `Dockerfile` — Training Container

```bash
docker build -t goodreads-train -f Dockerfile .
docker run goodreads-train                                          # Train only
docker run -e HF_TOKEN=<your_token> goodreads-train python train.py --push-to-hub  # Train + push to HF
```

### `Dockerfile.eval` — Production Evaluation Container

Pulls the trained model from HuggingFace Hub and runs evaluation on startup.

```bash
docker build -t goodreads-eval -f Dockerfile.eval .
docker run goodreads-eval
```

### Setting HuggingFace Token in Docker

The HF token is needed to push models or access private repos. Pass it via `-e` flag:

```bash
docker run -e HF_TOKEN=hf_xxxxx goodreads-train python train.py --push-to-hub
```

You can also use `--env-file`:
```bash
echo "HF_TOKEN=hf_xxxxx" > .env
docker run --env-file .env goodreads-train python train.py --push-to-hub
```

### Running Locally (without Docker)

```bash
pip install -r requirements.txt
python train.py                                    # Train
HF_TOKEN=<token> python train.py --push-to-hub     # Train + push
python eval.py --source hub                         # Evaluate from HF Hub
python eval.py --source local                       # Evaluate local model
```

## Challenges

1. **Data download size:** The Goodreads dataset files are large; streaming with `requests` and limiting reviews per genre keeps memory usage manageable
2. **Training time on CPU:** Full training (6400 samples, 3 epochs) takes ~12 hours on CPU; reduced dataset used here to complete in ~1.5 hours. DistilBERT helps mitigate this vs full BERT
3. **Docker image size:** PyTorch makes the Docker image large (~2GB+); using `python:3.10-slim` and `--no-cache-dir` helps reduce size
4. **Accuracy vs time tradeoff:** Smaller dataset yields lower accuracy (~47-52%) vs the notebook's ~59% with full data on GPU. The workflow and code remain identical — only data size differs
