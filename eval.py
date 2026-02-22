import os
import json
import argparse
import random
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from data import load_all_genres, split_data, encode_labels, ReviewDataset
from utils import compute_metrics, print_classification_report, save_results

os.environ["WANDB_DISABLED"] = "true"

HF_REPO = 'Tron2703/distilbert-goodreads-genre'
LOCAL_MODEL_DIR = './distilbert-reviews-genres'


def evaluate(model_source='hub', hf_token=None):
    if model_source == 'hub':
        print(f"Loading model from Hugging Face Hub: {HF_REPO}")
        token = hf_token or os.environ.get('HF_TOKEN')
        model = DistilBertForSequenceClassification.from_pretrained(HF_REPO, token=token)
        tokenizer = DistilBertTokenizerFast.from_pretrained(HF_REPO, token=token)
    else:
        print(f"Loading model from local directory: {LOCAL_MODEL_DIR}")
        model = DistilBertForSequenceClassification.from_pretrained(LOCAL_MODEL_DIR)
        tokenizer = DistilBertTokenizerFast.from_pretrained(LOCAL_MODEL_DIR)

    id2label = model.config.id2label
    label2id = model.config.label2id

    print("Loading and preparing evaluation data...")
    genre_reviews = load_all_genres()
    train_texts, train_labels, test_texts, test_labels = split_data(genre_reviews)
    _, test_encoded, _, _ = encode_labels(train_labels, test_labels)

    max_length = 512
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=max_length)
    test_dataset = ReviewDataset(test_encodings, test_encoded)

    training_args = TrainingArguments(
        output_dir='./eval_results',
        per_device_eval_batch_size=16,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
    )

    print("Running evaluation...")
    eval_results = trainer.evaluate(test_dataset)
    print(f"\nEvaluation results: {eval_results}")

    predicted = trainer.predict(test_dataset)
    pred_labels = predicted.predictions.argmax(-1).flatten().tolist()
    pred_labels_str = [id2label[str(l)] if isinstance(list(id2label.keys())[0], str) else id2label[l] for l in pred_labels]
    print("\nClassification Report:")
    print_classification_report(test_labels, pred_labels_str)

    result_file = f'evaluation_results_{model_source}.json'
    save_results({
        'eval_loss': eval_results['eval_loss'],
        'eval_accuracy': eval_results['eval_accuracy'],
        'eval_f1': eval_results['eval_f1'],
        'source': model_source,
    }, result_file)

    return eval_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', choices=['hub', 'local'], default='hub',
                        help='Model source: "hub" for HuggingFace repo, "local" for local directory')
    parser.add_argument('--hf-token', type=str, default=None, help='HuggingFace API token')
    args = parser.parse_args()
    evaluate(model_source=args.source, hf_token=args.hf_token)
