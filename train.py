import os
import json
import argparse
from transformers import DistilBertForSequenceClassification, TrainingArguments, Trainer
from data import prepare_datasets
from utils import compute_metrics, print_classification_report, save_results

os.environ["WANDB_DISABLED"] = "true"

MODEL_NAME = 'distilbert-base-cased'
HF_REPO = 'Tron2703/distilbert-goodreads-genre'


def train(push_to_hub=False, hf_token=None):
    print("Preparing datasets...")
    train_dataset, test_dataset, label2id, id2label, tokenizer, test_labels = prepare_datasets(MODEL_NAME)
    num_labels = len(id2label)

    print(f"Loading pre-trained model: {MODEL_NAME}")
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    training_args = TrainingArguments(
        num_train_epochs=2,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=5e-5,
        warmup_steps=50,
        weight_decay=0.01,
        output_dir='./results',
        logging_dir='./logs',
        logging_steps=100,
        eval_strategy='steps',
        save_strategy='epoch',
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()

    print("Saving model locally...")
    trainer.save_model('./distilbert-reviews-genres')
    tokenizer.save_pretrained('./distilbert-reviews-genres')

    with open('./distilbert-reviews-genres/label_mapping.json', 'w') as f:
        json.dump({'label2id': label2id, 'id2label': {str(k): v for k, v in id2label.items()}}, f)

    print("\nEvaluating model...")
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")

    predicted = trainer.predict(test_dataset)
    pred_labels = predicted.predictions.argmax(-1).flatten().tolist()
    pred_labels_str = [id2label[l] for l in pred_labels]
    report = print_classification_report(test_labels, pred_labels_str)

    save_results({
        'eval_loss': eval_results['eval_loss'],
        'eval_accuracy': eval_results['eval_accuracy'],
        'eval_f1': eval_results['eval_f1'],
        'source': 'local',
    }, 'evaluation_results_local.json')

    if push_to_hub:
        print(f"\nPushing model to Hugging Face Hub: {HF_REPO}")
        token = hf_token or os.environ.get('HF_TOKEN')
        model.push_to_hub(HF_REPO, token=token)
        tokenizer.push_to_hub(HF_REPO, token=token)
        print("Model pushed successfully!")

    return trainer, model, tokenizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--push-to-hub', action='store_true', help='Push model to HuggingFace Hub')
    parser.add_argument('--hf-token', type=str, default=None, help='HuggingFace API token')
    args = parser.parse_args()
    train(push_to_hub=args.push_to_hub, hf_token=args.hf_token)
