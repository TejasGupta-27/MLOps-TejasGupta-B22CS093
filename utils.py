import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


def print_classification_report(true_labels, predicted_labels):
    report = classification_report(true_labels, predicted_labels)
    print(report)
    return report


def save_results(results, filepath='evaluation_results.json'):
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Results saved to {filepath}')
