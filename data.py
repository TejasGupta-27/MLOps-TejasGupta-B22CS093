import gzip
import json
import random
import requests
import torch
from transformers import DistilBertTokenizerFast

GENRE_URLS = {
    'poetry':                 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_poetry.json.gz',
    'children':               'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_children.json.gz',
    'comics_graphic':         'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_comics_graphic.json.gz',
    'fantasy_paranormal':     'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_fantasy_paranormal.json.gz',
    'history_biography':      'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_history_biography.json.gz',
    'mystery_thriller_crime': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_mystery_thriller_crime.json.gz',
    'romance':                'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_romance.json.gz',
    'young_adult':            'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_young_adult.json.gz',
}


def load_reviews(url, head=3000, sample_size=500):
    reviews = []
    count = 0
    response = requests.get(url, stream=True)
    with gzip.open(response.raw, 'rt', encoding='utf-8') as file:
        for line in file:
            d = json.loads(line)
            reviews.append(d['review_text'])
            count += 1
            if head is not None and count >= head:
                break
    return random.sample(reviews, min(sample_size, len(reviews)))


def load_all_genres():
    genre_reviews = {}
    for genre, url in GENRE_URLS.items():
        print(f'Loading reviews for genre: {genre}')
        genre_reviews[genre] = load_reviews(url, head=3000, sample_size=500)
    return genre_reviews


def split_data(genre_reviews, train_size=200, sample_per_genre=250):
    train_texts, train_labels = [], []
    test_texts, test_labels = [], []

    for genre, reviews in genre_reviews.items():
        sampled = random.sample(reviews, min(sample_per_genre, len(reviews)))
        for review in sampled[:train_size]:
            train_texts.append(review)
            train_labels.append(genre)
        for review in sampled[train_size:]:
            test_texts.append(review)
            test_labels.append(genre)

    return train_texts, train_labels, test_texts, test_labels


def encode_labels(train_labels, test_labels):
    unique_labels = sorted(set(train_labels))
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    train_encoded = [label2id[y] for y in train_labels]
    test_encoded = [label2id[y] for y in test_labels]
    return train_encoded, test_encoded, label2id, id2label


class ReviewDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def prepare_datasets(model_name='distilbert-base-cased', max_length=512):
    genre_reviews = load_all_genres()
    train_texts, train_labels, test_texts, test_labels = split_data(genre_reviews)
    train_encoded, test_encoded, label2id, id2label = encode_labels(train_labels, test_labels)

    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=max_length)

    train_dataset = ReviewDataset(train_encodings, train_encoded)
    test_dataset = ReviewDataset(test_encodings, test_encoded)

    return train_dataset, test_dataset, label2id, id2label, tokenizer, test_labels
