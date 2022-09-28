import nltk
import json
import numpy as np
from sacrebleu.metrics import BLEU
from tqdm import tqdm
from collections import defaultdict
import random

def interbleu(hypotheses, labels):
    bleu = BLEU()
    avg_score = 0.0
    total = 0.0
    random.shuffle(hypotheses)
    for hypothesis1_idx, hypothesis1 in tqdm(enumerate(hypotheses)):
        if hypothesis1_idx > 1000:
            continue
        max_score = 0.0
        best_sentence = None
        curr_label = labels[hypothesis1_idx]
        for hypothesis2_idx, hypothesis2 in enumerate(hypotheses):
            if hypothesis1_idx == hypothesis2_idx:
                continue
            if labels[hypothesis2_idx] != curr_label:
                continue
            score = bleu.corpus_score([hypothesis1], [[hypothesis2]]).score
            if score > max_score:
                max_score = score
                best_sentence = hypothesis2
        avg_score += max_score
        total += 1
    return avg_score / total

def compute_stats(contexts, hypotheses, labels):
    words = set()
    bigrams = set()
    for hypothesis, label in tqdm(zip(hypotheses, labels)):
        for word in nltk.word_tokenize(hypothesis):
            words.add(word.lower())
        for bigram in list(nltk.bigrams(list(nltk.word_tokenize(hypothesis)))):
            bigrams.add(bigram)
    interbleu_scores = interbleu(hypotheses, labels)
    contradiction_rate = np.mean(np.array([a == 'contradiction' for a in labels]))
    context_counts = defaultdict(int)
    for context in contexts:
        context_counts[context] += 1
    return words, bigrams, interbleu_scores, contradiction_rate, context_counts

datasets = ['../data/non-adversarial-with-rounds.jsonl', '../data/static-adversarial-with-rounds.jsonl', '../data/dynamic-adversarial-with-rounds.jsonl']
for dataset in datasets:
    contexts = []
    hypotheses = []
    labels = []
    for line in open(dataset,'rb'):
        myjson = json.loads(line)
        contexts.append(myjson['sentence1'])
        hypotheses.append(myjson['sentence2'].strip())
        labels.append(myjson['label'])

    words, bigrams, interbleu_scores, contradiction_rate, context_counts = compute_stats(contexts, hypotheses, labels)
    print("Dataset", dataset)
    print('Num Unique Words', len(words))
    print('Num Unique Bigrams', len(bigrams))
    print('Inter-BLEU', interbleu_scores)
    print('Contradiction %', contradiction_rate)
    print('Num Examples Per Context', context_counts)
