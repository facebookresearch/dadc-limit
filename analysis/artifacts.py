import nltk
import json
from tqdm import tqdm
import numpy as np

def compute_stats(contexts, hypotheses, labels):
    high_overlap_rates_entailment = []
    high_overlap_rates_contradiction = []
    for context, hypothesis, label in tqdm(zip(contexts, hypotheses, labels)):
        context_tokens = nltk.word_tokenize(context)
        hypothesis_tokens = [h for h in nltk.word_tokenize(hypothesis) if h != '.']
        overlap_rate = sum([h in context_tokens for h in hypothesis_tokens]) / len(hypothesis_tokens)
        if overlap_rate > 0.9:
            if label == 'entailment':
                high_overlap_rates_entailment.append(1)
            elif label == 'contradiction':
                high_overlap_rates_contradiction.append(1)
            else:
                exit('label not in set')
        else:
            if label == 'contradiction':
                high_overlap_rates_contradiction.append(0)
            if label == 'entailment':
                high_overlap_rates_entailment.append(0)

    return np.array(high_overlap_rates_entailment), np.array(high_overlap_rates_contradiction)

datasets = ['../data/non-adversarial.jsonl', '../data/static-adversarial.jsonl', '../data/dynamic-adversarial.jsonl']
for dataset in datasets:
    contexts = []
    hypotheses = []
    labels = []
    for line in open(dataset,'rb'):
        myjson = json.loads(line)
        contexts.append(myjson['sentence1'])
        hypotheses.append(myjson['sentence2'].strip())
        labels.append(myjson['label'])

    high_overlap_rates_entailment, high_overlap_rates_contradiction = compute_stats(contexts, hypotheses, labels)
    print("Dataset", dataset)
    print('High Overlap Entailment Count', sum(high_overlap_rates_entailment))
    print('High Overlap Contradiction Count', sum(high_overlap_rates_contradiction))
    print('High Overlap Contradiction %', sum(high_overlap_rates_entailment) / (sum(high_overlap_rates_entailment) + sum(high_overlap_rates_contradiction)))
