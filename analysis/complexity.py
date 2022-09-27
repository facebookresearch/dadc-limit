import nltk
import textstat
from nltk.tree import Tree
import benepar, spacy
import json
import numpy as np
nlp = spacy.load('en_core_web_md')
nlp.add_pipe(benepar.BeneparComponent("benepar_en3"))

import torch, transformers
tokenizer = transformers.AutoTokenizer.from_pretrained('../training/checkpoint-all-nli-100000')
model = transformers.AutoModelForSequenceClassification.from_pretrained('../training/checkpoint-all-nli-100000')
model = model.eval().cuda()

# helpers for computing syntactic complexity
def calc_words(t):
    if type(t) == str:
        return 1
    else:
        val = 0
        for child in t:
            val += calc_words(child)
        return val

def calc_yngve(t, par):
    if type(t) == str:
        return par
    else:
        val = 0
        for i, child in enumerate(reversed(t)):
            val += calc_yngve(child, par+i)
        return val

def compute_syntactic_complexity(sentence):
    doc = nlp(sentence)
    sent = list(doc.sents)[0]
    line = sent._.parse_string
    t = Tree.fromstring(line)
    words = calc_words(t)
    yngve = calc_yngve(t, 0)
    return round(float(yngve)/words, 2)

def compute_stats(contexts, hypotheses, labels):
    readability_scores = []
    yngve_scores = []
    sentence_lengths = []
    model_fooled = []
    for idx, (context, hypothesis, label) in enumerate(zip(contexts, hypotheses, labels)):
        readability_scores.append(textstat.flesch_kincaid_grade(hypothesis))
        yngve_scores.append(compute_syntactic_complexity(hypothesis))
        sentence_lengths.append(len(nltk.word_tokenize(hypothesis)))

        tokens = tokenizer.encode(context, hypothesis, return_tensors='pt').cuda()
        model_pred = model(tokens).logits.detach()
        if torch.argmax(model_pred).item() == 2 and label == 'contradiction' or torch.argmax(model_pred).item() == 1 and label == 'entailment':
            model_fooled.append(1)
        else:
            model_fooled.append(0)
    return readability_scores, yngve_scores, sentence_lengths, model_fooled

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

    readability_scores, yngve_scores, sentence_lengths, model_fooled = compute_stats(contexts, hypotheses, labels)
    print("Dataset", dataset)
    print('Num Examples', len(hypotheses))
    print('Readability Scores', np.mean(readability_scores))
    print('Yngve Scores', np.mean(yngve_scores))
    print('Sentence Lengths', np.mean(sentence_lengths))
    print('Fooling Rate All-NLI Model', np.mean(model_fooled))

