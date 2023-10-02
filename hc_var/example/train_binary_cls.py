"""Training code for the detector model"""

import argparse
import torch
from torch import nn
from torch.optim import AdamW
import transformers
from transformers import tokenization_utils, RobertaTokenizer, RobertaForSequenceClassification
transformers.logging.set_verbosity_error()
import numpy as np
from sklearn.metrics import f1_score
from detector.data_loader import Loader, Domain_loader, Prompt_loader, Topic_Loader
import random
from sklearn.metrics import confusion_matrix, roc_auc_score

def train(model, tokenizer, optimizer, device, loader):
    model.train()
    all_loss = []

    for i, dat in enumerate(loader):
        texts, labels = dat
        texts = list(texts)
        result = tokenizer(texts, return_tensors="pt", padding = 'max_length', max_length = 256, truncation=True)
        texts, masks, labels = result['input_ids'].to(device), result['attention_mask'].to(device), labels.to(device)
        aa = model(texts, labels=labels, attention_mask = masks)
        loss = aa['loss']
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        all_loss.append(loss.item())

    print('avg loss ', np.mean(all_loss), flush = True)


def evaluate(model, tokenizer, device, loader):
    model.eval()
    m = nn.Softmax(dim = 1)
    with torch.no_grad():
        all_scores = []
        all_labels = []
        for i, dat in enumerate(loader):
            texts, labels = dat
            texts = list(texts)

            result = tokenizer(texts, return_tensors="pt", padding = 'max_length', max_length = 256, truncation=True)
            texts_encode, masks = result['input_ids'].to(device), result['attention_mask'].to(device),
            aa = model(texts_encode, attention_mask = masks)
            logits = aa['logits']
            score = m(logits)[:, 1]
            all_scores.append(score.cpu().numpy().flatten())
            all_labels.append(labels[0])

        all_scores_vec = np.concatenate(all_scores)
        all_labels = np.array(all_labels)
        auc = roc_auc_score(all_labels, all_scores_vec)
        f1 = f1_score(all_labels, all_scores_vec > 0.5)

        confusion = confusion_matrix(all_labels, all_scores_vec > 0.5)
        TP = confusion[1, 1]
        FP = confusion[0, 1]
        TN = confusion[0, 0]
        FN = confusion[1, 0]
        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)
        return auc, f1, TPR, 1 - FPR

def run(batch_size=24,
        detect_model_name = 'base',
        learning_rate=2e-5,
        max_epoch = 3,
        domain = 'news',
        seed = 1000,
        prompt = 'p1',
        topic = 'medical',
        cache_dir = 'data'
):

    args = locals()
    device = 'cuda'

    random.seed(seed)
    torch.manual_seed(seed)

    ## initiate RoBERTa model
    model_name = 'roberta-large' if detect_model_name == 'large' else 'roberta-base'
    tokenization_utils.logger.setLevel('ERROR')
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(model_name).to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    best_model = RobertaForSequenceClassification.from_pretrained(model_name).to(device)

    # load the dataset
    train_loader, valid_loader, test_loader = Loader(batch_size, domain=domain, prompt='all', topic='all', cache_dir = cache_dir)
    test_loader1 = Domain_loader(domain='news', cache_dir = cache_dir)
    test_loader2 = Domain_loader(domain='review', cache_dir = cache_dir)
    test_loader3 = Domain_loader(domain='writing', cache_dir = cache_dir)
    test_loader4 = Domain_loader(domain='qa', cache_dir = cache_dir)

    best_f1 = 0
    for epoch in range(1, 1 + max_epoch):
        print('Now training epoch ' + str(epoch), flush=True)
        train(model, tokenizer, optimizer, device, train_loader)
        _, f1, _, _ = evaluate(model, tokenizer, device, valid_loader)

        if f1 > best_f1:
            best_f1 = f1
            best_model.load_state_dict(model.state_dict())

            # torch.save(model.state_dict(), 'your model name here .pt')
            if best_f1 > 0.999:
                break

    ## test on the given domain
    a0, f0, tp0, fp0 = evaluate(best_model, tokenizer, device, test_loader)

    ## test on other domains
    a1, f1, tp1, fp1 = evaluate(best_model, tokenizer, device, test_loader1)
    a2, f2, tp2, fp2 = evaluate(best_model, tokenizer, device, test_loader2)
    a3, f3, tp3, fp3 = evaluate(best_model, tokenizer, device, test_loader3)
    a4, f4, tp4, fp4 = evaluate(best_model, tokenizer, device, test_loader4)

    print('** in domain **', np.round(np.array([a0, f0, tp0, fp0]), 3))

    ##
    print('** out of domain')
    print(np.round(np.array([a1, f1, tp1, fp1]), 3))
    print(np.round(np.array([a2, f2, tp2, fp2]), 3))
    print(np.round(np.array([a3, f3, tp3, fp3]), 3))
    print(np.round(np.array([a4, f4, tp4, fp4]), 3))
    print('''''')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epoch', type=int, default= 5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=1000)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--detect_model_name', type = str, default= 'base')
    parser.add_argument('--domain', type = str, default= 'qa')
    parser.add_argument('--topic', type = str, default= 'all')
    parser.add_argument('--prompt', type = str, default= 'all')
    parser.add_argument('--cache_dir', type = str, default= 'data')
    args = parser.parse_args()
    print('Training with RoBERTa Classification Model')
    print(args)
    run(**vars(args))
