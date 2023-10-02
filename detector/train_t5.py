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


def train_t5_sentinel(model, tokenizer, optimizer, device, loader, accumulaiton_steps = 2):
    word = 'positive'
    tokens = tokenizer.tokenize(word)
    positive_token_id = tokenizer.convert_tokens_to_ids(tokens)

    word = 'negative'
    tokens = tokenizer.tokenize(word)
    negative_token_id = tokenizer.convert_tokens_to_ids(tokens)

    model.train()
    for j, dat in enumerate(loader):
        texts, labels = dat
        texts = list(texts)
        result = tokenizer(texts, return_tensors="pt", padding = 'max_length', max_length = 256, truncation=True)
        texts, masks, labels = result['input_ids'].to(device), result['attention_mask'].to(device), labels.to(device)

        batch_size = texts.shape[0]
        t5_labels_list = []

        optimizer.zero_grad()
        for i in range(batch_size):
            if labels[i] == 0:
                t5_labels_list.append(positive_token_id)
            else:
                t5_labels_list.append(negative_token_id)
        t5_labels = torch.LongTensor(t5_labels_list).to(model.device)
        decoder_input_ids = torch.tensor([tokenizer.pad_token_id] * batch_size).unsqueeze(-1).to(model.device)

        outputs = model(texts, labels=t5_labels, decoder_input_ids=decoder_input_ids)
        loss = outputs['loss']
        loss = loss / accumulaiton_steps
        loss.backward()

        if (j+1) % accumulaiton_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

# validate the trained t5 sentinel model
def evaluate(model, tokenizer, device, loader):
    model.eval()
    all_scores = []
    all_labels = []
    word = 'positive'
    tokens = tokenizer.tokenize(word)
    positive_token_id = tokenizer.convert_tokens_to_ids(tokens)[0]

    word = 'negative'
    tokens = tokenizer.tokenize(word)
    negative_token_id = tokenizer.convert_tokens_to_ids(tokens)[0]

    with torch.no_grad():
        for i, dat in enumerate(loader):
            texts, labels = dat
            texts = list(texts)
            result = tokenizer(texts, return_tensors="pt", padding='max_length', max_length=256, truncation=True)
            texts, masks = result['input_ids'].to(device), result['attention_mask'].to(device)
            batch_size = texts.shape[0]

            decoder_input_ids = torch.tensor([tokenizer.pad_token_id] * batch_size).unsqueeze(-1).to(model.device)
            logits = model(input_ids=texts, decoder_input_ids=decoder_input_ids)
            logits = logits[0].squeeze(1)
            selected_logits = logits[:, [positive_token_id, negative_token_id]]
            logits = torch.softmax(selected_logits, dim=-1)
            score = (logits)[:, 1]
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
    model = transformers.T5ForConditionalGeneration.from_pretrained('t5-base').cuda()
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    tokenizer = transformers.T5Tokenizer.from_pretrained('t5-base')
    best_model = transformers.T5ForConditionalGeneration.from_pretrained('t5-base').cuda()

    # load the dataset
    train_loader, valid_loader, test_loader = Loader(batch_size, domain=domain, prompt='all', topic='all', cache_dir = cache_dir)
    test_loader1 = Domain_loader(domain='news', cache_dir = cache_dir)
    test_loader2 = Domain_loader(domain='review', cache_dir = cache_dir)
    test_loader3 = Domain_loader(domain='writing', cache_dir = cache_dir)
    test_loader4 = Domain_loader(domain='qa', cache_dir = cache_dir)

    best_f1 = 0
    for epoch in range(1, 1 + max_epoch):
        print('Now training epoch ' + str(epoch), flush=True)
        train_t5_sentinel(model, tokenizer, optimizer, device, train_loader)
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
