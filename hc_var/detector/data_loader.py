
from torch.utils.data import Dataset, DataLoader
import numpy as np
from datasets import load_dataset
from datasets.utils.logging import disable_progress_bar
disable_progress_bar()
import datasets
datasets.logging.set_verbosity_error()
import random

class TextDataset(Dataset):
    def __init__(self, real_texts, fake_texts):
        self.real_texts = real_texts
        self.fake_texts = fake_texts

    def __len__(self):
        return len(self.real_texts) + len(self.fake_texts)

    def __getitem__(self, index):
        if index < len(self.real_texts):
            answer = self.real_texts[index]
            label = 0
        else:
            answer = self.fake_texts[index - len(self.real_texts)]
            label = 1
        return answer, label

def Loader(batch_size, domain = 'review', prompt = 'all', topic = 'all', cache_dir = 'data'):

    all_dat0 = load_dataset("hannxu/hc_var", cache_dir = cache_dir, split = 'train')
    all_dat1 = [d for d in all_dat0 if d['domain'] == domain]

    if not (topic == 'all'):
        all_dat = [d for d in all_dat1 if d['topic'] == topic]
    else:
        all_dat = all_dat1

    random.shuffle(all_dat)
    if prompt == 'p1':
        fake_data = [d['text'] for d in all_dat if d['pp_id'] == 1]
    elif prompt == 'p2':
        fake_data = [d['text'] for d in all_dat if d['pp_id'] == 2]
    elif prompt == 'p3':
        fake_data = [d['text'] for d in all_dat if d['pp_id'] == 3]
    elif prompt == 'all':
        fake_data = [d['text'] for d in all_dat if d['label'] == 1]
    else:
        raise ValueError

    random.shuffle(fake_data)
    fake_data = fake_data[0:4500]
    fake_train = fake_data[0:len(fake_data) - 400]
    fake_valid = fake_data[len(fake_data) - 400:len(fake_data) - 150]
    fake_test = fake_data[len(fake_data) - 150:]

    real_data = [d['text'] for d in all_dat if d['label'] == 0]
    random.shuffle(real_data)
    real_train = real_data[0:len(fake_train)]  ## make the training set to be balanced
    train_dataset = TextDataset(real_train, fake_train)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=0)

    real_valid = real_data[len(real_data) - 400:len(real_data) - 250:]
    real_test = real_data[len(real_data) - 400:]

    test_dataset = TextDataset(real_test, fake_test)
    test_loader = DataLoader(test_dataset, 1, shuffle=True, num_workers=0)

    valid_dataset = TextDataset(real_valid, fake_valid)
    valid_loader = DataLoader(valid_dataset, 1, shuffle=True, num_workers=0)

    return train_loader, valid_loader, test_loader


def Domain_loader(domain = 'review', cache_dir = 'data'):
    all_dat = load_dataset("hannxu/hc_var", cache_dir = cache_dir, split = 'train')
    all_dat = [d for d in all_dat if d['domain'] == domain]
    random.shuffle(all_dat)

    real_data = [d['text'] for d in all_dat if d['label'] == 0]
    fake_data = [d['text'] for d in all_dat if d['label'] == 1]
    real_test = real_data[len(real_data) - 400:]
    fake_test = fake_data[len(fake_data) - 400:]

    test_dataset = TextDataset(real_test, fake_test)
    test_loader = DataLoader(test_dataset, 1, shuffle=True, num_workers=0)

    return test_loader

def Prompt_loader(domain = 'review', prompt = 'p1', topic = 'all', cache_dir = 'data'):
    all_dat = load_dataset("hannxu/hc_var", cache_dir = cache_dir, split = 'train')

    if not (topic == 'all'):
        all_dat = [d for d in all_dat if d['topic'] == topic]

    all_dat = [d for d in all_dat if d['domain'] == domain]
    random.shuffle(all_dat)

    real_data = [d['text'] for d in all_dat if d['label'] == 0]
    fake_data = [d for d in all_dat if d['label'] == 1]

    if prompt == 'p1':
        fake_data = [d['text'] for d in all_dat if d['pp_id'] == 1]
    if prompt == 'p2':
        fake_data = [d['text'] for d in all_dat if d['pp_id'] == 2]
    if prompt == 'p3':
        fake_data = [d['text'] for d in all_dat if d['pp_id'] == 3]
    if prompt == 'all':
        fake_data = [d['text'] for d in all_dat if d['label'] == 1]

    real_test = real_data[len(real_data) - 400:]
    fake_test = fake_data[len(fake_data) - 400:]
    test_dataset = TextDataset(real_test, fake_test)
    test_loader = DataLoader(test_dataset, 1, shuffle=True, num_workers=0)

    return test_loader


def Topic_Loader(domain = 'review', topic = 'NA', cache_dir = 'data'):
    all_dat = load_dataset("hannxu/hc_var", cache_dir = cache_dir, split = 'train')
    all_dat = [d for d in all_dat if d['domain'] == domain]
    all_dat = [d for d in all_dat if d['topic'] == topic]

    random.shuffle(all_dat)
    real_data = [d['text'] for d in all_dat if d['label'] == 0]
    fake_data = [d['text'] for d in all_dat if d['label'] == 1]

    real_test = real_data[len(real_data) - 400:]
    fake_test = fake_data[len(fake_data) - 400:]
    test_dataset = TextDataset(real_test, fake_test)
    test_loader = DataLoader(test_dataset, 1, shuffle=True, num_workers=0)

    return test_loader