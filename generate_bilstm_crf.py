import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from pathlib import Path
import matplotlib.pyplot as plt
from models import *
import json
import sys
import os
from pytorch_transformers import BertTokenizer

import argparse

parser = argparse.ArgumentParser(description='Run a deep learning model with CRF for POS tagging. Example usage: python3 generate_bilstm_crf_filename.py bilstm_h512_n2_lr0.001000_d0.5 --checkpoint_dir checkpoint/best')
parser.add_argument('--arch', default="bilstm", type=str, help='')
parser.add_argument('--dataset', default="EN", type=str, help='dataset to evaluate on (EN / ES)')
parser.add_argument('--dataset_split', default="EN", type=str, help='dataset split to evaluate on (train / dev / test)')
parser.add_argument('--checkpoint_dir', default="checkpoint", type=str, help='where the checkpoint files are located')
parser.add_argument('filename', type=str, help='model checkpoint filename (without .pt)')

args = parser.parse_args()
model_name = args.filename
dataset = args.dataset
assert(dataset in ["EN", "ES"])
dataset_split = args.dataset_split
assert(dataset_split in ["train", "dev", "test"])
model_arch = args.arch

print(args)

data_dir = Path("data/")
checkpoint_dir = Path(args.checkpoint_dir)
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
ttoi_file = f'ttoi_{dataset}.txt'
itot_file = f'itot_{dataset}.txt'
wtoi_file = f'wtoi_{dataset}.txt'

class POSGenDataset(Dataset):
    '''
    Part Of Speech Tagging Dataset for Generation
    Inputs:
        path (Path object or str): path on the local directory to the dataset to load from.
        train_wtoi (dict): Dictionary to map word to index
        train_ttoi (dict): Dictionary to map tag to index
        dataset_choice (str): 'EN' or 'ES', defaults to 'EN'
    '''

    def __init__(self, path, train_wtoi, train_ttoi, dataset_choice='EN', dataset_split='dev'):
        self.path = path
        self.dataset_choice = dataset_choice
        self.dataset = []
        self.true_word = []
        self.wtoi = train_wtoi
        self.ttoi = train_ttoi

        self.tokenizer = BertTokenizer.from_pretrained(
            'bert-base-multilingual-cased', do_lower_case=False)

        with open(self.path / self.dataset_choice / dataset_split + ".in", encoding="utf-8") as f:
            data = []
            for line in f:
                # Strip newline
                formatted_line = line.strip()
                # Only process lines that are not newlines
                if len(formatted_line) > 0:
                    # Save the captialisation
                    self.true_word.append(formatted_line)
                    x = formatted_line.lower()

                    if model_arch == "bilstm":
                        # Check if word in vocab
                        if x in self.wtoi:
                            # Add index of word if it exist in vocab
                            data.append(self.wtoi[x])
                        else:
                            # Add index of UNK if it does not
                            data.append(self.wtoi['UNK'])
                    else:
                        data.append(x)
                else:
                    # End of sentence
                    self.dataset.append(data)
                    data = []

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]

        if model_arch == "bilstm":
            return torch.tensor(data).long()

        # Manual token to id to fit the way our dataset works
        input_ids = self.tokenizer.encode(' '.join(data))

        input_ids = [x for x in input_ids if x != 17]
        reverse_token = [
            self.tokenizer._convert_id_to_token(x) for x in input_ids]

        idx_track = []
        ctr = 0
        construct = ''
        first = True
        # Find the respective starting index of each word
        for i in range(len(reverse_token)):
            if reverse_token[i] == data[ctr] or reverse_token[i] == '[UNK]':
                idx_track.append(i)
                ctr += 1
            else:
                if first:
                    idx_track.append(i)
                    first = False
                if reverse_token[i].startswith('#'):
                    construct += reverse_token[i][2:]
                else:
                    construct += reverse_token[i]

                if construct == data[ctr]:
                    ctr += 1
                    construct = ''
                    first = True

        return torch.tensor(input_ids), torch.tensor(idx_track)


def generate(gen_loader, model, device, split_words):
    assert split_words == 'first' or split_words == 'avg'
    model.eval()
    gen_tag = []
    with torch.no_grad():
        for gen_dat in gen_loader:
            if model_arch == "bilstm":
                data = gen_dat.to(device)
                score, pred = model(data)
            else:
                data, idx = gen_dat
                data, idx = data.to(device), idx.to(device)
                score, pred = model(data, idx)

            gen_tag.append(pred)
    return gen_tag


if __name__ == '__main__':
    with open(ttoi_file, 'r', encoding="utf-8") as f:
        ttoi = json.load(f)
    with open(wtoi_file, 'r', encoding="utf-8") as f:
        wtoi = json.load(f)
    with open(itot_file, 'r', encoding="utf-8") as f:
        itot = json.load(f)

    output_dir = "preds"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Load model
    model = torch.load(checkpoint_dir / '{}.pt'.format(model_name), map_location=device)
    model.to(device)

    SPLIT_WORDS = 'first'

    posgendata = POSGenDataset(data_dir, wtoi, ttoi, dataset_choice=dataset, dataset_split=dataset_split)
    gen_loader = DataLoader(posgendata)

    gen_tag = generate(gen_loader, model, device, SPLIT_WORDS)
    ctr = 0
    with open(f'{output_dir}/{model_name}_{dataset_split}.p5.out', 'w', encoding="utf-8") as f:
        for sentence in gen_tag:
            for i in range(len(sentence)):
                f.write('{} {}\n'.format(
                    posgendata.true_word[ctr], itot[str(sentence[i])]))
                ctr += 1
            f.write('\n')
    print('Completed')
