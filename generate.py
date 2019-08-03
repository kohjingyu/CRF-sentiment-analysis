import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from pathlib import Path
import matplotlib.pyplot as plt
from models import *
import json

data_dir = Path("data/")
model_dir = Path("checkpoint/")
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
# What models are available
models_set = ['baseline', 'seq2seq', 'attnseq2seq']
model_choice = 0  # Choice of model, tallies with models_set
ttoi_file = 'ttoi.txt'
itot_file = 'itot.txt'
wtoi_file = 'wtoi.txt'
itow_file = 'itow.txt'


class POSGenDataset(Dataset):
    '''
    Part Of Speech Tagging Dataset for Generation
    Inputs:
        path (Path object or str): path on the local directory to the dataset to load from.
        train_wtoi (dict): Dictionary to map word to index
        train_ttoi (dict): Dictionary to map tag to index
        dataset_choice (str): 'EN' or 'ES', defaults to 'EN'
    '''

    def __init__(self, path, train_wtoi, train_ttoi, train_itow, dataset_choice='EN'):
        self.path = path
        self.dataset_choice = dataset_choice
        self.wtoi = train_wtoi
        self.ttoi = train_ttoi
        self.itow = train_itow
        self.dataset = []
        self.true_word = []

        with open(self.path / self.dataset_choice / 'dev.in', encoding="utf-8") as f:
            data = []
            for line in f:
                # Strip newline
                formatted_line = line.strip()
                # Only process lines that are not newlines
                if len(formatted_line) > 0:
                    # Save the captialisation
                    self.true_word.append(formatted_line)
                    # Split into (x, y) pair
                    x = formatted_line.lower()

                    # Add index of word and index of tag into data and target
                    # Check if word in vocab
                    if x in self.wtoi:
                        # Add index of word if it exist in vocab
                        data.append(self.wtoi[x])
                    else:
                        # Add index of UNK if it does not
                        data.append(self.wtoi['UNK'])

                else:
                    # End of sentence
                    self.dataset.append(data)
                    data = []

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        return torch.Tensor(data).long()


def generate(gen_loader, model, device):
    model.eval()
    gen_tag = []
    with torch.no_grad():
        for data in gen_loader:
            data = data.to(device)
            output = model(data)
            pred = output.argmax(dim=2)
            gen_tag.append(pred)
    return gen_tag


if __name__ == '__main__':
    # Open indexing files
    with open(ttoi_file, 'r', encoding="utf-8") as f:
        ttoi = json.load(f)
    with open(wtoi_file, 'r', encoding="utf-8") as f:
        wtoi = json.load(f)
    with open(itot_file, 'r', encoding="utf-8") as f:
        itot = json.load(f)
    with open(itow_file, 'r', encoding="utf-8") as f:
        itow = json.load(f)

    # Load model
    model = torch.load(
        model_dir / '{}.pt'.format(models_set[model_choice]), map_location=device)
    model.to(device)
    print(model)

    posgendata = POSGenDataset(data_dir, wtoi, ttoi, itow)
    gen_loader = DataLoader(posgendata)

    gen_tag = generate(gen_loader, model, device)
    ctr = 0
    with open('dev.p5.out', 'w', encoding="utf-8") as f:
        for sentence in gen_tag:
            for i in range(len(sentence[0])):
                f.write('{} {}\n'.format(
                    posgendata.true_word[ctr], itot[str(sentence[0][i].item())]))
                ctr += 1
            f.write('\n')
    print('Completed')
