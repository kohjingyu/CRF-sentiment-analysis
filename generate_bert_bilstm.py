import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from pathlib import Path
import matplotlib.pyplot as plt
from models import *
import json
from pytorch_transformers import BertTokenizer

data_dir = Path("data/")
model_dir = Path("checkpoint/")
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
models_set = ['bertlstm']  # What models are available
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

    def __init__(self, path, dataset_choice='EN'):
        self.path = path
        self.dataset_choice = dataset_choice
        self.dataset = []
        self.true_word = []

        self.tokenizer = BertTokenizer.from_pretrained(
            'bert-base-multilingual-cased', do_lower_case=False)

        with open(self.path / self.dataset_choice / 'dev.in', encoding="utf-8") as f:
            data = []
            for line in f:
                # Strip newline
                formatted_line = line.strip()
                # Only process lines that are not newlines
                if len(formatted_line) > 0:
                    # Save the captialisation
                    self.true_word.append(formatted_line)
                    x = formatted_line.lower()

                    data.append(x)

                else:
                    # End of sentence
                    self.dataset.append(data)
                    data = []

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]

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
        for data, idx in gen_loader:
            data, idx = data.to(device), idx.to(device)
            score, pred = model(data, idx)
            gen_tag.append(pred)
    return gen_tag


if __name__ == '__main__':
    # Open indexing files
    with open(itot_file, 'r', encoding="utf-8") as f:
        itot = json.load(f)

    # Load model
    model = torch.load(
        model_dir / '{}.pt'.format(models_set[model_choice]), map_location=device)
    model.to(device)

    SPLIT_WORDS = 'first'

    posgendata = POSGenDataset(data_dir)
    gen_loader = DataLoader(posgendata)

    gen_tag = generate(gen_loader, model, device, SPLIT_WORDS)
    ctr = 0
    with open('dev.p5.out', 'w', encoding="utf-8") as f:
        for sentence in gen_tag:
            for i in range(len(sentence)):
                f.write('{} {}\n'.format(
                    posgendata.true_word[ctr], itot[str(sentence[i])]))
                ctr += 1
            f.write('\n')
    print('Completed')
