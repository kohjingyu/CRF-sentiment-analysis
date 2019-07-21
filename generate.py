import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from pathlib import Path
import matplotlib.pyplot as plt
from models import *

data_dir = Path("data/")
model_dir = Path("checkpoint/")
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
models_set = ['baseline']  # What models are available
model_choice = 0  # Choice of model, tallies with models_set


class POSTrainDataset(Dataset):
    '''
    Part Of Speech Tagging Train Dataset
    Inputs:
        path (Path object or str): path on the local directory to the dataset to load from.
        dataset_choice (str): 'EN' or 'ES', defaults to 'EN'
    '''

    def __init__(self, path, dataset_choice='EN'):
        self.path = path
        self.dataset_choice = dataset_choice
        self.itow = {0: 'UNK'}  # Dict to map index to word
        self.wtoi = {'UNK': 0}  # Dict to map word to index
        self.itot = {}  # Dict to map index to tag
        self.ttoi = {}  # Dict to map tag to index
        self.dataset = []

        # Counters to keep track of last used index for mapping
        word_ctr = 1
        tag_ctr = 0
        # Read Data from path
        with open(self.path / self.dataset_choice / 'train', encoding="utf-8") as f:
            data = []
            target = []
            for line in f:
                # Strip newline
                formatted_line = line.strip()
                # Only process lines that are not newlines
                if len(formatted_line) > 0:
                    # Split into (x, y) pair
                    split_data = formatted_line.split(" ")
                    x, y = split_data[0].lower(), split_data[1]

                    # Add x to maps if it does not exist
                    if x not in self.wtoi:
                        self.wtoi[x] = word_ctr
                        self.itow[word_ctr] = x
                        word_ctr += 1

                    # Add y to maps if it does not exist
                    if y not in self.ttoi:
                        self.ttoi[y] = tag_ctr
                        self.itot[tag_ctr] = y
                        tag_ctr += 1

                    # Add index of word and index of tag into data and target
                    data.append(self.wtoi[x])
                    target.append(self.ttoi[y])

                else:
                    # End of sentence
                    self.dataset.append((data, target))
                    data = []
                    target = []

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, target = self.dataset[idx]
        return torch.Tensor(data).long(), torch.Tensor(target).long()


class POSGenDataset(Dataset):
    '''
    Part Of Speech Tagging Dataset for Generation
    Inputs:
        path (Path object or str): path on the local directory to the dataset to load from.
        train_wtoi (dict): Dictionary to map word to index
        train_ttoi (dict): Dictionary to map tag to index
        dataset_choice (str): 'EN' or 'ES', defaults to 'EN'
    '''

    def __init__(self, path, train_wtoi, train_ttoi, dataset_choice='EN'):
        self.path = path
        self.dataset_choice = dataset_choice
        self.wtoi = train_wtoi
        self.ttoi = train_ttoi
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

def main():
    pass


if __name__ == '__main__':
    # Init Train Dataset for vocab
    posdataset = POSTrainDataset(data_dir)

    if model_choice == 0:
        # Load baseline
        baselinemodel = torch.load(model_dir / 'baseline.pt', map_location=device)
        baselinemodel.to(device)

    posgendata = POSGenDataset(data_dir, posdataset.wtoi, posdataset.ttoi)
    gen_loader = DataLoader(posgendata)

    gen_tag = generate(gen_loader, baselinemodel, device)
    ctr = 0
    with open('dev.p5.out', 'w') as f:
        for sentence in gen_tag:
            for i in range(len(sentence[0])):
                f.write('{} {}\n'.format(posgendata.true_word[ctr], posdataset.itot[sentence[0][i].item()]))
                ctr += 1
            f.write('\n')
    print('Completed')