import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from pathlib import Path
import sys
import os
import matplotlib.pyplot as plt

data_dir = Path("data/")
output_model_dir = Path("models/")
if not os.path.exists(output_model_dir):
    os.mkdir(output_model_dir)
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

class POSDataset(Dataset):
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

class BaseLineBLSTM(nn.Module):
    '''
    Baseline Bi-directional LSTM Model to check what we expect to see when using a Bi-directional LSTM
    Inputs:
        vocab_size (int): size of vocabulary
        embedding_dim (int): size of embedding layer
        hidden_dim (int): size of hidden_dim for LSTM
        n_layers (int): number of LSTM to stack
        tagset_size (int): size of output space (number of tags)
    '''

    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers, tagset_size):
        super(BaseLineBLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embed = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, bidirectional=True)
        self.hiddentotag = nn.Linear(hidden_dim*2, tagset_size)
    
    def forward(self, x):
        embeds = self.word_embed(x)
        lstm_out, _ = self.lstm(embeds)
        tag_space = self.hiddentotag(lstm_out)
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

class OSCAR(nn.Module):

    def __init__(self):
        pass

    def forward(self, x):
        pass

def train(train_loader, model, optimizer, criterion, device):
    # Set model to training mode
    model.train()
    # Iterate through training data
    total_loss = 0
    for data, target in train_loader:
        # Send data and target to device (cuda if cuda is available)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        # Get predictions from output
        output = model(data)
        output = output.transpose(1,2)
        # Calculate loss
        loss = criterion(output, target)
        total_loss += loss.item()
        # Update Model
        loss.backward()
        optimizer.step()
    return total_loss/len(train_loader)


def main():
    # Hyper Parameters
    EMBEDDING_SIZE = 256
    HIDDEN_DIM = 256
    N_LAYERS = 2
    LEARNING_RATE = 1e-4
    MOMENTUM = 0.9
    WEIGHT_DECAY = 1e-5
    EPOCHS = 30

    # Criterion to for loss
    criterion = nn.NLLLoss()

    # Init Dataset
    posdataset = POSDataset(data_dir)
    train_loader = DataLoader(posdataset)

    # Define Baseline Model
    baselinemodel = BaseLineBLSTM(len(posdataset.wtoi), EMBEDDING_SIZE, HIDDEN_DIM, N_LAYERS, len(posdataset.ttoi))
    baselinemodel.to(device)
    # Set up optimizer for Baseline Model
    optimizer = optim.SGD(baselinemodel.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    # Baseline model training
    best_loss = sys.maxsize
    losses = []
    for epoch in range(1, EPOCHS+1):
        loss = train(train_loader, baselinemodel, optimizer, criterion, device)
        losses.append(loss)
        if loss < best_loss:
            best_loss = loss
            torch.save(baselinemodel, output_model_dir / 'baseline-{}.pt'.format(loss))

    plt.figure()
    plt.title('Base Line Model Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(losses)
    plt.savefig('BaselineTraining.png')


if __name__ == '__main__':
    main()