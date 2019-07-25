import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from pathlib import Path
import sys
import os
import random
import matplotlib.pyplot as plt
from models import *
import json

data_dir = Path("data/")
output_model_dir = Path("checkpoint/")
if not os.path.exists(output_model_dir):
    os.mkdir(output_model_dir)
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
models_set = ['baseline', 'seq2seq']  # What models are available
model_choice = 1  # Choice of model, tallies with models_set


class POSTrainDataset(Dataset):
    '''
    Part Of Speech Tagging Train Dataset
    Inputs:
        path (Path object or str): path on the local directory to the dataset to load from.
        dataset_choice (str): 'EN' or 'ES', defaults to 'EN'
        split (float): Between 0 - 1, indicates size of train set where 1-split is the size of validation set
    '''

    def __init__(self, path, dataset_choice='EN', split=0.8):
        self.path = path
        self.dataset_choice = dataset_choice
        self.itow = {0: 'UNK'}  # Dict to map index to word
        self.wtoi = {'UNK': 0}  # Dict to map word to index
        self.itot = {}  # Dict to map index to tag
        self.ttoi = {}  # Dict to map tag to index
        self.dataset = []
        self.train = True

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

        # Shuffle data and split into train and val
        random.shuffle(self.dataset)
        self.train_data = self.dataset[:int(len(self.dataset)*split)]
        self.val_data = self.dataset[int(len(self.dataset)*split):]
        with open('wtoi.txt', 'w', encoding="utf-8") as f:
            json.dump(self.wtoi, f)
        with open('ttoi.txt', 'w', encoding="utf-8") as f:
            json.dump(self.ttoi, f)
        with open('itot.txt', 'w', encoding="utf-8") as f:
            json.dump(self.itot, f)

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.val_data)

    def __getitem__(self, idx):
        if self.train:
            data, target = self.train_data[idx]
        else:
            data, target = self.val_data[idx]
        return torch.Tensor(data).long(), torch.Tensor(target).long()


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
        output = output.transpose(1, 2)
        # Calculate loss
        loss = criterion(output, target)
        total_loss += loss.item()
        # Update Model
        loss.backward()
        optimizer.step()
    return total_loss/len(train_loader)


def eval(eval_loader, model, criterion, device):
    # Set model to eval mode
    model.eval()

    # Counters
    total_loss = 0
    total_correct = 0
    total_constituents = 0
    stats = {}

    # Iterate evaluluation data
    # Set no grad
    with torch.no_grad():
        for data, target in eval_loader:
            # Send data and target to device (cuda if cuda is available)
            data, target = data.to(device), target.to(device)

            # Get predictions
            output = model(data)
            output = output.transpose(1, 2)
            # Calculate Loss
            loss = criterion(output, target)
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            for i in range(len(pred)):
                total_constituents += 1
                if target[0][i].item() not in stats:
                    stats[target[0][i].item()] = {'TP': 0, 'FP': 0, 'FN': 0}
                if pred[0][i].item() == target[0][i].item():
                    total_correct += 1
                    if pred[0][i].item() not in stats:
                        stats[pred[0][i].item()] = {'TP': 0, 'FP': 0, 'FN': 0}
                    stats[pred[0][i].item()]['TP'] += 1
                if pred[0][i].item() != target[0][i].item():
                    if pred[0][i].item() not in stats:
                        stats[pred[0][i].item()] = {'TP': 0, 'FP': 0, 'FN': 0}
                    if target[0][i].item() not in stats:
                        stats[target[0][i].item()] = {
                            'TP': 0, 'FP': 0, 'FN': 0}
                    stats[pred[0][i].item()]['FP'] += 1
                    stats[target[0][i].item()]['FN'] += 1

    avg_precision = []
    avg_recall = []
    for key in stats:
        if (stats[key]['TP']+stats[key]['FP']) != 0:
            avg_precision.append(
                stats[key]['TP']/(stats[key]['TP']+stats[key]['FP']))
        if (stats[key]['TP']+stats[key]['FN']) != 0:
            avg_recall.append(stats[key]['TP'] /
                              (stats[key]['TP']+stats[key]['FN']))

    return total_loss/len(eval_loader), total_correct/total_constituents, sum(avg_precision)/len(avg_precision), sum(avg_recall)/len(avg_recall)


def main():
    # Init Train Dataset
    posdataset = POSTrainDataset(data_dir)
    loader = DataLoader(posdataset)

    # Criterion to for loss
    weighted_loss = torch.ones(len(posdataset.ttoi))
    weighted_loss[posdataset.ttoi['O']] = 0.1
    criterion = nn.CrossEntropyLoss(weight=weighted_loss)

    if model_choice == 0:
        # Hyper Parameters
        EMBEDDING_SIZE = 512
        HIDDEN_DIM = 512
        N_LAYERS = 2
        LEARNING_RATE = 1e-3
        MOMENTUM = 0.9
        WEIGHT_DECAY = 1e-5
        EPOCHS = 200
        # Define Baseline Model
        model = BaseLineBLSTM(
            len(posdataset.wtoi), EMBEDDING_SIZE, HIDDEN_DIM, N_LAYERS, len(posdataset.ttoi))
        model.to(device)
        # Set up optimizer for Baseline Model
        optimizer = optim.SGD(model.parameters(
        ), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    elif model_choice == 1:
        # Hyper Parameters
        EMBEDDING_SIZE = 512
        HIDDEN_DIM = 512
        N_LAYERS = 2
        LEARNING_RATE = 1e-3
        MOMENTUM = 0.9
        WEIGHT_DECAY = 1e-5
        EPOCHS = 200
        # Define Seq2Seq Model
        model = Seq2Seq(
            len(posdataset.wtoi), EMBEDDING_SIZE, HIDDEN_DIM, N_LAYERS, len(posdataset.ttoi))
        model.to(device)
        # Set up optimizer for Seq2Seq Model
        optimizer = optim.SGD(model.parameters(
        ), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    # Implement Early stop
    early_stop = 0

    # Model training and eval
    best_loss = sys.maxsize
    train_losses = []
    eval_losses = []
    eval_accuracies = []
    for epoch in range(1, EPOCHS+1):
        # Toggle Train set
        posdataset.train = True
        trainloss = train(loader, model,
                          optimizer, criterion, device)
        train_losses.append(trainloss)
        # Toggle Validation Set
        posdataset.train = False
        loss, accuracy, precision, recall = eval(
            loader, model, criterion, device)
        eval_losses.append(loss)
        eval_accuracies.append(accuracy)
        print('Epoch {}, Training Loss: {}, Evaluation Loss: {}, Evaluation Accuracy: {}, Evaluation Precision: {}, Evaluation Recall: {}'.format(
            epoch, trainloss, loss, accuracy, precision, recall))

        # Check if current loss is better than previous
        if loss < best_loss:
            best_loss = loss
            torch.save(model, output_model_dir /
                       '{}.pt'.format(models_set[model_choice]))
            early_stop = 0

        # If loss has stagnate, early stop
        else:
            early_stop += 1
            if early_stop >= 5:
                print('Early Stopping')
                break

    # Plot respective graphs for visualisation
    plt.figure()
    plt.title('{} Model Training'.format(models_set[model_choice]))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(train_losses)
    plt.savefig('{}Training.png'.format(models_set[model_choice]))

    plt.figure()
    plt.title('{} Model Evaluation'.format(models_set[model_choice]))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(eval_losses)
    plt.savefig('{}EvalLoss.png'.format(models_set[model_choice]))

    plt.figure()
    plt.title('{} Model Evaluation'.format(models_set[model_choice]))
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(eval_accuracies)
    plt.savefig('{}EvalAcc.png'.format(models_set[model_choice]))


if __name__ == '__main__':
    main()
