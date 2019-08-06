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
from bert_bilstm_crf import BERT_BiLSTM_CRF
from bilstm_crf import BiLSTM_CRF

import json
import copy
from pytorch_transformers import BertTokenizer, AdamW

import argparse

parser = argparse.ArgumentParser(description='Run a deep learning model with CRF for POS tagging.')
parser.add_argument('--hidden', default=1024, type=int, help='hidden dim of lstm')
parser.add_argument('--nlayers', default=1, type=int, help='number of layers in the lstm')
parser.add_argument('--lr', default=5e-5, type=float, help='learning rate for AdamW optimizer')
parser.add_argument('--arch', default="bertlstm", type=str, help='')

args = parser.parse_args()
print(args)

data_dir = Path("data/")
output_model_dir = Path("checkpoint/")
if not os.path.exists(output_model_dir):
    os.mkdir(output_model_dir)
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

START_TAG = "<START>"
STOP_TAG = "<STOP>"

model_architecture = args.arch # ["bertlstm", "bilstm"]
assert(model_architecture in ["bertlstm", "bilstm"])

# Hyper Parameters
HIDDEN_DIM = args.hidden
N_LAYERS = args.nlayers
LEARNING_RATE = args.lr
ADAMEPS = 1e-8
SCHEDULER_GAMMA = 0.95

class POSTrainDataset(Dataset):
    '''
    Part Of Speech Tagging Train Dataset
    Inputs:
        path (Path object or str): path on the local directory to the dataset to load from.
        dataset_choice (str): 'EN' or 'ES', defaults to 'EN'
        split (float): Between 0 - 1, indicates size of train set where 1-split is the size of validation set
        unk_chance (float): Between 0 - 1, probability of changing a word to 'UNK' at train time (To try to account for UNK in validation), default as 0.01
    '''

    def __init__(self, path, dataset_choice='EN', split=0.8, unk_chance=0.05):
        self.path = path
        self.dataset_choice = dataset_choice
        self.itow = {0: 'UNK', 1: START_TAG, 2: STOP_TAG}  # Dict to map index to word
        self.wtoi = {'UNK': 0, START_TAG: 1, STOP_TAG: 2}  # Dict to map word to index
        self.itot = {0: START_TAG, 1: STOP_TAG}  # Dict to map index to tag
        self.ttoi = {START_TAG: 0, STOP_TAG: 1}  # Dict to map tag to index
        self.dataset = []
        self.train = True
        self.unk_chance = unk_chance

        self.tokenizer = BertTokenizer.from_pretrained(
            'bert-base-multilingual-cased', do_lower_case=False)

        # Counters to keep track of last used index for mapping
        word_ctr = len(self.wtoi)
        tag_ctr = len(self.itot)
        # Read Data from path
        with open(self.path / self.dataset_choice / 'train', encoding="utf-8") as f:
            data = []
            target = []
            self.entities = {}
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
                    if model_architecture == "bilstm":
                        data.append(self.wtoi[x])
                    else:
                        data.append(x)

                    target.append(self.ttoi[y])

                else:
                    # End of sentence
                    self.dataset.append((data, target))
                    data = []
                    target = []

                # For weighted loss
                if y not in self.entities:
                    self.entities[y] = 1
                else:
                    self.entities[y] += 1

        highest = self.entities[max(self.entities, key=self.entities.get)]
        for key in self.entities:
            self.entities[key] = 1/(self.entities[key]/highest)

        # Shuffle data and split into train and val
        random.shuffle(self.dataset)
        self.train_data = self.dataset[:int(len(self.dataset)*split)]
        self.val_data = self.dataset[int(len(self.dataset)*split):]

        with open('itot.txt', 'w', encoding="utf-8") as f:
            json.dump(self.itot, f)

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.val_data)

    def __getitem__(self, idx):
        if self.train:
            data, target = copy.deepcopy(self.train_data[idx])
            for i in range(len(data)):
                if random.random() < self.unk_chance:
                    data[i] = self.wtoi['UNK']
        else:
            data, target = self.val_data[idx]

        if model_architecture == "bilstm":
            return torch.tensor(data).long(), torch.tensor(target).long()
        else:
            # Manual token to id to fit the way our dataset works
            input_ids = self.tokenizer.encode(' '.join(data))

            reverse_token = [
                self.tokenizer._convert_id_to_token(x) for x in input_ids]

            # Find the respective starting index of each word
            idx_track = []
            ctr = 0
            construct = ''
            first = True
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

            return torch.tensor(input_ids), torch.tensor(idx_track), torch.Tensor(target).long()


def train(train_loader, model, optimizer, criterion, device, split_words):
    assert split_words == 'first' or split_words == 'avg'
    # Set model to training mode
    model.train()
    # Iterate through training data
    total_loss = 0
    for train_data in train_loader:
        optimizer.zero_grad()

        if model_architecture == "bilstm":
            data, target = train_data
            data, target = data.to(device), target.to(device)
            # Get predictions from output
            loss = model.neg_log_likelihood(data, target)
        else:
            data, idx, target = train_data
            data, idx, target = data.to(device), idx.to(device), target.to(device)
            # Get predictions from output
            loss = model.neg_log_likelihood(data, idx, target)

        total_loss += loss.item()

        # Update Model
        loss.backward()
        optimizer.step()

    return total_loss/len(train_loader)


def eval(eval_loader, model, criterion, device, split_words):
    assert split_words == 'first' or split_words == 'avg'
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
        for eval_data in eval_loader:
            if model_architecture == "bilstm":
                data, target = eval_data
                # Send data and target to device (cuda if cuda is available)
                data, target = data.to(device), target.to(device)
                score, pred = model(data)

                # Calculate Loss
                loss = model.neg_log_likelihood(data, target)
            else:
                data, idx, target = eval_data
                # Send data and target to device (cuda if cuda is available)
                data, idx, target = data.to(device), idx.to(device), target.to(device)

                # Get predictions
                score, pred = model(data, idx)

                # Calculate Loss
                loss = model.neg_log_likelihood(data, idx, target)


            total_loss += loss.item()
            for i in range(len(pred)):
                total_constituents += 1
                if target[0][i].item() not in stats:
                    stats[target[0][i].item()] = {'TP': 0, 'FP': 0, 'FN': 0}
                if pred[i] == target[0][i].item():
                    total_correct += 1
                    if pred[i] not in stats:
                        stats[pred[i]] = {'TP': 0, 'FP': 0, 'FN': 0}
                    stats[pred[i]]['TP'] += 1
                if pred[i] != target[0][i].item():
                    if pred[i] not in stats:
                        stats[pred[i]] = {'TP': 0, 'FP': 0, 'FN': 0}
                    if target[0][i].item() not in stats:
                        stats[target[0][i].item()] = {
                            'TP': 0, 'FP': 0, 'FN': 0}
                    stats[pred[i]]['FP'] += 1
                    stats[target[0][i].item()]['FN'] += 1


    avg_precision = []
    avg_recall = []
    for key in stats:
        if (stats[key]['TP']+stats[key]['FP']) != 0 and key != eval_loader.dataset.ttoi['O']:
            avg_precision.append(
                stats[key]['TP']/(stats[key]['TP']+stats[key]['FP']))
        if (stats[key]['TP']+stats[key]['FN']) != 0 and key != eval_loader.dataset.ttoi['O']:
            avg_recall.append(stats[key]['TP'] /
                              (stats[key]['TP']+stats[key]['FN']))

    if sum(avg_precision) == 0:
        avg_precision = 0
    else:
        avg_precision = sum(avg_precision)/len(avg_precision)

    if sum(avg_recall) == 0:
        avg_recall = 0
    else:
        avg_recall = sum(avg_recall)/len(avg_recall)

    return total_loss/len(eval_loader), total_correct/total_constituents, avg_precision, avg_recall


def main():
    EPOCHS = 500
    EARLY_STOP_EPOCHS = 30
    SPLIT_WORDS = 'first'

    # Init Train Dataset
    posdataset = POSTrainDataset(data_dir, unk_chance=0)
    loader = DataLoader(posdataset)

    # Criterion to for loss (weighted)
    weighted_loss = torch.ones(len(posdataset.ttoi))

    for key in posdataset.entities:
        weighted_loss[posdataset.ttoi[key]] = posdataset.entities[key]
    weighted_loss = weighted_loss.to(device)
    criterion = nn.CrossEntropyLoss(weight=weighted_loss)

    if model_architecture == "bertlstm":
        model = BERT_BiLSTM_CRF(len(posdataset.wtoi), posdataset.ttoi, HIDDEN_DIM, N_LAYERS)
        # No grad Bert layer
        for p in model.model.parameters():
            p.requires_grad = False
    elif model_architecture == "bilstm":
        EMBEDDING_SIZE = 256
        model = BiLSTM_CRF(len(posdataset.wtoi), posdataset.ttoi, EMBEDDING_SIZE, HIDDEN_DIM, N_LAYERS)

    model.to(device)

    optimizer = AdamW(model.parameters(
    ), lr=LEARNING_RATE, eps=ADAMEPS)
    scheduler = optim.lr_scheduler.ExponentialLR(
        optimizer, SCHEDULER_GAMMA)

    model_name = f"{model_architecture}_h{HIDDEN_DIM}_n{N_LAYERS}_lr{LEARNING_RATE}"
    print(f"Running for {model_name}")

    # Implement Early stop
    early_stop = 0

    # Model training and eval
    best_loss = sys.maxsize
    train_losses = []
    eval_losses = []
    eval_recall = []
    eval_precision = []
    for epoch in range(1, EPOCHS+1):
        scheduler.step()
        # Toggle Train set
        posdataset.train = True
        trainloss = train(loader, model,
                          optimizer, criterion, device, SPLIT_WORDS)
        train_losses.append(trainloss)
        # Toggle Validation Set
        posdataset.train = False
        loss, accuracy, precision, recall = eval(
            loader, model, criterion, device, SPLIT_WORDS)
        eval_losses.append(loss)
        eval_recall.append(recall)
        eval_precision.append(precision)
        print('Epoch {}, Training Loss: {}, Evaluation Loss: {}, Evaluation Accuracy: {}, Evaluation Precision: {}, Evaluation Recall: {}'.format(
            epoch, trainloss, loss, accuracy, precision, recall), flush=True)

        # Check if current loss is better than previous
        if loss < best_loss:
            best_loss = loss
            torch.save(model, output_model_dir /
                       '{}.pt'.format(model_name))
            early_stop = 0

        # If loss has stagnated, early stop
        else:
            early_stop += 1
            if early_stop >= EARLY_STOP_EPOCHS:
                print('Early Stopping')
                break

    # Plot respective graphs for visualisation
    plt.figure()
    plt.title('{} Model Training'.format(model_name))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(train_losses)
    plt.savefig('{}_Training.png'.format(model_name))

    plt.figure()
    plt.title('{} Model Evaluation'.format(model_name))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(eval_losses)
    plt.savefig('{}_EvalLoss.png'.format(model_name))

    plt.figure()
    plt.title('{} Model Evaluation'.format(model_name))
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.plot(eval_precision)
    plt.savefig('{}_EvalPrec.png'.format(model_name))

    plt.figure()
    plt.title('{} Model Evaluation'.format(model_name))
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.plot(eval_recall)
    plt.savefig('{}_EvalRecall.png'.format(model_name))


if __name__ == '__main__':
    main()
