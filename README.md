# CRFs for Sentiment Analysis
50.040 Natural Language Processing project.

Authors: Change Jun Qing, Koh Jing Yu, Vincent Setiawan.

## Results
Our results for the two datasets are located in the `output/` directory. Contained within are the following files are the outputs for the respective tasks:

```
output/
    EN/
        dev.p2.out
        dev.p4.out
        dev.p5.out
        test.p5.out
    ES/
        dev.p2.out
        dev.p4.out
        dev.p5.out
        test.p5.out
```

## Directory
Directory structure should look like the following. The `Evaluation_Script` and `data` folders can be acquired from the course website.

```
CRF_FinalProject.ipynb
models.py
train.py
generate.py
Evaluation_Script/
data/
    EN/
        train
        dev.in
        test.in
    ES/
        train
        dev.in
        test.in
models/
itot_EN.txt
itot_ES.txt
itow_EN.txt
itow_ES.txt
ttoi_EN.txt
ttoi_ES.txt
wtoi_EN.txt
wtoi_ES.txt
```

## Training
To train our BiLSTM + CRF and BERT + BiLSTM + CRF models, execute the following line in the terminal:

```
python train_lstm_crf.py
```

The hyperparameters of the models can be set. For example, to run our best model on the EN dataset:
```
python train_lstm_crf.py --hidden 512 --nlayers 2 --lr 0.001 --arch bilstm --dropout 0.5 --dataset EN
```

Similarly, for the ES dataset:
```
python train_lstm_crf.py --hidden 256 --nlayers 2 --lr 0.0005 --arch bilstm --dropout 0.5 --dataset ES
```

### Grid Search
We included a bash script, `grid_search.sh` that shows how to run grid search over the set of hyperparameters as detailed in our report. To run the grid search, execute:

```
./grid_search.sh
```

in the terminal. This runs a batch of jobs in parallel, and saves the models and results to file.

## Inference
We also provide scripts to perform inference. Our best models are saved in `checkpoint/best/`. To use our best performing BiLSTM model for inference on the dev set, run:

```
python generate_bilstm_crf.py bilstm_h512_n2_lr0.001000_d0.5_EN --checkpoint_dir checkpoint/best/ --dataset EN --dataset_split dev
```

A similar command can also be run for the test data:
```
python generate_bilstm_crf.py bilstm_h512_n2_lr0.001000_d0.5_EN --checkpoint_dir checkpoint/best/ --dataset EN --dataset_split test
```

The outputs will be saved to the `preds/` directory by default.
