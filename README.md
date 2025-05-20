# DL-assignment-3
DA6401 DL assignment 3 IITM

[Wandb report](https://wandb.ai/cs24m041-iit-madras/DA6401%20DL%20Assignment%203/reports/DA6401-Assignment-3--VmlldzoxMjM4OTg5MQ?accessToken=ilszgsw764080zwi6kbkh2fewh0qqvk8z8gwhnqamkp6fys0m7xknqnzvlpv9lyo)

---

# Malayalam-English Transliteration with Seq2Seq + Attention (PyTorch)

This project implements a sequence-to-sequence (Seq2Seq) model with optional attention for character-level **Malayalam to English** transliteration. It uses PyTorch for training and inference, and supports configurable hyperparameters through `argparse`.

---

##  Features

* Character-level transliteration using LSTM/GRU/RNN cells
* Optional attention mechanism
* Configurable encoder and decoder layers, embedding/hidden sizes, dropout, and cell types
* Preprocessing with Keras Tokenizer for consistent character-level tokenization
* Train/validation/test split from the [Dakshina dataset](https://github.com/google-research/dakshina)
* Accuracy metric based on full sequence match

---


---

##  Dependencies

```bash
pip install torch pandas tqdm tensorflow
```
---

##  Usage

### Train the model

```bash
python train.py \
  --batch_size 64 \
  --max_sequence_length 20 \
  --emb_dim 128 \
  --enc_hidden_dim 256 \
  --dec_hidden_dim 256 \
  --enc_layers 2 \
  --dec_layers 3 \
  --dropout 0.3 \
  --cell_type lstm \
  --n_epochs 10 \
  --lr 0.001 \
  --use_attention
```

### Argparse Options

| Argument                | Type  | Default | Description                            |
| ----------------------- | ----- | ------- | -------------------------------------- |
| `--batch_size`          | int   | 64      | Batch size for training                |
| `--max_sequence_length` | int   | 20      | Max length of input sequences          |
| `--emb_dim`             | int   | 128     | Embedding dimension                    |
| `--enc_hidden_dim`      | int   | 256     | Hidden size of encoder                 |
| `--dec_hidden_dim`      | int   | 256     | Hidden size of decoder                 |
| `--enc_layers`          | int   | 2       | Number of layers in encoder            |
| `--dec_layers`          | int   | 3       | Number of layers in decoder            |
| `--dropout`             | float | 0.3     | Dropout rate                           |
| `--cell_type`           | str   | 'lstm'  | RNN cell type: `lstm`, `gru`, or `rnn` |
| `--n_epochs`            | int   | 10      | Number of training epochs              |
| `--lr`                  | float | 0.001   | Learning rate                          |
| `--use_attention`       | flag  | False   | Whether to use attention mechanism     |

---

##  Evaluation

* Accuracy is computed based on full-sequence matches (ignoring padding).

---

##  Dataset

Uses the Malayalam-English subset of the [Dakshina Dataset](https://github.com/google-research/dakshina).

* `ml.translit.sampled.train.tsv`
* `ml.translit.sampled.dev.tsv`
* `ml.translit.sampled.test.tsv`

Each file contains:

```
<malayalam_word>    <english_transliteration>    <count>
```

---


