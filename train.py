import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
import wandb
import argparse
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import animation
from PIL import Image
import numpy as np
import io

from tqdm import tqdm
def load_english_data():
    train_path = "ml.translit.sampled.train.tsv"
    dev_path = "ml.translit.sampled.dev.tsv"
    test_path = "ml.translit.sampled.test.tsv"

    train_data = pd.read_csv(train_path, sep='\t', header=None, names=['malayalam', 'english', 'count'])
    dev_data = pd.read_csv(dev_path, sep='\t', header=None, names=['malayalam', 'english', 'count'])
    test_data = pd.read_csv(test_path, sep='\t', header=None, names=['malayalam', 'english', 'count'])

    return train_data, dev_data, test_data

def preprocess_data(train_data, dev_data, test_data, max_sequence_length=20):
    train_data = train_data.dropna()
    dev_data = dev_data.dropna()
    test_data = test_data.dropna()

    all_src = pd.concat([train_data['english'], dev_data['english'], test_data['english']])
    all_trg = pd.concat([train_data['malayalam'], dev_data['malayalam'], test_data['malayalam']])

    src_tokenizer = Tokenizer(char_level=True, lower=False)
    src_tokenizer.fit_on_texts(all_src)

    trg_tokenizer = Tokenizer(char_level=True, lower=False)
    trg_tokenizer.fit_on_texts(all_trg)

    sos_token = '<s>'
    eos_token = '</s>'
    trg_tokenizer.word_index[sos_token] = len(trg_tokenizer.word_index) + 1
    trg_tokenizer.word_index[eos_token] = len(trg_tokenizer.word_index) + 1

    def process_sequences(texts, tokenizer, max_len):
        seq = tokenizer.texts_to_sequences(texts)
        return pad_sequences(seq, maxlen=max_len, padding='post')

    def process_target_sequences(texts, tokenizer, max_len):
        sos = tokenizer.word_index[sos_token]
        eos = tokenizer.word_index[eos_token]
        seq = tokenizer.texts_to_sequences(texts)
        seq = [[sos] + s + [eos] for s in seq]
        return pad_sequences(seq, maxlen=max_len+2, padding='post')

    X_train = process_sequences(train_data['english'], src_tokenizer, max_sequence_length)
    y_train = process_target_sequences(train_data['malayalam'], trg_tokenizer, max_sequence_length)

    X_dev = process_sequences(dev_data['english'], src_tokenizer, max_sequence_length)
    y_dev = process_target_sequences(dev_data['malayalam'], trg_tokenizer, max_sequence_length)

    X_test = process_sequences(test_data['english'], src_tokenizer, max_sequence_length)
    y_test = process_target_sequences(test_data['malayalam'], trg_tokenizer, max_sequence_length)

    return X_train, y_train, X_dev, y_dev, X_test, y_test, src_tokenizer, trg_tokenizer

class Seq2SeqDataset(Dataset):
    def __init__(self, src, trg):
        self.src = torch.LongTensor(src)
        self.trg = torch.LongTensor(trg)

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return self.src[idx], self.trg[idx]
class Attention(nn.Module):
    def __init__(self, enc_hidden_dim, dec_hidden_dim):
        super().__init__()
        self.attn = nn.Linear(enc_hidden_dim + dec_hidden_dim, dec_hidden_dim)
        self.v = nn.Linear(dec_hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        src_len = encoder_outputs.size(1)
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        return torch.softmax(attention, dim=1)
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, n_layers, cell_type='lstm', dropout=0.0):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        rnn_cls = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN}[cell_type.lower()]
        self.rnn = rnn_cls(emb_dim, hidden_dim, num_layers=n_layers,
                           dropout=dropout if n_layers > 1 else 0, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.cell_type = cell_type.lower()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(embedded)
        return outputs, hidden
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hidden_dim, dec_hidden_dim, n_layers, cell_type='lstm', dropout=0.0, use_attention=True):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim)
        rnn_cls = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN}[cell_type.lower()]
        rnn_input_dim = emb_dim + enc_hidden_dim if use_attention else emb_dim
        self.rnn = rnn_cls(rnn_input_dim, dec_hidden_dim, num_layers=n_layers,
                           dropout=dropout if n_layers > 1 else 0, batch_first=True)
        self.use_attention = use_attention
        if use_attention:
            self.attention = Attention(enc_hidden_dim, dec_hidden_dim)
        self.fc_out = nn.Linear(dec_hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.cell_type = cell_type.lower()
        self.n_layers = n_layers
        self.dec_hidden_dim = dec_hidden_dim

    def forward(self, trg, hidden, encoder_outputs, teacher_forcing=True):
        batch_size = trg.size(0)
        trg_len = trg.size(1)
        outputs = []

        input_t = trg[:, 0]  # Start with <sos> token

        if self.cell_type == 'lstm':
            h, c = hidden
        else:
            h = hidden
            c = None

        for t in range(1, trg_len):
            emb_t = self.dropout(self.embedding(input_t)).unsqueeze(1)
            hidden_t = h[-1]

            if self.use_attention:
                attn_weights = self.attention(hidden_t, encoder_outputs)
                context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
                rnn_input = torch.cat((emb_t, context), dim=2)
            else:
                rnn_input = emb_t

            if self.cell_type == 'lstm':
                output, (h, c) = self.rnn(rnn_input, (h, c))
            else:
                output, h = self.rnn(rnn_input, h)

            pred = self.fc_out(output.squeeze(1))
            outputs.append(pred.unsqueeze(1))

            input_t = trg[:, t] if teacher_forcing else pred.argmax(1)

        outputs = torch.cat(outputs, dim=1)
        return outputs, (h, c) if self.cell_type == 'lstm' else h
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.enc_n_layers = encoder.n_layers
        self.dec_n_layers = decoder.n_layers
        self.enc_hidden_dim = encoder.hidden_dim
        self.dec_hidden_dim = decoder.dec_hidden_dim
        self.cell_type = encoder.cell_type
        self.different_dims = self.enc_hidden_dim != self.dec_hidden_dim

        if self.different_dims:
            self.hidden_projection = nn.Linear(self.enc_hidden_dim, self.dec_hidden_dim)
            if self.cell_type == 'lstm':
                self.cell_projection = nn.Linear(self.enc_hidden_dim, self.dec_hidden_dim)

    def _adapt_hidden_state(self, encoder_hidden):
        if self.cell_type == 'lstm':
            h, c = encoder_hidden
            if self.enc_n_layers != self.dec_n_layers:
                if self.enc_n_layers > self.dec_n_layers:
                    h = h[-self.dec_n_layers:]
                    c = c[-self.dec_n_layers:]
                else:
                    h = torch.cat([h] + [h[-1:].clone()] * (self.dec_n_layers - self.enc_n_layers), dim=0)
                    c = torch.cat([c] + [c[-1:].clone()] * (self.dec_n_layers - self.enc_n_layers), dim=0)
            if self.different_dims:
                h = self.hidden_projection(h)
                c = self.cell_projection(c)
            return (h, c)
        else:
            h = encoder_hidden
            if self.enc_n_layers != self.dec_n_layers:
                if self.enc_n_layers > self.dec_n_layers:
                    h = h[-self.dec_n_layers:]
                else:
                    h = torch.cat([h] + [h[-1:].clone()] * (self.dec_n_layers - self.enc_n_layers), dim=0)
            if self.different_dims:
                h = self.hidden_projection(h)
            return h

    def forward(self, src, trg, teacher_forcing=True):
        encoder_outputs, encoder_hidden = self.encoder(src)
        decoder_hidden = self._adapt_hidden_state(encoder_hidden)
        outputs, _ = self.decoder(trg, decoder_hidden, encoder_outputs, teacher_forcing=teacher_forcing)
        return outputs
def calculate_sequence_accuracy(preds, trg):
    pred_tokens = preds.argmax(-1)
    match = ((pred_tokens == trg[:, 1:]) | (trg[:, 1:] == 0)).all(dim=1)
    return match.float().mean()

def train(model, iterator, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    pbar = tqdm(iterator, desc="Training", leave=False)
    for src, trg in pbar:
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()
        output = model(src, trg, teacher_forcing=True)
        loss = criterion(output.view(-1, output.shape[-1]), trg[:, 1:].reshape(-1))
        acc = calculate_sequence_accuracy(output, trg)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        pbar.set_postfix(loss=loss.item(), acc=acc.item())
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion, device):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    with torch.no_grad():
        for src, trg in iterator:
            src, trg = src.to(device), trg.to(device)
            output = model(src, trg, teacher_forcing=False)
            loss = criterion(output.view(-1, output.shape[-1]), trg[:, 1:].reshape(-1))
            acc = calculate_sequence_accuracy(output, trg)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def train_seq2seq_model( batch_size=64,max_sequence_length=20,emb_dim=128,enc_hidden_dim=256,dec_hidden_dim=256,enc_layers=2,  dec_layers=3,dropout=0.3,cell_type='lstm',n_epochs=10,lr=0.001,use_attention=True,device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    train_data, dev_data, test_data = load_english_data()
    X_train, y_train, X_dev, y_dev, X_test, y_test, src_tokenizer, trg_tokenizer = preprocess_data(
        train_data, dev_data, test_data, max_sequence_length)

    train_loader = DataLoader(Seq2SeqDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(Seq2SeqDataset(X_dev, y_dev), batch_size=batch_size)
    test_loader = DataLoader(Seq2SeqDataset(X_test, y_test), batch_size=batch_size)

    input_dim = len(src_tokenizer.word_index) + 1
    output_dim = len(trg_tokenizer.word_index) + 1

    encoder = Encoder(input_dim, emb_dim, enc_hidden_dim, enc_layers, cell_type, dropout)
    decoder = Decoder(output_dim, emb_dim, enc_hidden_dim, dec_hidden_dim, dec_layers, cell_type, dropout, use_attention)

    model = Seq2Seq(encoder, decoder).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    best_valid_loss = float('inf')

    for epoch in range(n_epochs):
        print(f"Epoch {epoch+1}/{n_epochs}")
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        valid_loss, valid_acc = evaluate(model, dev_loader, criterion, device)
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {valid_loss:.4f} | Val Acc: {valid_acc:.4f}")

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'best_seq2seq_model.pt')
            print("Model saved!")

    model.load_state_dict(torch.load('best_seq2seq_model.pt'))

    return model, src_tokenizer, trg_tokenizer

def train_seq2seq_model_wandb(config=None):
   
    with wandb.init(config=config):
        config = wandb.config

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        train_data, dev_data, test_data = load_english_data()
        X_train, y_train, X_dev, y_dev, X_test, y_test, src_tokenizer, trg_tokenizer = preprocess_data(
            train_data, dev_data, test_data, config.max_sequence_length)

        train_loader = DataLoader(Seq2SeqDataset(X_train, y_train), batch_size=config.batch_size, shuffle=True)
        dev_loader = DataLoader(Seq2SeqDataset(X_dev, y_dev), batch_size=config.batch_size)
        test_loader = DataLoader(Seq2SeqDataset(X_test, y_test), batch_size=config.batch_size)

        input_dim = len(src_tokenizer.word_index) + 1
        output_dim = len(trg_tokenizer.word_index) + 1

        encoder = Encoder(input_dim=input_dim,emb_dim=config.emb_dim,hidden_dim=config.hidden_dim, n_layers=config.enc_layers,cell_type=config.cell_type,dropout=config.dropout)
        decoder = Decoder(output_dim=output_dim,emb_dim=config.emb_dim,enc_hidden_dim=config.hidden_dim,dec_hidden_dim=config.hidden_dim,n_layers=config.dec_layers,cell_type=config.cell_type,dropout=config.dropout,use_attention=True)
        model = Seq2Seq(encoder, decoder).to(device)
        optimizer = optim.Adam(model.parameters(), lr=config.lr)
        criterion = nn.CrossEntropyLoss(ignore_index=0)


        for epoch in range(config.n_epochs):
            print(f"Epoch {epoch+1}/{config.n_epochs}")
            train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
            valid_loss, valid_acc = evaluate(model, dev_loader, criterion, device)

            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {valid_loss:.4f} | Val Acc: {valid_acc:.4f}")

            wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': valid_loss,
                'val_acc': valid_acc
            })
        wandb.finish()



def visualize_test_sample_heatmaps(model, test_dataset, src_tokenizer, trg_tokenizer, device, num_samples=5):
    model.eval()

    font_path = "/root/.fonts/NotoSansMalayalam-VariableFont_wdth,wght.ttf" #needs fonts to be installed for visualization in malayalam

    if os.path.exists(font_path):
        malayalam_font = fm.FontProperties(fname=font_path)
        print(f" Malayalam font loaded: {malayalam_font.get_name()}")
    else:
        malayalam_font = None
        print(" Malayalam font file not found. Text rendering may be incorrect.")

    for i in range(min(num_samples, len(test_dataset))):
        src, trg = test_dataset[i*100]
        src = src.unsqueeze(0).to(device)
        trg = trg.unsqueeze(0).to(device)

        attention_matrix, src_tokens, trg_tokens = model.generate_attention_heatmap(
            src, trg, src_tokenizer, trg_tokenizer
        )

        if attention_matrix is None:
            print(f"Sample {i+1}: No attention matrix available.")
            continue

        print(f"\nSample {i+1} - Tokens:")
        print("Source:", src_tokens)
        print("Target:", trg_tokens)

        plt.figure(figsize=(12, 8))
        plt.imshow(attention_matrix, cmap='YlOrRd', aspect='auto')
        cbar = plt.colorbar()
        cbar.set_label('Attention Weight')

        plt.title(f"Attention Heatmap - Sample {i+1}", fontsize=14, fontproperties=malayalam_font)
        plt.xlabel("Source Tokens", fontsize=12, fontproperties=malayalam_font)
        plt.ylabel("Target Tokens", fontsize=12, fontproperties=malayalam_font)

        plt.xticks(
            ticks=np.arange(len(src_tokens)),
            labels=src_tokens,
            rotation=90,
            fontsize=10,
            fontproperties=malayalam_font
        )
        plt.yticks(
            ticks=np.arange(len(trg_tokens)),
            labels=trg_tokens,
            fontsize=10,
            fontproperties=malayalam_font
        )

        for y in range(attention_matrix.shape[0]):
            for x in range(attention_matrix.shape[1]):
                weight = attention_matrix[y, x]
                plt.text(
                    x, y, f'{weight:.2f}',
                    ha="center", va="center",
                    color="black" if weight < 0.5 else "white",
                    fontsize=8
                )

        plt.tight_layout()
        plt.show()


def create_attention_gif(input_seq, output_seq, attention_matrix, save_path="attention.gif"):
    """
    input_seq: str
    output_seq: str
    attention_matrix: (len(output_seq), len(input_seq)) — attention weights
    """

    font_path = "/root/.fonts/NotoSansMalayalam-VariableFont_wdth,wght.ttf" #needs fonts to be installed for visualization in malayalam

    if os.path.exists(font_path):
        malayalam_font = fm.FontProperties(fname=font_path)
        print(f" Malayalam font loaded: {malayalam_font.get_name()}")
    else:
        malayalam_font = None
        print(" Malayalam font file not found. Text rendering may be incorrect.")

    frames = []
    fig, ax = plt.subplots(figsize=(len(input_seq) * 0.6, 2))

    for t in range(len(output_seq)):
        ax.clear()
        ax.axis("off")

        for i, ch in enumerate(input_seq):
            weight = attention_matrix[t][i]
            color = (1, 0.7 - weight, 0.7 - weight)  
            ax.text(i, 1, ch, fontsize=16, ha='center', va='center',
                    bbox=dict(facecolor=color, edgecolor='black' if weight > 0.2 else 'none', boxstyle='round,pad=0.3'))

        for j, ch in enumerate(output_seq):
            color = "lightgreen" if j == t else "white"
            ax.text(j, 0, ch, fontsize=16, ha='center', va='center',
                    bbox=dict(facecolor=color, edgecolor='black', boxstyle='round,pad=0.3'))

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        buf.seek(0)
        img = Image.open(buf)
        frames.append(img.convert('RGB'))
        buf.close()

    frames[0].save(
        save_path, save_all=True, append_images=frames[1:], optimize=True, duration=700, loop=0
    )
    print(f"Saved GIF to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Seq2Seq model for Malayalam-English transliteration")

    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--max_sequence_length", type=int, default=20, help="Maximum sequence length")
    parser.add_argument("--emb_dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--enc_hidden_dim", type=int, default=256, help="Encoder hidden dimension")
    parser.add_argument("--dec_hidden_dim", type=int, default=256, help="Decoder hidden dimension")
    parser.add_argument("--enc_layers", type=int, default=2, help="Number of layers in the encoder")
    parser.add_argument("--dec_layers", type=int, default=3, help="Number of layers in the decoder")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--cell_type", type=str, choices=["lstm", "gru", "rnn"], default="lstm", help="RNN cell type")
    parser.add_argument("--n_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--use_attention", action='store_true', help="Enable attention mechanism")
    parser.add_argument("--no_attention", dest='use_attention', action='store_false', help="Disable attention mechanism")
    parser.set_defaults(use_attention=True)

    args = parser.parse_args()

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_seq2seq_model(
        batch_size=args.batch_size,
        max_sequence_length=args.max_sequence_length,
        emb_dim=args.emb_dim,
        enc_hidden_dim=args.enc_hidden_dim,
        dec_hidden_dim=args.dec_hidden_dim,
        enc_layers=args.enc_layers,
        dec_layers=args.dec_layers,
        dropout=args.dropout,
        cell_type=args.cell_type,
        n_epochs=args.n_epochs,
        lr=args.lr,
        use_attention=args.use_attention,
        device=args.device
    )
