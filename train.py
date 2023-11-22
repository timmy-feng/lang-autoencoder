import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import argparse
import yaml
from tqdm import tqdm

from model import TransformerEncoder, Transformer, Autoencoder

parser = argparse.ArgumentParser()
parser.add_argument('config')
args = parser.parse_args()

config = yaml.safe_load(open(args.config, 'r'))

input_vocab_size = config['lang']['input']['vocab_size']
output_vocab_size = config['lang']['output']['vocab_size']

input_max_len = config['lang']['input']['max_seq_len']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

src_embed = nn.Linear(input_vocab_size, config['model']['d_model'], bias = False)
conlang_embed = nn.Linear(output_vocab_size, config['model']['d_model'], bias = False)

conlang_encoder = TransformerEncoder(
    input_vocab_size = input_vocab_size,
    output_vocab_size = output_vocab_size,
    output_seq_len = config['lang']['output']['max_seq_len'],
    input_embed = src_embed,
    d_model = config['model']['d_model'],
    nhead = config['model']['nhead'],
    num_layers = config['model']['num_encoder_layers'],
    dim_feedforward = config['model']['dim_feedforward'],
    temperature = config['model']['temperature'],
)

conlang_decoder = Transformer(
    input_vocab_size = output_vocab_size,
    output_vocab_size = input_vocab_size,
    input_embed = conlang_embed,
    output_embed = src_embed,
    d_model = config['model']['d_model'],
    nhead = config['model']['nhead'],
    num_encoder_layers = config['model']['num_encoder_layers'],
    num_decoder_layers = config['model']['num_decoder_layers'],
    dim_feedforward = config['model']['dim_feedforward'],
)

model = Autoencoder(conlang_encoder, conlang_decoder)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = config['train']['lr'])

train_dataset = torch.load(config['dataset']['train'])
train_loader = DataLoader(train_dataset, batch_size = config['train']['batch_size'], shuffle = True)

model.to(device)
torch.nn.utils.clip_grad_norm_(model.parameters(), config['train']['clip_grad_norm'])

for epoch in range(config['train']['epochs']):
    progress_bar = tqdm(total = len(train_loader), position = 0)
    train_loss_log = tqdm(total = 0, position = 1, bar_format = '{desc}')

    progress_bar.write(f'Epoch {epoch + 1}')

    for src, in train_loader:
        src = src[:, :input_max_len].to(device)
        src_one_hot = F.one_hot(src, input_vocab_size).float().to(device)

        output = model(src_one_hot)
        output = output.transpose(1, 2)

        src = src[:, 1:]
        loss = criterion(output, src)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        accuracy = ((torch.argmax(output, dim = 1) == src).sum() - (src == 0).sum()) / (src != 0).sum()

        train_loss_log.set_description_str(f'Train loss: {loss.item():.4f}, Train accuracy: {accuracy:.4f}')
        progress_bar.update(1)