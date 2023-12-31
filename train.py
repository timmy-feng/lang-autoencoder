import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import argparse
import yaml
import math
import os

from tqdm import tqdm

from utils.model import Autoencoder

UNK_ID, SOS_ID, EOS_ID, PAD_ID = 0, 1, 2, 3

parser = argparse.ArgumentParser()
parser.add_argument('config', type=str)
args = parser.parse_args()

config = yaml.safe_load(open(args.config, 'r'))

num_epochs = config['train']['epochs']

input_vocab_size = config['lang']['input']['vocab_size']
input_len = config['lang']['input']['seq_len']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Autoencoder(
    input_vocab_size = config['lang']['input']['vocab_size'],
    output_vocab_size = config['lang']['output']['vocab_size'],
    input_len = config['lang']['input']['seq_len'],
    output_len = config['lang']['output']['seq_len'],
    d_model = config['model']['d_model'],
    nhead = config['model']['nhead'],
    num_layers = config['model']['num_encoder_layers'],
    dim_feedforward = config['model']['dim_feedforward'],
    dropout = config['train']['dropout'],
    discrete = config['train']['discrete'],
    temperature = config['train']['temperature'],
)

if 'load_path' in config['model']:
    model.load_state_dict(torch.load(config['model']['load_path'], map_location=device))
    print('Model loaded from ' + config["model"]["load_path"])

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr = config['train']['lr'],
    weight_decay = config['train']['weight_decay'],
)

train_dataset = torch.load(config['dataset']['train'])
val_dataset = torch.load(config['dataset']['val'])
train_loader = DataLoader(train_dataset, batch_size = config['train']['batch_size'], shuffle = True)
val_loader = DataLoader(val_dataset, batch_size = config['train']['batch_size'])

if 'test' in config['dataset']:
    test_dataset = torch.load(config['dataset']['test'])
    test_loader = DataLoader(test_dataset, batch_size = config['train']['batch_size'])

model.to(device)
torch.nn.utils.clip_grad_norm_(model.parameters(), config['train']['clip_grad_norm'])
torch.autograd.set_detect_anomaly(True, check_nan = True)

for epoch in range(num_epochs):
    progress_bar = tqdm(total = len(train_loader), position = 0)
    train_loss_log = tqdm(total = 0, position = 1, bar_format = '{desc}')

    progress_bar.write(f'Epoch {epoch + 1}')

    model.train()

    for src, in train_loader:
        src = src[:, :input_len].to(device)
        src_one_hot = F.one_hot(src, input_vocab_size).float().to(device)

        output = model(src_one_hot)
        output = output.transpose(1, 2)

        loss = criterion(output, src)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        accuracy = ((torch.argmax(output, dim = 1) == src) * (src != PAD_ID)).sum() / (src != PAD_ID).sum()

        train_loss_log.set_description_str(f'Train loss: {loss.item():.4f}, '
            f'Train accuracy: {accuracy:.4f}')
        progress_bar.update(1)

    progress_bar.close()
    train_loss_log.close()

    model.eval()

    total_loss, total_samples, total_matches, total_tokens = 0, 0, 0, 0
    for src, in val_loader:
        src = src[:, :input_len].to(device)
        src_one_hot = F.one_hot(src, input_vocab_size).float().to(device)

        output = model(src_one_hot)
        output = output.transpose(1, 2)

        loss = criterion(output, src)

        total_loss += loss.detach().item() * src.size(0)
        total_samples += src.size(0)
        total_matches += ((torch.argmax(output, dim = 1) == src) * (src != PAD_ID)).sum()
        total_tokens += (src != PAD_ID).sum()

    print(f'Val loss: {(total_loss / total_samples):.4f}, '
        f'Val accuracy: {(total_matches / total_tokens):.4f}')
    print()

if 'test' in config['dataset']:
    total_loss, total_samples, total_matches, total_tokens = 0, 0, 0, 0
    for src, in test_loader:
        src = src[:, :input_len].to(device)
        src_one_hot = F.one_hot(src, input_vocab_size).float().to(device)

        output = model(src_one_hot)
        output = output.transpose(1, 2)

        loss = criterion(output, src)

        total_loss += loss.detach().item() * src.size(0)
        total_samples += src.size(0)
        total_matches += ((torch.argmax(output, dim = 1) == src) * (src != PAD_ID)).sum()
        total_tokens += (src != PAD_ID).sum()

    print(f'Train loss: {(total_loss / total_samples):.4f}, '
        f'Train accuracy: {(total_matches / total_tokens):.4f}')
    print()
    
save_path = config['model']['save_path']
save_dir = os.path.dirname(save_path)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
torch.save(model.state_dict(), save_path)
print(f'Model saved to {save_path}')
