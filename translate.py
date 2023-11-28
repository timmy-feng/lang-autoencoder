import sentencepiece as spm
import argparse
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.model import TransformerEncoder, Transformer, Autoencoder

UNK_ID, SOS_ID, EOS_ID, PAD_ID = 0, 1, 2, 3

parser = argparse.ArgumentParser()
parser.add_argument('config', type=str)
args = parser.parse_args()

config = yaml.safe_load(open(args.config, 'r'))

input_vocab_size = config['lang']['input']['vocab_size']
output_vocab_size = config['lang']['output']['vocab_size']

input_max_len = config['lang']['input']['seq_len']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

src_embed = nn.Linear(input_vocab_size, config['model']['d_model'], bias = False)
lang_embed = nn.Linear(output_vocab_size, config['model']['d_model'], bias = False)

lang_encoder = TransformerEncoder(
    input_vocab_size = input_vocab_size,
    output_vocab_size = output_vocab_size,
    output_len = config['lang']['output']['seq_len'],
    input_embed = src_embed,
    d_model = config['model']['d_model'],
    nhead = config['model']['nhead'],
    num_layers = config['model']['num_encoder_layers'],
    dim_feedforward = config['model']['dim_feedforward'],
    dropout = config['train']['dropout'],
)

lang_decoder = Transformer(
    input_vocab_size = output_vocab_size,
    output_vocab_size = input_vocab_size,
    input_embed = lang_embed,
    output_embed = src_embed,
    d_model = config['model']['d_model'],
    nhead = config['model']['nhead'],
    num_encoder_layers = config['model']['num_encoder_layers'],
    num_decoder_layers = config['model']['num_decoder_layers'],
    dim_feedforward = config['model']['dim_feedforward'],
    dropout = config['train']['dropout'],
)

model = Autoencoder(
    lang_encoder,
    lang_decoder,
    temperature = config['model']['temperature'],
    discrete = config['train']['discrete'],
)

model.load_state_dict(torch.load(config['model']['path']))
sp = spm.SentencePieceProcessor(model_file=config['predict']['tokenizer'])

model.eval()

print('Model loaded. Ready for inference.\n')

outputs = {
    'tokenized': False,
    'translation': True,
    'backtranslation': False,
    'bleu': False,
    'nll': False,
}

while True:
    print('>>> ', end = '')
    sentence = input().strip()
    print()

    # empty line; skip
    if len(sentence) == 0:
        continue

    # toggling option
    if sentence[0] == '!':
        sentence = sentence.strip('! ')
        if sentence == 'quit':
            exit()
        if sentence not in outputs:
            options = list(outputs.keys()) + ['quit']
            print(f'Invalid option. Options include: {options}')
        else:
            outputs[sentence] = not outputs[sentence]
            print(f'Options updated: {outputs}\n')
        continue

    tokens = sp.encode(sentence)
    src = torch.tensor([[SOS_ID] + tokens + [EOS_ID] + (input_max_len - len(tokens) - 2) * [PAD_ID]])
    src = F.one_hot(src, input_vocab_size).float().to(device)

    logits = model.encoder(src)
    lang = model.softmax(logits)
    result = model.decoder(lang, src[:, :-1])

    if outputs['tokenized']:
        print(f'Tokenized: {tokens}')

    if outputs['translation']:
        print(f'Translation: {list(map(int,torch.argmax(lang[0], dim=1)))}')
        print()

    if outputs['backtranslation']:
        backtranslation = list(
            filter(lambda token: token != PAD_ID, 
                map(int,torch.argmax(result[0], dim=1)))
        )
        print(f'Backtranslation: {sp.decode(backtranslation)}')
        print()