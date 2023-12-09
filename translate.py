import sentencepiece as spm
import argparse
import yaml

import torch
import torch.nn.functional as F

from utils.model import Autoencoder

UNK_ID, SOS_ID, EOS_ID, PAD_ID = 0, 1, 2, 3

parser = argparse.ArgumentParser()
parser.add_argument('config', type=str)
args = parser.parse_args()

config = yaml.safe_load(open(args.config, 'r'))

num_epochs = config['train']['epochs']

input_vocab_size = config['lang']['input']['vocab_size']
output_vocab_size = config['lang']['output']['vocab_size']
input_len = config['lang']['input']['seq_len']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Autoencoder(
    input_vocab_size = config['lang']['input']['vocab_size'],
    output_vocab_size = config['lang']['output']['vocab_size'],
    input_len = config['lang']['input']['seq_len'],
    output_len = config['lang']['output']['seq_len'],
    d_model = config['model']['d_model'],
    nhead = config['model']['nhead'],
    num_encoder_layers = config['model']['num_encoder_layers'],
    num_decoder_layers = config['model']['num_encoder_layers'],
    dim_feedforward = config['model']['dim_feedforward'],
    dropout = 0,
    discrete = True,
    temperature = config['predict']['temperature'],
)

model.load_state_dict(torch.load(config['model']['save_path'], map_location=device))
sp = spm.SentencePieceProcessor(model_file=config['predict']['tokenizer'])
model.to(device)
model.eval()

print('Model loaded. Ready for inference.\n')

outputs = {
    'tokenized': False,
    'translation': True,
    'backtranslation': True,
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
    src = torch.tensor([[SOS_ID] + tokens + [EOS_ID] + (input_len - len(tokens) - 2) * [PAD_ID]])
    src = F.one_hot(src, input_vocab_size).float().to(device)

    translation = model.translate(src)
    translation_one_hot = F.one_hot(translation, output_vocab_size).float()
    backtranslation = model.backtranslate(translation_one_hot, input_len)

    if outputs['tokenized']:
        print(f'Tokenized: ', end = '')
        print(*tokens)
        print()

    if outputs['translation']:
        print(f'Translation: ', end = '')
        print(*map(int, translation[0]))
        print()

    if outputs['backtranslation']:
        backtranslation = list(filter(lambda token: token != PAD_ID, map(int, backtranslation[0])))
        print(f'Backtranslation: {sp.decode(backtranslation)}\n')