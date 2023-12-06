import sentencepiece as spm
import argparse
from os import path

from torch.utils.data import TensorDataset, random_split
import torch

UNK_ID, SOS_ID, EOS_ID, PAD_ID = 0, 1, 2, 3

parser = argparse.ArgumentParser()
parser.add_argument('corpus', type=str)
parser.add_argument('--save_path', type=str)
parser.add_argument('--max_len', nargs='?', default=-1, type=int)
parser.add_argument('--vocab_size', nargs='?', default=2048, type=int)
parser.add_argument('--val_split', nargs='?', default=0.1, type=float)
parser.add_argument('--test_split', nargs='?', default=0.1, type=float)
args = parser.parse_args()

assert args.val_split + args.test_split < 1

spm.SentencePieceTrainer.train(
    input = args.corpus,
    model_prefix = path.join(args.save_path, 'tokenizer'),
    vocab_size = args.vocab_size,
    model_type = 'bpe',
    user_defined_symbols = ['<pad>'],
)

sp = spm.SentencePieceProcessor(model_file=path.join(args.save_path, 'tokenizer.model'))

with open(args.corpus) as f:
    sentences = []
    max_len = 0 if args.max_len == -1 else args.max_len - 2
    for sentence in f.read().split('\n'):
        if args.max_len != -1 and len(sentence) > max_len:
            continue
        tokens = sp.encode(sentence)
        max_len = max(max_len, len(tokens))
        sentences.append(tokens)

    # account for <s> and </s> tokens
    max_len += 2

    sentences = list(map(
        lambda sentence: [SOS_ID] + sentence + [EOS_ID] + (max_len - len(sentence) - 2) * [PAD_ID],
        sentences,
    ))

    print(f'Max sentence length: {max_len} tokens')

    dataset = TensorDataset(torch.tensor(sentences, dtype = torch.long))

    val_size = int(len(dataset) * args.val_split)
    test_size = int(len(dataset) * args.test_split)
    train_size = len(dataset) - val_size - test_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    torch.save(train_dataset, path.join(args.save_path, 'train.pt'))
    torch.save(val_dataset, path.join(args.save_path, 'val.pt'))
    torch.save(test_dataset, path.join(args.save_path, 'test.pt'))

    dataset_save_path = path.join(args.save_path, 'train.pt')
    print(f'Saving datasets to {dataset_save_path},val.pt,test.pt')