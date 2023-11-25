import re

import pickle

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from torch.utils.data import TensorDataset, random_split
import torch

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2

MAX_SEQ_LEN = 128

VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

def line_to_sentence(line):
    num_quotes = 0
    for i, char in enumerate(line):
        if num_quotes == 5:
            line = line[i:]
            break
        if char == '"':
            num_quotes += 1
    
    return line[:-1]

def tokenize(sentence):
    sentence = re.sub("[^a-zA-Z'-']", " ", sentence)
    sentence = sentence.lower()
    #tokens = word_tokenize(sentence)
    tokens = sentence.split()
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return tokens

def get_word_dict(tokens):
    result = {
        '<pad>': PAD_IDX,
        '<sos>': SOS_IDX,
        '<eos>': EOS_IDX,
    }

    for token in tokens:
        if token not in result:
            result[token] = len(result)
    return result

def replace_words(tokens, word_dict):
    return [SOS_IDX] + [word_dict[token] for token in tokens] + [EOS_IDX]

def pad(tokens, length):
    return tokens + [PAD_IDX] * (length - len(tokens))

if __name__ == '__main__':
    text = open('datasets/starwars/SW_EpisodeIV.txt', 'r').read()

    tokenized_text = [tokenize(line_to_sentence(line)) for line in text.split('\n')[1:]]
    word_dict = get_word_dict([token for line in tokenized_text for token in line])
    indexed_text = [pad(replace_words(line, word_dict), MAX_SEQ_LEN) for line in tokenized_text]

    print(tokenized_text[0:3])

    dataset = TensorDataset(torch.tensor(indexed_text, dtype = torch.long))

    val_size = int(len(dataset) * VAL_SPLIT)
    test_size = int(len(dataset) * TEST_SPLIT)
    train_size = len(dataset) - val_size - test_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    torch.save(train_dataset, 'datasets/starwars/ep4/train.pt')
    torch.save(val_dataset, 'datasets/starwars/ep4/val.pt')
    torch.save(test_dataset, 'datasets/starwars/ep4/test.pt')

    with open('datasets/starwars/ep4/word_dict.pkl', 'wb') as f:
        pickle.dump(word_dict, f)