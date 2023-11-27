import re

import pickle

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from torch.utils.data import TensorDataset, Dataset, random_split
import torch
import tensorflow as tf
import keras
from keras import layers

UNK_THRESHOLD = 5

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3

MAX_SEQ_LEN = 135
MAX_VOCAB_SIZE = 700

VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

counts = {}

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
    for w in tokens:
        if w not in counts:
            counts[w] = 1
        else:
            counts[w] += 1
    return tokens

def replace_unk(tokens):
    for i, w in enumerate(tokens):
        if counts[w] <= UNK_THRESHOLD:
            tokens[i] = '<unk>'

def get_word_dict(tokens):
    result = {
        '<pad>': PAD_IDX,
        '<sos>': SOS_IDX,
        '<eos>': EOS_IDX,
        '<unk>': UNK_IDX,
    }

    for token in tokens:
        if token not in result:
            result[token] = len(result)
    return result

def replace_words(tokens, word_dict):
    return [SOS_IDX] + [word_dict[token] for token in tokens] + [EOS_IDX]

def pad(tokens, length):
    return tokens + [PAD_IDX] * (length - len(tokens))

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = torch.from_numpy(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

if __name__ == '__main__':
    path = 'starwars/datasets/'
    sentences = []
    for name in ['SW_EpisodeIV.txt', 'SW_EpisodeV.txt', 'SW_EpisodeVI.txt']:
        text = open(path + name, 'r').read()
        sentences += [line_to_sentence(line) for line in text.split('\n')[1: -1]]
        print (sentences[-5:])
        print ()
    
    text_vectorizer = layers.TextVectorization(
        max_tokens=MAX_VOCAB_SIZE,
        standardize='lower_and_strip_punctuation',
        split='whitespace',
        ngrams=None,
        output_mode='int',
        output_sequence_length=None,
        pad_to_max_tokens=False,
        vocabulary=None,
        idf_weights=None,
        sparse=False,
        ragged=False,
        encoding='utf-8',
    )
    text_vectorizer.adapt(sentences)
    indexed_text = text_vectorizer(sentences).numpy()

    print (sentences[:3])
    print (indexed_text[: 3])

    vocabulary = text_vectorizer.get_vocabulary()
    word_to_index = {word: index for index, word in enumerate(vocabulary)}

    print (word_to_index)

    dataset = TensorDataset(torch.tensor(indexed_text, dtype = torch.long))

    val_size = int(len(dataset) * VAL_SPLIT)
    test_size = int(len(dataset) * TEST_SPLIT)
    train_size = len(dataset) - val_size - test_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    torch.save(train_dataset, 'starwars/datasets/keras_tokenizer/train.pt')
    torch.save(val_dataset, 'starwars/datasets/keras_tokenizer/val.pt')
    torch.save(test_dataset, 'starwars/datasets/keras_tokenizer/test.pt')

    with open('starwars/datasets/keras_tokenizer/word_dict.pkl', 'wb') as f:
        pickle.dump(word_to_index, f)
    