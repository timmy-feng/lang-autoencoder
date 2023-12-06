import hashlib
import re, os
import tarfile
from pathlib import Path

import requests
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download the dataset
fname = 'wikitext-103.tar.gz'
url = 'https://dax-cdn.cdn.appdomain.cloud/dax-wikitext-103/1.0.1/' + fname
r = requests.get(url)
Path(fname).write_bytes(r.content)

# Verify the file was downloaded properly by comparing sha512 checksums
sha512sum = 'c8186919aa1840af6b734ea41abc580574ea8efe2fafda220f5d01002464d17566d84be5199b875136c9593f0e0678fb5d7c84bb2231de8b4151cb9c83fa2109'
sha512sum_computed = hashlib.sha512(Path('wikitext-103.tar.gz').read_bytes()).hexdigest()
assert sha512sum == sha512sum_computed

# Extract the dataset
with tarfile.open(fname) as tar:
    tar.extractall()

# Read train, val, and test sets into string objects
train_data = Path('wikitext-103/wiki.train.tokens').read_text()
val_data = Path('wikitext-103/wiki.valid.tokens').read_text()
test_data = Path('wikitext-103/wiki.test.tokens').read_text()

# Store regular expression pattern to search for wikipedia article headings
heading_pattern = '( \n \n = [^=]*[^=] = \n \n )'
# Split out train headings and articles
train_split = re.split(heading_pattern, train_data)
train_headings = [x[7:-7] for x in train_split[1::2]]
train_articles = [x for x in train_split[2::2]]

# Split out validation headings and articles
val_split = re.split(heading_pattern, val_data)
val_headings = [x[7:-7] for x in val_split[1::2]]
val_articles = [x for x in val_split[2::2]]

# Split out test headings and articles
test_split = re.split(heading_pattern, test_data)
test_headings = [x[7:-7] for x in test_split[1::2]]
test_articles = [x for x in test_split[2::2]]

for i in range (len(train_articles)):
    train_articles[i] = re.sub("[^A-Za-z0-9 .,?!/()'-]+", '',  train_articles[i]).lower()

my_new_text = re.sub('[^ a-zA-Z0-9]|unk', '', train_data[:2010011])
stop_words = set(stopwords.words('english'))
lemma = WordNetLemmatizer()
word_tokens = word_tokenize(my_new_text.lower())
filtered_sentence = (w for w in word_tokens if w not in stop_words)
normalized = " ".join(lemma.lemmatize(word) for word in filtered_sentence)

sentences = []

for example in train_articles:
    prev_ind = 0
    for i, char in enumerate(example):
        if char == "." or char == "?" or char == "!":
            if i - 1 > 0 and example[i-1] == " ":
                sentences.append(example[prev_ind: i + 1])
                prev_ind = i + 2

os.makedirs('wikitext/datasets', exist_ok = True)
with open("wikitext/datasets/full_wikitext.txt", "w") as f:
    for sentence in sentences:
        f.write(sentence + "\n")