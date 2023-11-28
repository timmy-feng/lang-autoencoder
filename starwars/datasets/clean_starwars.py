import re

filename = 'SW_Complete.txt'
with open(filename, 'r') as f:
    text = f.read()

# extract dialouge from each line
# "line no" "character" "dialouge"
cleaned_text = '\n'.join(map(lambda line: line.split('\"')[5], text.split('\n')))

# make each sentence a line
sentences = re.split(r'(?<=[.?!\n])', cleaned_text)

# filter short sentences ( <= 3 words )
filtered_sentences = filter(lambda sentence: len(sentence.split()) > 3, sentences)

# strip and lower
text = '\n'.join(map(lambda sentence: sentence.strip().lower(), filtered_sentences))

with open(filename, 'w') as f:
    f.write(text)