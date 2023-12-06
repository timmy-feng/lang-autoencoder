#!/bin/bash
python wikitext/load.py
python utils/tokenizer.py wikitext/datasets/wikitext.txt --save_path wikitext/datasets --max_len 64