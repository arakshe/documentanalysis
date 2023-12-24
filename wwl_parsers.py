"""
Parsers to support the WorldWarLetters Framework
"""

import json
from collections import Counter

def json_parser(filename):
    f = open(filename)
    raw = json.load(f)
    text = raw['text']
    words = text.split(" ")
    wc = Counter(words)
    num = len(wc)
    word_lengths = Counter(len(word) for word in text.split())
    f.close()
    return {'wordcount': wc, 'numwords':num, 'wordlength':word_lengths}
