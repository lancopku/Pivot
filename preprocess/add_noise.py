import sys
sys.path.append('../')
import re
import json
import os
from typing import List, Dict, Iterator
from utils.jsonl import loads, dumps
import random

data_path = os.path.expanduser('~/data/wiki2bio/')
vocabulary = open(os.path.join(data_path, 'dicts-20000-noshare-p2t', 'tokens.txt')).read().strip().split('\n')

def join(path: str) -> str:
    return os.path.join(data_path, path)

def add_noise(line: str, noise_prob: float) -> str:
    words = line.split(' ')
    words = [w if random.random()>noise_prob else \
                vocabulary[random.randint(0, len(vocabulary)-1)] for w in words]
    words = ' '.join(words)
    return words


def construct_noise_corpus(path: str, noise_prob: float) -> List[Dict]:

    ori_datas = loads(open(path))[1:]

    statistic = {'length': len(ori_datas)}
    datas = [statistic]

    count = 0
    src_len, tgt_len = [], []
    
    for d in ori_datas:
        data = {'source': add_noise(d['source'], noise_prob), 'target': d['target']}
        datas.append(data)
        src_len.append(len(data['source'].split(' ')))
        tgt_len.append(len(data['target'].split(' ')))
        count += 1
        if count % 100000 == 0:
            print(count)
    
    print('max len: ', max(src_len), max(tgt_len))
    print('avg len: ', sum(src_len)*1.0/len(src_len), sum(tgt_len)*1.0/len(tgt_len))
    
    return datas


if __name__ == '__main__':
    train_noise_datas = construct_noise_corpus(join('train.p2t.jsonl'), noise_prob=0.2)
    dumps(train_noise_datas, open(os.path.join(data_path, 'train.noise.jsonl'), 'w'))