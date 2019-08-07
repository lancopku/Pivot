import sys
sys.path.append('../')
import re
import json
import os
from typing import List, Dict, Iterator
from utils.jsonl import loads, dumps
import random

data_path = os.path.expanduser('~/data/wiki2bio/')
index = list(map(int, open(os.path.join(data_path, 'index.txt')).read().strip().split('\n')))

def join(path: str) -> str:
    return os.path.join(data_path, path)

def partion_list(data: List, index: List) -> List:
    return [data[i] for i in index]

def construct_super_corpus(path: str) -> List[Dict]:

    ori_datas = loads(open(path))[1:]
    ori_datas = partion_list(ori_datas, index)

    statistic = {'length': len(ori_datas)}
    datas = [statistic] + ori_datas
    
    return datas


if __name__ == '__main__':
    train_noise_datas = construct_super_corpus(join('train.jsonl'))
    dumps(train_noise_datas, open(os.path.join(data_path, 'train.sub.jsonl'), 'w'))