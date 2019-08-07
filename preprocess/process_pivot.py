import sys
sys.path.append('../')
import re
import json
import os
from typing import List, Dict, Iterator
from utils.jsonl import loads, dumps
import random

data_path = os.path.expanduser('~/data/wiki2bio/')

def join(path: str) -> str:
    return os.path.join(data_path, path)

def partion_list(data: List, index: List) -> List:
    return [data[i] for i in index]

def get_partion_index(path: str, limit: int) -> List[int]:
    ori_datas = loads(open(path))[1:]
    index = list(range(len(ori_datas)))
    random.shuffle(index)
    return index[:limit]

def write_into_file(path: str, datas: List) -> None:
    fout = open(path, 'w')
    for data in datas:
        print(data, end='\n', file=fout)
    fout.close()

def construct_table2pivot(path: str, index: List[int]=None) -> List[Dict]:

    ori_datas = loads(open(path))[1:]
    if index is not None:
        ori_datas = partion_list(ori_datas, index)

    statistic = {'length': len(ori_datas)}
    datas = [statistic]

    count = 0
    src_len, tgt_len = [], []
    
    for d in ori_datas:
        data = {'value': d['value'], 'label': d['label'], 'field': d['field'], 'lpos': d['lpos'], 'rpos': d['rpos']}
        datas.append(data)
        src_len.append(len(data['value'].split(' ')))
        tgt_len.append(len(data['label'].split(' ')))
        count += 1
        if count % 100000 == 0:
            print(count)
    
    print('max len: ', max(src_len), max(tgt_len))
    print('avg len: ', sum(src_len)*1.0/len(src_len), sum(tgt_len)*1.0/len(tgt_len))
    
    return datas


def construct_pivot2text(path: str, index: List[int]=None) -> List[Dict]:

    ori_datas = loads(open(path))[1:]
    if index is not None:
        index = set(index)

    statistic = {'length': len(ori_datas)}
    datas = [statistic]

    count = 0
    src_len, tgt_len = [], []
    
    for i, d in enumerate(ori_datas):
        if index is None or i in index:
            data = {'source': d['pivot'], 'target': d['text']}
        else:
            data = {'source': d['entity'], 'target': d['text']}
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
    indexes = get_partion_index(join('train.jsonl'), 10000)
    write_into_file(join('index.txt'), indexes)

    train_t2p_datas = construct_table2pivot(join('train.pivot.jsonl'), indexes)
    test_t2p_datas = construct_table2pivot(join('test.pivot.jsonl'))
    valid_t2p_datas = construct_table2pivot(join('valid.pivot.jsonl'))

    train_p2t_datas = construct_pivot2text(join('train.pivot.jsonl'), indexes)
    test_p2t_datas = construct_pivot2text(join('test.pivot.jsonl'))
    valid_p2t_datas = construct_pivot2text(join('valid.pivot.jsonl'))

    dumps(train_t2p_datas, open(os.path.join(data_path, 'train.t2p.jsonl'), 'w'))
    dumps(test_t2p_datas, open(os.path.join(data_path, 'test.t2p.jsonl'), 'w'))
    dumps(valid_t2p_datas, open(os.path.join(data_path, 'valid.t2p.jsonl'), 'w'))

    dumps(train_p2t_datas, open(os.path.join(data_path, 'train.p2t.jsonl'), 'w'))
    dumps(test_p2t_datas, open(os.path.join(data_path, 'test.p2t.jsonl'), 'w'))
    dumps(valid_p2t_datas, open(os.path.join(data_path, 'valid.p2t.jsonl'), 'w'))