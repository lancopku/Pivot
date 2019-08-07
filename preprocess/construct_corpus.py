import sys
sys.path.append('../')
import re
import json
import os
from typing import List, Dict, Iterator
from utils.jsonl import loads, dumps
import argparse

parser = argparse.ArgumentParser(description='construct.py')
parser.add_argument('-parallel_size', type=int, default=10000, help="Parallel size")
parser.add_argument('-mode', type=str, default='train', help="Training set or testing set")
opt = parser.parse_args()

data_path = os.path.expanduser('~/data/wiki2bio/')

def join(path: str) -> str:
    return os.path.join(data_path, path)

def partion_list(data: List, index: List) -> List:
    return [data[i] for i in index]

def get_partion_index(limit: int) -> List[int]:
    index = list(map(int, open(join('shuffle_index.txt')).read().strip().split('\n')))
    return index[:limit]


def train_parallel_dataset(r_path: str, w_path: str, index: List[int]) -> List[Dict]:
    '''
    source: value;
    target: text.
    '''
    ori_datas = loads(open(r_path))[1:]
    ori_datas = partion_list(ori_datas, index)

    statistic = {'length': len(ori_datas)}
    datas = [statistic]

    for d in ori_datas:
        data = {'source': d['value'], 'target': d['text'], 'field': d['field'], 'lpos': d['lpos'], 'rpos': d['rpos']}
        datas.append(data)
    
    dumps(datas, open(w_path, 'w'))


def train_t2p_dataset(r_path: str, w_path: str, index: List[int]) -> List[Dict]:
    '''
    value, label, field, lpos, rpos
    '''
    ori_datas = loads(open(r_path))[1:]
    ori_datas = partion_list(ori_datas, index)

    statistic = {'length': len(ori_datas)}
    datas = [statistic]

    for d in ori_datas:
        data = {'value': d['value'], 'label': d['label'], 'field': d['field'],
                'lpos': d['lpos'], 'rpos': d['rpos']}
        datas.append(data)
    
    dumps(datas, open(w_path, 'w'))

def get_filter_data(data: Dict) -> Dict:
    _data = {}
    label = data['label'].split(' ')
    field = data['field'].split(' ')
    lpos = data['lpos'].split(' ')
    rpos = data['rpos'].split(' ')
    _field = [f for f, l in zip(field, label) if l == '1']
    _lpos = [f for f, l in zip(lpos, label) if l == '1']
    _rpos = [f for f, l in zip(rpos, label) if l == '1']

    return {'source': data['pivot'], 'target': data['text'],
            'field': ' '.join(_field), 'lpos': ' '.join(_lpos),
            'rpos': ' '.join(_rpos)}

def train_p2t_dataset(r_path: str, w_path: str, index: List[int]) -> List[Dict]:
    '''
    value, text, field, lpos, rpos, pivot, entity
    '''
    ori_datas = loads(open(r_path))[1:]

    statistic = {'length': len(ori_datas)}
    datas = [statistic]
    index = set(index)

    for i, d in enumerate(ori_datas):
        if i in index:
            #datas.append({'source': d['pivot'], 'target': d['text']})
            datas.append(get_filter_data(d))
        else:
            datas.append({'source': d['entity'], 'target': d['text']})
    
    dumps(datas, open(w_path, 'w'))

def train_aug_dataset(r_path: str, w_path: str, index: List[int]) -> List[Dict]:
    '''
    value, text, field, lpos, rpos, pivot, entity
    '''
    ori_datas = loads(open(r_path))[1:]

    statistic = {'length': len(ori_datas)}
    datas = [statistic]
    index = set(index)

    for i, d in enumerate(ori_datas):
        if i in index:
            datas.append({'source': d['value'], 'target': d['text'], 'field': d['field'],
                          'lpos': d['lpos'], 'rpos': d['rpos']})
        else:
            datas.append({'source': d['entity'], 'target': d['text']})
    
    dumps(datas, open(w_path, 'w'))

def train_semi_dataset(r_path: str, w_path: str, index: List[int]) -> List[Dict]:
    '''
    value, text, field, lpos, rpos, pivot, entity
    '''
    ori_datas = loads(open(r_path))[1:]

    statistic = {'length': len(ori_datas)}
    datas = [statistic]
    index = set(index)

    for i, d in enumerate(ori_datas):
        if i in index:
            datas.append({'source': d['value'], 'target': d['text'], 'field': d['field'],
                            'lpos': d['lpos'], 'rpos': d['rpos']})
        else:
            datas.append({'source': d['text'], 'target': d['text']})
    
    dumps(datas, open(w_path, 'w'))

def train_pretrain_dataset(r_path: str, w_path: str, index: List[int]) -> List[Dict]:
    '''
    value, text, field, lpos, rpos, pivot, entity
    '''
    ori_datas = loads(open(r_path))[1:]

    statistic = {'length': len(ori_datas)}
    datas = [statistic]
    index = set(index)

    for i, d in enumerate(ori_datas):
        if i in index:
            datas.append({'source': d['value'], 'target': d['text'], 'field': d['field'],
                            'lpos': d['lpos'], 'rpos': d['rpos']})
        else:
            datas.append({'source': '', 'target': d['text']})
    
    dumps(datas, open(w_path, 'w'))

def test_parallel_dataset(r_path: str, w_path: str) -> List[Dict]:
    '''
    source: value;
    target: text.
    '''
    ori_datas = loads(open(r_path))[1:]

    statistic = {'length': len(ori_datas)}
    datas = [statistic]

    for d in ori_datas:
        data = {'source': d['value'], 'target': d['text'], 'field': d['field'], 'lpos': d['lpos'], 'rpos': d['rpos']}
        datas.append(data)
    
    dumps(datas, open(w_path, 'w'))


def test_t2p_dataset(r_path: str, w_path: str) -> List[Dict]:
    '''
    value, label, field, lpos, rpos
    '''
    ori_datas = loads(open(r_path))[1:]

    statistic = {'length': len(ori_datas)}
    datas = [statistic]

    for d in ori_datas:
        data = {'value': d['value'], 'label': d['label'], 'field': d['field'],
                'lpos': d['lpos'], 'rpos': d['rpos']}
        datas.append(data)
    
    dumps(datas, open(w_path, 'w'))


def test_p2t_dataset(r_path: str, w_path: str) -> List[Dict]:
    '''
    value, text, field, lpos, rpos, pivot, entity
    '''
    ori_datas = loads(open(r_path))[1:]

    statistic = {'length': len(ori_datas)}
    datas = [statistic]

    for d in ori_datas:
        datas.append({'source': d['pivot'], 'target': d['text']})
    
    dumps(datas, open(w_path, 'w'))



if __name__ == '__main__':
    indexes = get_partion_index(opt.parallel_size)

    if opt.mode == 'train':
        train_parallel_dataset(join('train.pivot.jsonl'), join('train.parallel.{0}.jsonl'.format(opt.parallel_size)), index=indexes)
        train_p2t_dataset(join('train.pivot.jsonl'), join('train.p2t.{0}.jsonl'.format(opt.parallel_size)), index=indexes)
        train_t2p_dataset(join('train.pivot.jsonl'), join('train.t2p.{0}.jsonl'.format(opt.parallel_size)), index=indexes)
        train_aug_dataset(join('train.pivot.jsonl'), join('train.aug.{0}.jsonl'.format(opt.parallel_size)), index=indexes)
        train_semi_dataset(join('train.pivot.jsonl'), join('train.semi.{0}.jsonl'.format(opt.parallel_size)), index=indexes)
        train_pretrain_dataset(join('train.pivot.jsonl'), join('train.pretrain.{0}.jsonl'.format(opt.parallel_size)), index=indexes)
    else:
        test_parallel_dataset(join('test.pivot.jsonl'), join('test.parallel.jsonl'))
        test_p2t_dataset(join('test.pivot.jsonl'), join('test.p2t.jsonl'))
        test_t2p_dataset(join('test.pivot.jsonl'), join('test.t2p.jsonl'))

        test_parallel_dataset(join('valid.pivot.jsonl'), join('valid.parallel.jsonl'))
        test_p2t_dataset(join('valid.pivot.jsonl'), join('valid.p2t.jsonl'))
        test_t2p_dataset(join('valid.pivot.jsonl'), join('valid.t2p.jsonl'))
    