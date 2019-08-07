import sys
sys.path.append('../')
import re
import json
import os
from typing import List, Dict, Iterator
from utils import jsonl

data_path = os.path.expanduser('~/data/wiki2bio/')

def extract_pivot(data: Dict) -> Dict:
    value = data['source'].split(' ')
    field = data['field'].split(' ')
    text = data['target'].split(' ')

    text = set(text)
    selected_field = {}
    for f, v in zip(field, value):
        if v in text:
            selected_field[f] = True
            
    labels, entity = [], []
    for f, v in zip(field, value):
        if f in selected_field:
            labels.append('1')
            entity.append(v)
        else:
            labels.append('0')

    data['label'] = ' '.join(labels)
    data['pivot'] = ' '.join(entity)
    return data


def write_into_file(path: str, datas: List[Dict]) -> None:
    fout = open(path, 'w')
    for data in datas[1:]:
        print(data['pivot'], end='\n', file=fout)
    fout.close()


def transform(path: str) -> List[Dict]:
    box_values = open(path+'.box.val', 'r').read().strip().split('\n')
    box_fields = open(path+'.box.lab', 'r').read().strip().split('\n')
    box_lpos = open(path+'.box.pos', 'r').read().strip().split('\n')
    box_rpos = open(path+'.box.rpos', 'r').read().strip().split('\n')
    texts = open(path+'.text', 'r').read().strip().split('\n')

    statistic = {'length': len(texts)}
    datas = [statistic]

    count = 0
    src_len, tgt_len = [], []
    
    for v, f, l, r, t in zip(box_values, box_fields, box_lpos, box_rpos, texts):
        data = {'source': v.strip(), 'target': t.strip(), 'field': f.strip(), 'lpos': l.strip(), 'rpos': r.strip()}
        data = extract_pivot(data)
        datas.append(data)
        src_len.append(len(data['source'].split(' ')))
        tgt_len.append(len(data['target'].split(' ')))
        count += 1
        if count % 10000 == 0:
            print(count)
    
    print('max len: ', max(src_len), max(tgt_len))
    print('avg len: ', sum(src_len)*1.0/len(src_len), sum(tgt_len)*1.0/len(tgt_len))
    
    return datas


if __name__ == '__main__':
    train_datas = transform(os.path.join(data_path, 'train'))
    test_datas = transform(os.path.join(data_path, 'test'))
    valid_datas = transform(os.path.join(data_path, 'valid'))

    write_into_file(os.path.join(data_path, 'train.pivot'), train_datas)
    write_into_file(os.path.join(data_path, 'test.pivot'), test_datas)
    write_into_file(os.path.join(data_path, 'valid.pivot'), valid_datas)

    jsonl.dumps(train_datas, open(os.path.join(data_path, 'train.jsonl'), 'w'))
    jsonl.dumps(test_datas, open(os.path.join(data_path, 'test.jsonl'), 'w'))
    jsonl.dumps(valid_datas, open(os.path.join(data_path, 'valid.jsonl'), 'w'))