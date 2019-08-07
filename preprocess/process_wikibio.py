import sys
sys.path.append('../')
import re
import json
import os
from typing import List, Dict, Iterator
from utils import jsonl

data_path = os.path.expanduser('~/data/wiki2bio/')
#popular_word_list = open(os.path.join(data_path, 'dicts-20000-noshare-feature', 'tokens.txt')).read().strip().split('\n')
#popular_word_set = set(popular_word_list[:500])

def extract_exact_query_and_answer(data: Dict) -> Dict:
    text = data['target']
    field = data['field'].split(' ')
    value = data['source'].split(' ')

    dicts = {}
    for f, v in zip(field, value):
        if f in dicts:
            dicts[f] += ' ' + v
        else:
            dicts[f] = v

    index, count = [], 0
    for t in text:
        index.append(str(count))
        if t == ' ':
            count += 1
    
    query, start, end = [], [], []
    duplicate_dicts = {}
    for f in dicts:
        v = dicts[f]
        if v in duplicate_dicts:
            continue
        pos = text.find(v)
        duplicate_dicts[v] = pos
        if pos>-1:
            query.append(f)
            start.append(index[pos])
            end.append(index[pos+len(v)-1])
    data['query'] = ' '.join(query)
    data['start'] = ' '.join(start)
    data['end'] = ' '.join(end)

    return data

def extract_greedy_query_and_answer(data: Dict) -> Dict:
    text = data['target'].split(' ')
    field = data['field'].split(' ')
    value = data['source'].split(' ')

    dicts = {}
    for f, v in zip(field, value):
        if v not in dicts:
            dicts[v] = f
    
    query, start, end = [], [], []
    i = 0
    while (i<len(text)):
        if text[i] in dicts:
            field = dicts[text[i]]
            j = i + 1
            while (j < len(text) and text[j] in dicts and dicts[text[j]] == field):
                j += 1
            query.append(field)
            start.append(str(i))
            end.append(str(j-1))
            i = j
        else:
            i = i + 1
    data['query'] = ' '.join(query)
    data['start'] = ' '.join(start)
    data['end'] = ' '.join(end)

    return data

def remove_extra_info(data: Dict) -> Dict:
    value_set = set(data['source'].split(' '))
    text = data['target'].split(' ')
    new_text = []
    remove_count = 0
    for t in text:
        if t in value_set or t in popular_word_set:
            new_text.append(t)
        else:
            remove_count += 1
    data['target'] = ' '.join(new_text)
    return data, remove_count

def write_into_file(path: str, datas: List[Dict]) -> None:
    fout = open(path, 'w')
    for data in datas[1:]:
        print(data['target'], end='\n', file=fout)
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
    #remove_count = 0
    for v, f, l, r, t in zip(box_values, box_fields, box_lpos, box_rpos, texts):
        data = {'source': v.strip(), 'target': t.strip(), 'field': f.strip(), 'lpos': l.strip(), 'rpos': r.strip()}
        #data = extract_exact_query_and_answer(data)
        #if remove:
        #    data, ct = remove_extra_info(data)
        #    remove_count += ct
        datas.append(data)
        src_len.append(len(data['source'].split(' ')))
        tgt_len.append(len(data['target'].split(' ')))
        count += 1
        if count % 5000 == 0:
            print(count)
    
    print('max len: ', max(src_len), max(tgt_len))
    print('avg len: ', sum(src_len)*1.0/len(src_len), sum(tgt_len)*1.0/len(tgt_len))
    #print('remove count: ', remove_count)
    
    return datas

if __name__ == '__main__':
    train_datas = transform(os.path.join(data_path, 'train'))
    test_datas = transform(os.path.join(data_path, 'test'))
    valid_datas = transform(os.path.join(data_path, 'valid'))

    #write_into_file(os.path.join(data_path, 'new.train.summary'), train_datas)

    jsonl.dumps(train_datas, open(os.path.join(data_path, 'train.jsonl'), 'w'))
    jsonl.dumps(test_datas, open(os.path.join(data_path, 'test.jsonl'), 'w'))
    jsonl.dumps(valid_datas, open(os.path.join(data_path, 'valid.jsonl'), 'w'))