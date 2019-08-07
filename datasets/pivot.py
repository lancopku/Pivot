from typing import Iterator, List, Dict
from allennlp.data import Instance
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from torch.utils.data import Dataset
from datasets.sampler import BatchSampler, NoisySortedSampler, SortedSampler
from datasets.collate import basic_collate
from metrics import calc_rouge_score, calc_bleu_score, calc_nist_score
from pathlib import Path
import json
import linecache
import os
import torch
import time
from utils.jsonl import loads, dumps
import numpy as np


class Table2PivotDataset(Dataset):

    def __init__(self, path: str, max_len: int, limit: int=0) -> None:
        self.path = path
        self.max_len = max_len
        self.token_indexers = {"tokens": SingleIdTokenIndexer()}
        self.field_indexers = {"keys": SingleIdTokenIndexer(namespace='keys')}
        self.lpos_indexers = {"lpos": SingleIdTokenIndexer(namespace='lpos')}
        self.rpos_indexers = {"rpos": SingleIdTokenIndexer(namespace='rpos')}
        statistic = json.loads(linecache.getline(path, 1))
        if limit == 0:
            self.length = statistic['length']
        else:
            self.length = limit

    def text_to_instance(self, src: List[Token], field: List[Token], lpos: List[Token],
                         rpos: List[Token], labels: List[str]) -> Instance:
        src_field = TextField(src, self.token_indexers)
        field_field = TextField(field, self.field_indexers)
        lpos_field = TextField(lpos, self.lpos_indexers)
        rpos_field = TextField(rpos, self.rpos_indexers)
        label_field = SequenceLabelField(labels=labels, sequence_field=src_field)
        fields = {"source": src_field,
                  "keys": field_field,
                  "lpos": lpos_field,
                  "rpos": rpos_field, 
                  "label": label_field}
        return Instance(fields)

    def read(self) -> Iterator[Dict]:
        with open(self.path) as f:
            for line in f:
                pairs = json.loads(line)
                if 'length' not in pairs:
                    yield pairs

    def __getitem__(self, index: int) -> Instance:
        data = json.loads(linecache.getline(self.path, index+2))
        src, labels = data['value'].split(' '), data['label'].split(' ')
        field, lpos, rpos = data['field'].split(' '), data['lpos'].split(' '), data['rpos'].split(' ')

        if self.max_len > 0:
            src = src[:self.max_len]
            labels = labels[:self.max_len]
            field = field[:self.max_len]
            lpos = lpos[:self.max_len]
            rpos = rpos[:self.max_len]

        src = [Token(word) for word in src]
        field = [Token(word) for word in field]
        lpos = [Token(word) for word in lpos]
        rpos = [Token(word) for word in rpos]
        return self.text_to_instance(src, field, lpos, rpos, labels)

    def __len__(self) -> int:
        return self.length
    
    def __iter__(self) -> Iterator[Instance]:
        for i in range(len(self)):
            yield self[i]


class Pivot2TextDataset(Dataset):

    def __init__(self, path: str, src_max_len: int, tgt_max_len: int, share: bool, limit: int=0, append_rate: int=0, drop_rate: float=0, blank_rate: float=0, use_feature=False) -> None:
        self.path = path
        self.src_max_len = src_max_len
        self.tgt_max_len = tgt_max_len
        if share:
            self.token_indexers = {"tokens": SingleIdTokenIndexer()}
        else:
            self.src_token_indexers = {"tokens": SingleIdTokenIndexer(namespace='src_tokens')}
            self.tgt_token_indexers = {"tokens": SingleIdTokenIndexer(namespace='tokens')}
        if use_feature:
            self.keys_indexers = {'keys': SingleIdTokenIndexer(namespace='keys')}
            self.lpos_indexers = {'lpos': SingleIdTokenIndexer(namespace='lpos')}
            self.rpos_indexers = {'rpos': SingleIdTokenIndexer(namespace='rpos')}
        statistic = json.loads(linecache.getline(path, 1))
        if limit == 0:
            self.length = statistic['length']
        else:
            self.length = limit
        print(path, self.length)
        self.share = share
        self.append_rate = append_rate
        self.drop_rate = drop_rate
        self.blank_rate = blank_rate
        self.use_feature = use_feature

    def text_to_instance(self, src: List[Token], tgt: List[Token], key: List[Token]=None, lpos: List[Token]=None, rpos: List[Token]=None) -> Instance:
        if self.share:
            src_field = TextField(src, self.token_indexers)
            tgt_field = TextField(tgt, self.token_indexers)
        else:
            src_field = TextField(src, self.src_token_indexers)
            tgt_field = TextField(tgt, self.tgt_token_indexers)
        
        if self.use_feature:
            key_field = TextField(key, self.keys_indexers)
            lpos_field = TextField(lpos, self.lpos_indexers)
            rpos_field = TextField(rpos, self.rpos_indexers)
            fields = {"src": src_field, 
                    "tgt": tgt_field,
                    "key": key_field,
                    "lpos": lpos_field,
                    "rpos": rpos_field}
        else:
            fields = {"src": src_field, 
                      "tgt": tgt_field}
        return Instance(fields)

    def read(self) -> Iterator[Dict]:
        with open(self.path) as f:
            for line in f:
                pairs = json.loads(line)
                if 'length' not in pairs:
                    yield pairs

    '''
    def word_shuffle(self, words):
        """
        Randomly shuffle input words.
        """
        if self.shuffle_rate == 0:
            return words

        noise = np.random.uniform(0, self.shuffle_rate, size=len(words))

        # be sure to shuffle entire words
        word_idx = np.array(list(range(len(words))))
        word_idx = np.argsort(word_idx+noise)
        words = np.array(words)[word_idx]
        words = words.tolist()

        return words
    '''

    def word_drop(self, words):
        if self.drop_rate == 0:
            return words
        
        noise = np.random.uniform(0, 1, size=len(words)) >= self.drop_rate
        noise = noise.tolist()
        _words = [w for w, n in zip(words, noise) if n > 0.5]
        if len(_words) == 0:
            _words = [words[np.random.randint(0, len(words))]]

        return _words
    
    def word_blank(self, words):
        if self.blank_rate == 0:
            return words
        
        noise = np.random.uniform(0, 1, size=len(words)) >= self.blank_rate
        noise = noise.tolist()
        words = [w if n > 0.5 else '@@UNKNOWN@@' for w, n in zip(words, noise)]

        return words
    
    def word_append(self, words):
        if self.append_rate == 0:
            return words
        
        _words = []
        n = 0
        while (n<len(words)):
            if np.random.uniform(0, 1) >= self.append_rate:
                _words.append(words[n])
                n+=1
            else:
                _words.append('@@UNKNOWN@@')

        return _words
    
    def add_noise(self, words):
        #words = self.word_shuffle(words)
        words = self.word_drop(words)
        words = self.word_blank(words)
        words = self.word_append(words)

        return words

    def __getitem__(self, index: int) -> Instance:
        data = json.loads(linecache.getline(self.path, index+2))
        src, tgt = data['source'].split(' '), data['target'].split(' ')
        if self.use_feature:
            key, lpos, rpos = data['field'].split(' '), data['lpos'].split(' '), data['rpos'].split(' ')

        if self.src_max_len > 0:
            src = src[:self.src_max_len]
            if self.use_feature:
                key = key[:self.src_max_len]
                lpos = lpos[:self.src_max_len]
                rpos = rpos[:self.src_max_len]
        if self.tgt_max_len > 0:
            tgt = tgt[:self.tgt_max_len]

        src = self.add_noise(src)
        tgt = [START_SYMBOL] + tgt + [END_SYMBOL]
        src = [Token(word) for word in src]
        tgt = [Token(word) for word in tgt]

        if self.use_feature:
            key = [Token(word) for word in key]
            lpos = [Token(word) for word in lpos]
            rpos = [Token(word) for word in rpos]
            assert len(src) == len(key) == len(lpos) == len(rpos)
            return self.text_to_instance(src, tgt, key, lpos, rpos)
        else:
            return self.text_to_instance(src, tgt)

    def __len__(self) -> int:
        return self.length
    
    def __iter__(self) -> Iterator[Instance]:
        for i in range(len(self)):
            yield self[i]


class Table2PivotCorpus(object):

    def __init__(self, vocab_size: int, max_len: int, batch_size: int, log_dir: str = '', mode: str='train', scale: int=10000):

        data_path = os.path.expanduser('~/data/wiki2bio/')

        train_path = os.path.join(data_path, 'train.t2p.{0}.jsonl'.format(scale))
        dev_path = os.path.join(data_path, 'valid.t2p.jsonl')
        test_path = os.path.join(data_path, 'test.t2p.jsonl')

        vocab_dir = os.path.join(data_path, 'dicts-{0}-t2p-{1}'.format(vocab_size, scale))
        self.metrics = '+f1'
        self.data_path = data_path
        self.mode = mode

        self.train_dataset = Table2PivotDataset(path=train_path, max_len=max_len)
        self.test_dataset = Table2PivotDataset(path=test_path, max_len=max_len)
        self.dev_dataset = Table2PivotDataset(path=dev_path, max_len=max_len)

        if os.path.exists(vocab_dir):
            vocab = Vocabulary.from_files(vocab_dir)
        else:
            vocab = Vocabulary.from_instances(instances=self.train_dataset,
                                              max_vocab_size=vocab_size)
            vocab.save_to_files(vocab_dir)
    
        collate_fn = basic_collate(vocab=vocab)
        
        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                        batch_size=batch_size,
                                                        collate_fn=collate_fn,
                                                        shuffle=True
                                                       )
        self.dev_loader = torch.utils.data.DataLoader(dataset=self.dev_dataset,
                                                      batch_size=128,
                                                      collate_fn=collate_fn,
                                                      shuffle=False
                                                      )
        self.test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset,
                                                       batch_size=128,
                                                       collate_fn=collate_fn,
                                                       shuffle=False
                                                       )
        self.vocab = vocab
        self.scale = scale

        if not log_dir:
            self.log_dir = Path(data_path) / 'log' / time.strftime("%Y-%m-%dT%H_%M_%S")
        else:
            self.log_dir = Path(data_path) / 'log' / log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = str(self.log_dir)
    

    def print_result_into_file(self, model_outputs: Dict, dataset: Dataset):
        model_ids = model_outputs['output_ids']
        sources, model_words = [], []
        fields, lposs, rposs = [], [], []
        _fields, _lposs, _rposs = [], [], []

        for data in dataset.read():
            sources.append(data['value'].split(' '))
            fields.append(data['field'].split(' '))
            lposs.append(data['lpos'].split(' '))
            rposs.append(data['rpos'].split(' '))

        for ids, source, field, lpos, rpos in zip(model_ids, sources, fields, lposs, rposs):
            words = [s for id, s in zip(ids, source) if id > 0]
            _field = [s for id, s in zip(ids, field) if id > 0]
            _lpos = [s for id, s in zip(ids, lpos) if id > 0]
            _rpos = [s for id, s in zip(ids, rpos) if id > 0]
            model_words.append(' '.join(words))
            _fields.append(' '.join(_field))
            _lposs.append(' '.join(_lpos))
            _rposs.append(' '.join(_rpos))
        
        with open(os.path.join(self.data_path, 'predict-{0}.txt'.format(self.scale)), 'w') as f:
            print('\n'.join(model_words), file=f)
        
        ori_datas = loads(open(os.path.join(self.data_path, 'test.p2t.jsonl')))
        for m, d, f, l, r in zip(model_words, ori_datas[1:], _fields, _lposs, _rposs):
            d['source'] = m
            d['field'] = f
            d['lpos'] = l
            d['rpos'] = r

        dumps(ori_datas, open(os.path.join(self.data_path, 'test.predict.{0}.jsonl'.format(self.scale)), 'w'))


    def evaluate(self, model_outputs: Dict, dataset: Dataset) -> str:

        correct_tokens, total_tokens = model_outputs['correct'], model_outputs['total']
        hit, r_total, p_total = model_outputs['hit'], model_outputs['r_total'], model_outputs['p_total'] 
        
        accuracy = correct_tokens * 100.0 / total_tokens
        recall = hit * 100.0 / r_total
        precision = hit * 100.0 / p_total
        f_score = 2*recall*precision/(recall+precision)

        metric_dicts = {'score': {'acc': accuracy, 'recall': recall, 'precision': precision, 'f1': f_score}, 
                        'logging': 'Accuracy: {0}, Precision: {1}, Recall: {2}, F1: {3}\n'.format(accuracy, recall, precision, f_score)}
        
        if self.mode == 'test':
            self.print_result_into_file(model_outputs, dataset)

        return metric_dicts


class Pivot2TextCorpus(object):

    def __init__(self, vocab_size: int, src_max_len: int, tgt_max_len: int, 
                 batch_size: int, share: bool, append_rate: int, drop_rate: float, blank_rate: float, 
                 log_dir: str = '', setting='pivot', mode='train', scale: int=10000, use_feature: bool=False):

        data_path = os.path.expanduser('~/data/wiki2bio/')
        
        if setting == 'pivot':
            train_path = os.path.join(data_path, 'train.p2t.{0}.jsonl'.format(scale))
            dev_path = os.path.join(data_path, 'valid.p2t.jsonl')
            test_path = os.path.join(data_path, 'test.predict.{0}.jsonl'.format(scale))
        elif setting == 'aug':
            train_path = os.path.join(data_path, 'train.aug.{0}.jsonl'.format(scale))
            dev_path = os.path.join(data_path, 'valid.parallel.jsonl')
            test_path = os.path.join(data_path, 'test.parallel.jsonl')
        elif setting == 'semi':
            train_path = os.path.join(data_path, 'train.semi.{0}.jsonl'.format(scale))
            dev_path = os.path.join(data_path, 'valid.parallel.jsonl')
            test_path = os.path.join(data_path, 'test.parallel.jsonl')
        elif setting == 'pretrain':
            train_path = os.path.join(data_path, 'train.pretrain.{0}.jsonl'.format(scale))
            dev_path = os.path.join(data_path, 'valid.parallel.jsonl')
            test_path = os.path.join(data_path, 'test.parallel.jsonl')
        else:
            train_path = os.path.join(data_path, 'train.parallel.{0}.jsonl'.format(scale))
            dev_path = os.path.join(data_path, 'valid.parallel.jsonl')
            test_path = os.path.join(data_path, 'test.parallel.jsonl')
            
        vocab_dir = os.path.join(data_path, 'dicts-{0}-share-p2t-{1}'.format(vocab_size, scale)) \
            if share else os.path.join(data_path, 'dicts-{0}-noshare-p2t-{1}'.format(vocab_size, scale))
        if setting == 'pivot':
            vocab_dir += '-pivot'
        if use_feature:
            vocab_dir += '-feature'
        self.metrics = '+bleu'
        self.mode = mode

        self.train_dataset = Pivot2TextDataset(path=train_path, src_max_len=src_max_len, tgt_max_len=tgt_max_len, share=share, append_rate=append_rate, drop_rate=drop_rate, blank_rate=blank_rate, use_feature=use_feature)
        self.test_dataset = Pivot2TextDataset(path=test_path, src_max_len=src_max_len, tgt_max_len=tgt_max_len, share=share, use_feature=use_feature)
        self.dev_dataset = Pivot2TextDataset(path=dev_path, src_max_len=src_max_len, tgt_max_len=tgt_max_len, share=share, use_feature=use_feature)

        if os.path.exists(vocab_dir):
            vocab = Vocabulary.from_files(vocab_dir)
        else:
            vocab = Vocabulary.from_instances(instances=self.train_dataset,
                                              max_vocab_size=vocab_size)
            vocab.save_to_files(vocab_dir)
    
        collate_fn = basic_collate(vocab=vocab)
        
        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                        batch_size=batch_size,
                                                        collate_fn=collate_fn,
                                                        shuffle=True
                                                       )
        self.dev_loader = torch.utils.data.DataLoader(dataset=self.dev_dataset,
                                                      batch_size=128,
                                                      collate_fn=collate_fn,
                                                      shuffle=False
                                                      )
        self.test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset,
                                                       batch_size=128,
                                                       collate_fn=collate_fn,
                                                       shuffle=False
                                                       )
        self.vocab = vocab

        if not log_dir:
            self.log_dir = Path(data_path) / 'log' / time.strftime("%Y-%m-%dT%H_%M_%S")
        else:
            self.log_dir = Path(data_path) / 'log' / log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = str(self.log_dir)
    
    def evaluate(self, model_outputs: Dict, dataset: Dataset) -> str:
        vocabulary = self.vocab.get_index_to_token_vocabulary('tokens')

        predictions, sources, references, alignments = [], [], [], model_outputs['alignments']
        
        for data in dataset.read():
            sources.append(data['source'])
            references.append(data['target'])

        for pred in model_outputs['output_ids']:
            pred_sent = list(map(vocabulary.get, pred))
            if END_SYMBOL in pred_sent:
                pred_sent = pred_sent[:pred_sent.index(END_SYMBOL)]
            pred_sent = ' '.join(pred_sent)
            predictions.append(pred_sent)
        
        for i in range(len(predictions)):
            source_sent = sources[i].split(' ')
            pred_sent = predictions[i].split(' ')
            for j in range(len(pred_sent)):
                if pred_sent[j] == '@@UNKNOWN@@' and alignments[i][j] < len(source_sent):
                    pred_sent[j] = source_sent[alignments[i][j]]
            predictions[i] = ' '.join(pred_sent)

        metric_dicts = calc_bleu_score(predictions, references, self.log_dir)

        if self.mode == 'test':
            print(metric_dicts['logging'])
            print('Start Calculate Rouge...')
            start_time = time.time()
            rouge_dicts = calc_rouge_score(predictions, references, self.log_dir)
            metric_dicts['logging'] += rouge_dicts['logging']
            print('time: {0}'.format(time.time()-start_time))

            print('Start Calculate Nist...')
            start_time = time.time()
            nist_dicts = calc_nist_score(predictions, references, self.log_dir)
            metric_dicts['logging'] += nist_dicts['logging']
            print('time: {0}'.format(time.time()-start_time))
        
        return metric_dicts
    