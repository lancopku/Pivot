from typing import Iterator, List, Dict
from allennlp.data import Instance
from allennlp.data.fields import TextField, ListField
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from torch.utils.data import Dataset
from datasets.sampler import BatchSampler, NoisySortedSampler, SortedSampler
from datasets.collate import basic_collate
from metrics import calc_rouge_score, calc_bleu_score
from pathlib import Path
import json
import linecache
import os
import torch
import time


class BioDataset(Dataset):

    def __init__(self, path: str, src_max_len: int, tgt_max_len: int, share: bool, limit: int=0) -> None:
        self.path = path
        self.src_max_len = src_max_len
        self.tgt_max_len = tgt_max_len
        if share:
            self.token_indexers = {"tokens": SingleIdTokenIndexer()}
        else:
            self.src_token_indexers = {"tokens": SingleIdTokenIndexer(namespace='src_tokens')}
            self.tgt_token_indexers = {"tokens": SingleIdTokenIndexer(namespace='tokens')}
        self.keys_indexers = {'keys': SingleIdTokenIndexer(namespace='keys')}
        self.lpos_indexers = {'lpos': SingleIdTokenIndexer(namespace='lpos')}
        self.rpos_indexers = {'rpos': SingleIdTokenIndexer(namespace='rpos')}
        statistic = json.loads(linecache.getline(path, 1))
        if limit == 0:
            self.length = statistic['length']
        else:
            self.length = limit
        self.share = share

    def text_to_instance(self, src: List[Token], tgt: List[Token],
                         key: List[Token], lpos: List[Token], rpos: List[Token]) -> Instance:
        if self.share:
            src_field = TextField(src, self.token_indexers)
            tgt_field = TextField(tgt, self.token_indexers)
        else:
            src_field = TextField(src, self.src_token_indexers)
            tgt_field = TextField(tgt, self.tgt_token_indexers)
        key_field = TextField(key, self.keys_indexers)
        lpos_field = TextField(lpos, self.lpos_indexers)
        rpos_field = TextField(rpos, self.rpos_indexers)
        fields = {"src": src_field, 
                  "tgt": tgt_field,
                  "key": key_field,
                  "lpos": lpos_field,
                  "rpos": rpos_field}
        return Instance(fields)

    def read(self) -> Iterator[Dict]:
        with open(self.path) as f:
            for line in f:
                pairs = json.loads(line)
                if 'length' not in pairs:
                    yield pairs

    def __getitem__(self, index: int) -> Instance:
        data = json.loads(linecache.getline(self.path, index+2))
        src, tgt = data['source'].split(' '), data['target'].split(' ')
        key, lpos, rpos = data['field'].split(' '), data['lpos'].split(' '), data['rpos'].split(' ')
        #query, start, end = data['query'].split(' '), data['start'].split(' '), data['end'].split(' ')

        if self.src_max_len > 0:
            src = src[:self.src_max_len]
            key = key[:self.src_max_len]
            lpos = lpos[:self.src_max_len]
            rpos = rpos[:self.src_max_len]
        if self.tgt_max_len > 0:
            tgt = tgt[:self.tgt_max_len]

        tgt = [START_SYMBOL] + tgt + [END_SYMBOL]
        src = [Token(word) for word in src]
        tgt = [Token(word) for word in tgt]
        key = [Token(word) for word in key]
        lpos = [Token(word) for word in lpos]
        rpos = [Token(word) for word in rpos]
        assert len(src) == len(key) == len(lpos) == len(rpos)
        return self.text_to_instance(src, tgt, key, lpos, rpos)

    def __len__(self) -> int:
        return self.length
    
    def __iter__(self) -> Iterator[Instance]:
        for i in range(len(self)):
            yield self[i]


class BioCorpus(object):

    def __init__(self, vocab_size: int, src_max_len: int, tgt_max_len: int, batch_size: int, share: bool, limit: int, log_dir: str = ''):

        data_path = os.path.expanduser('~/data/wiki2bio/')
        train_path, dev_path = os.path.join(data_path, 'train.jsonl'), os.path.join(data_path, 'valid.jsonl')
        test_path = os.path.join(data_path, 'test.jsonl')
        vocab_dir = os.path.join(data_path, 'dicts-{0}-share-feature'.format(vocab_size)) if share else os.path.join(data_path, 'dicts-{0}-noshare-feature'.format(vocab_size))
        self.metrics = '+bleu'

        self.train_dataset = BioDataset(path=train_path, src_max_len=src_max_len, tgt_max_len=tgt_max_len, share=share, limit=limit)
        self.test_dataset = BioDataset(path=test_path, src_max_len=src_max_len, tgt_max_len=tgt_max_len, share=share)
        self.dev_dataset = BioDataset(path=dev_path, src_max_len=src_max_len, tgt_max_len=tgt_max_len, share=share)

        if os.path.exists(vocab_dir):
            vocab = Vocabulary.from_files(vocab_dir)
        else:
            vocab = Vocabulary.from_instances(instances=self.train_dataset,
                                              max_vocab_size=vocab_size)
            vocab.save_to_files(vocab_dir)
            print('src vocab: ', vocab.get_vocab_size('src_tokens'))
            print('tgt vocab: ', vocab.get_vocab_size('tokens'))
            print('key vocab: ', vocab.get_vocab_size('keys'))
            print('lpos vocab: ', vocab.get_vocab_size('lpos'))
            print('rpos vocab: ', vocab.get_vocab_size('rpos'))
    
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
        
        return metric_dicts
    