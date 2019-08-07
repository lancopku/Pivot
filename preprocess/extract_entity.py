from stanfordcorenlp import StanfordCoreNLP
from typing import List, Dict
from collections import defaultdict
import logging
import json
import os

data_path = os.path.expanduser('~/data/wiki2bio/')

class StanfordNLP:
    def __init__(self, host='http://localhost', port=9000):
        self.nlp = StanfordCoreNLP(host, port=port,
                                   timeout=30000)  # , quiet=False, logging_level=logging.DEBUG)
        self.props = {
            'annotators': 'tokenize,ssplit,pos,lemma,ner,parse,depparse,dcoref,relation',
            'pipelineLanguage': 'en',
            'outputFormat': 'json'
        }

    def word_tokenize(self, sentence):
        return self.nlp.word_tokenize(sentence)

    def pos(self, sentence):
        return self.nlp.pos_tag(sentence)

    def ner(self, sentence):
        return self.nlp.ner(sentence)

    def parse(self, sentence):
        return self.nlp.parse(sentence)

    def dependency_parse(self, sentence):
        return self.nlp.dependency_parse(sentence)

    def annotate(self, sentence):
        return json.loads(self.nlp.annotate(sentence, properties=self.props))

    @staticmethod
    def tokens_to_dict(_tokens):
        tokens = defaultdict(dict)
        for token in _tokens:
            tokens[int(token['index'])] = {
                'word': token['word'],
                'lemma': token['lemma'],
                'pos': token['pos'],
                'ner': token['ner']
            }
        return tokens

def extract_entity(path: str, snlp: StanfordCoreNLP) -> List[str]:
    sents = open(path).read().strip().split('\n')
    datas = []
    for i, sent in enumerate(sents):
        sent = sent.strip().replace('-lrb-', '(').replace('-rrb-', ')')
        postags = snlp.pos(sent)
        new_sent = []
        for word, pos in postags:
            if pos[0] == 'N' or pos[0] == 'J' or pos == 'CD' or pos == 'FW':
                if word == '(':
                    word = '-lrb-'
                if word == ')':
                    word = '-rrb-'
                new_sent.append(word)
        new_sent = ' '.join(new_sent)
        datas.append(new_sent)
        #if (i+1) == 50:
        #    break
        if (i+1) % 1000 == 0:
            print(i+1)
    return datas

def write_into_file(path: str, datas: List[str]) -> None:
    with open(path, 'w') as f:
        print('\n'.join(datas), end='\n', file=f)

if __name__ == '__main__':
    snlp = StanfordNLP()
    train_text = extract_entity(os.path.join(data_path, 'train.text'), snlp)
    test_text = extract_entity(os.path.join(data_path, 'test.text'), snlp)
    valid_text = extract_entity(os.path.join(data_path, 'valid.text'), snlp)

    write_into_file(os.path.join(data_path, 'train.entity'), train_text)
    write_into_file(os.path.join(data_path, 'test.entity'), test_text)
    write_into_file(os.path.join(data_path, 'valid.entity'), valid_text)
