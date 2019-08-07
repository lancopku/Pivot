'''
 @Author: Shuming Ma
 @mail:   shumingma@pku.edu.cn
 @homepage : shumingma.com
'''
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
import modules
import modules.rnn as rnn
from typing import List, Dict, Iterator
from models.util import sequence_cross_entropy_with_logits
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.models import Model
from allennlp.data import Vocabulary
from metrics import SequenceAccuracy
from models.seq2seq import Seq2Seq
from models.table2text import Table2Text
from models.transformer import Transfomer
from models.sequence_labeling import SequenceLabeling


class Pivot(Model):

    def __init__(self, 
                 emb_size: int, 
                 key_emb_size: int,
                 pos_emb_size: int,
                 hidden_size: int, 
                 n_hidden: int,
                 ff_size: int,
                 n_head: int,
                 n_block: int,
                 enc_layers: int, 
                 dec_layers: int, 
                 dropout: float, 
                 bidirectional: bool, 
                 beam_size: int, 
                 max_decoding_step: int,
                 minimum_length: int,
                 label_smoothing: float, 
                 share: bool,
                 part: str,
                 vocab: Vocabulary,
                 use_feature: bool,
                 arch: str) -> None:

        super().__init__(vocab)
        self.beam_size = beam_size
        self.max_decoding_step = max_decoding_step
        self.part = part
        self.use_feature = use_feature
        
        if self.part == 'table2pivot':
            self.table2pivot = SequenceLabeling(emb_size=emb_size,
                                                key_emb_size=key_emb_size,
                                                pos_emb_size=pos_emb_size,
                                                hidden_size=hidden_size,
                                                enc_layers=enc_layers,
                                                dropout=dropout,
                                                bidirectional=bidirectional,
                                                vocab=vocab)

        if self.part == 'pivot2text':
            if arch == 'transformer':
                self.pivot2text = Transfomer(emb_size=emb_size, 
                                            n_hidden=n_hidden, 
                                            ff_size=ff_size,
                                            n_head=n_head,
                                            n_block=n_block, 
                                            dropout=dropout, 
                                            beam_size=beam_size, 
                                            max_decoding_step=max_decoding_step,
                                            minimum_length=minimum_length,
                                            label_smoothing=label_smoothing, 
                                            share=share,
                                            vocab=vocab)
            else:
                self.pivot2text = Table2Text(emb_size=emb_size,
                                            pos_emb_size=pos_emb_size,
                                            key_emb_size=key_emb_size,
                                            hidden_size=hidden_size,
                                            enc_layers=enc_layers,
                                            dec_layers=dec_layers,
                                            dropout=dropout,
                                            bidirectional=bidirectional,
                                            beam_size=beam_size,
                                            max_decoding_step=max_decoding_step,
                                            minimum_length=minimum_length,
                                            label_smoothing=label_smoothing,
                                            share=share,
                                            vocab=vocab,
                                            use_feature=use_feature)


    def forward(self,
                **kwargs) -> Dict[str, torch.Tensor]:
        
        if self.part == 'table2pivot':
            loss = self.table2pivot(src=kwargs['source'], tgt=kwargs['label'], key=kwargs['keys'],
                                    lpos=kwargs['lpos'], rpos=kwargs['rpos'])['loss']
        
        if self.part == 'pivot2text':
            if self.use_feature:
                loss = self.pivot2text(src=kwargs['src'], tgt=kwargs['tgt'], key=kwargs['key'],
                                    lpos=kwargs['lpos'], rpos=kwargs['rpos'])['loss']
            else:
                loss = self.pivot2text(src=kwargs['src'], tgt=kwargs['tgt'])['loss']

        outputs = {'loss': loss}

        return outputs
        

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        if self.part == 'table2pivot':
            accuracy = self.table2pivot.accuracy.get_metric(reset)
        
        if self.part == 'pivot2text':
            accuracy = self.pivot2text.accuracy.get_metric(reset)
            
        return {'accuracy': accuracy}


    def predict(self, 
                **kwargs) -> Dict[str, torch.Tensor]:

        if self.part == 'table2pivot':
            outputs = self.table2pivot.predict(src=kwargs['source'], tgt=kwargs['label'], key=kwargs['keys'],
                                               lpos=kwargs['lpos'], rpos=kwargs['rpos'])
        
        if self.part == 'pivot2text':
            if self.use_feature:
                outputs = self.pivot2text.predict(src=kwargs['src'], tgt=kwargs['tgt'], key=kwargs['key'],
                                               lpos=kwargs['lpos'], rpos=kwargs['rpos'])
            else:
                outputs = self.pivot2text.predict(src=kwargs['src'], tgt=kwargs['tgt'])
        
        return outputs
