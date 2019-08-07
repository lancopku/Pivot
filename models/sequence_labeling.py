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


class SequenceLabeling(Model):

    def __init__(self, 
                 emb_size: int, 
                 key_emb_size: int, 
                 pos_emb_size: int, 
                 hidden_size: int, 
                 enc_layers: int,
                 dropout: float, 
                 bidirectional: bool, 
                 vocab: Vocabulary) -> None:

        super().__init__(vocab)

        self.vocab = vocab
        self.vocab_size = vocab.get_vocab_size('tokens')
        self.label_size = vocab.get_vocab_size('labels')

        self.src_embedding = nn.Embedding(self.vocab_size, emb_size)
        print(vocab.get_vocab_size('keys'))
        print(vocab.get_vocab_size('lpos'))
        print(vocab.get_vocab_size('rpos'))
        self.key_embedding = nn.Embedding(vocab.get_vocab_size('keys'), key_emb_size)
        self.lpos_embedding = nn.Embedding(vocab.get_vocab_size('lpos'), pos_emb_size)
        self.rpos_embedding = nn.Embedding(vocab.get_vocab_size('rpos'), pos_emb_size)

        self.encoder = rnn.rnn_encoder(emb_size+key_emb_size+pos_emb_size*2, hidden_size, enc_layers, dropout, bidirectional)
        self.decoder = nn.Linear(hidden_size, self.label_size)
        self.accuracy = SequenceAccuracy()


    def _get_lengths(self, x: torch.Tensor) -> torch.Tensor:
        lengths = (x > 0).sum(-1)
        return lengths


    def forward(self,
                src: Dict[str, torch.Tensor],
                tgt: torch.Tensor,
                key: Dict[str, torch.Tensor],
                lpos: Dict[str, torch.Tensor],
                rpos: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        
        src = src['tokens']
        keys, lpos, rpos = key['keys'], lpos['lpos'], rpos['rpos']

        lengths = self._get_lengths(src)
        lengths, indices = lengths.sort(dim=0, descending=True)
        src = src.index_select(dim=0, index=indices)
        tgt = tgt.index_select(dim=0, index=indices)
        keys = keys.index_select(dim=0, index=indices)
        lpos = lpos.index_select(dim=0, index=indices)
        rpos = rpos.index_select(dim=0, index=indices)

        src_embs = torch.cat([self.src_embedding(src), 
                              self.key_embedding(keys),
                              self.lpos_embedding(lpos),
                              self.rpos_embedding(rpos)], dim=-1)

        src_embs = pack(src_embs, lengths, batch_first=True)
        encode_outputs = self.encoder(src_embs)
        out_logits = self.decoder(encode_outputs['hidden_outputs'])
        seq_mask = (src>0).float()

        self.accuracy(predictions=out_logits, gold_labels=tgt, mask=seq_mask)
        loss = sequence_cross_entropy_with_logits(logits=out_logits, 
                                                  targets=tgt,
                                                  weights=seq_mask,
                                                  average='token')
        outputs = {'loss': loss}

        return outputs
        

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {'accuracy': self.accuracy.get_metric(reset)}


    def predict(self, 
                src: Dict[str, torch.Tensor],
                tgt: torch.Tensor,
                key: Dict[str, torch.Tensor],
                lpos: Dict[str, torch.Tensor],
                rpos: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        with torch.no_grad(): 
            src = src['tokens']
            keys, lpos, rpos = key['keys'], lpos['lpos'], rpos['rpos']
            lengths = self._get_lengths(src)
            lengths, indices = lengths.sort(dim=0, descending=True)
            rev_indices = indices.sort()[1]
            src = src.index_select(dim=0, index=indices)
            keys = keys.index_select(dim=0, index=indices)
            lpos = lpos.index_select(dim=0, index=indices)
            rpos = rpos.index_select(dim=0, index=indices)

            src_embs = torch.cat([self.src_embedding(src), 
                                self.key_embedding(keys),
                                self.lpos_embedding(lpos),
                                self.rpos_embedding(rpos)], dim=-1)

            src_embs = pack(src_embs, lengths, batch_first=True)
            encode_outputs = self.encoder(src_embs)
            out_logits = self.decoder(encode_outputs['hidden_outputs'])
            outputs = out_logits.max(-1)[1]

            outputs = outputs.index_select(dim=0, index=rev_indices)
            src = src.index_select(dim=0, index=rev_indices)

            seq_mask = src>0
            correct_tokens = (tgt.eq(outputs)*seq_mask).sum()
            total_tokens = seq_mask.sum()

            h_tokens = (tgt.eq(outputs)*seq_mask*(tgt.eq(0))).sum()
            r_total_tokens = (tgt.eq(0)*seq_mask).sum()
            p_total_tokens = (outputs.eq(0)*seq_mask).sum()

            output_ids = 1-outputs

            return {'correct': correct_tokens.item(), 'total': total_tokens.item(), 'hit': h_tokens.item(),
                    'r_total': r_total_tokens.item(), 'p_total': p_total_tokens.item(),
                    'output_ids': output_ids.tolist()}