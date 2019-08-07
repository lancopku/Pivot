'''
 @Author: Shuming Ma
 @mail:   shumingma@pku.edu.cn
 @homepage : shumingma.com
'''
import torch
import torch.nn as nn
from torch.autograd import Variable
import modules
from modules.transformer import TransformerEncoder, TransformerDecoder, Embeddings, PositionalEncoding, subsequent_mask
from typing import List, Dict, Iterator
from models.util import sequence_cross_entropy_with_logits
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.models import Model
from allennlp.data import Vocabulary
from metrics import SequenceAccuracy


class Transfomer(Model):

    def __init__(self, 
                 emb_size: int, 
                 n_hidden: int, 
                 ff_size: int,
                 n_head: int,
                 n_block: int, 
                 dropout: float, 
                 beam_size: int, 
                 max_decoding_step: int,
                 minimum_length: int,
                 label_smoothing: float, 
                 share: bool,
                 vocab: Vocabulary) -> None:

        super().__init__(vocab)

        self.vocab = vocab
        self.vocab_size = vocab.get_vocab_size('tokens')
        self.beam_size = beam_size
        self.max_decoding_step = max_decoding_step
        self.minimum_length = minimum_length
        self.label_smoothing = label_smoothing
        self._bos = self.vocab.get_token_index(START_SYMBOL)
        self._eos = self.vocab.get_token_index(END_SYMBOL)

        if share:
            self.src_embedding = nn.Sequential(Embeddings(emb_size, self.vocab_size), PositionalEncoding(n_hidden, dropout))
            self.tgt_embedding = self.src_embedding
        else:
            src_vocab_size = vocab.get_vocab_size('src_tokens')
            self.src_embedding = nn.Sequential(Embeddings(emb_size, src_vocab_size), PositionalEncoding(n_hidden, dropout))
            self.tgt_embedding = nn.Sequential(Embeddings(emb_size, self.vocab_size), PositionalEncoding(n_hidden, dropout))
        
        self.encoder = TransformerEncoder(n_hidden, ff_size, n_head, dropout, n_block)
        self.decoder = TransformerDecoder(n_hidden, ff_size, n_head, dropout, n_block)

        self.generator = nn.Linear(n_hidden, self.vocab_size)
        self.accuracy = SequenceAccuracy()

    def encode(self, src):
        embs = self.src_embedding(src)
        out = self.encoder(embs)
        return out

    def decode(self, tgt, memory, step_wise=False):
        embs = self.tgt_embedding(tgt)
        outputs = self.decoder(embs, memory, step_wise)
        out, attns = outputs['outs'], outputs['attns']
        out = self.generator(out)
        return out, attns

    def forward(self,
                src: Dict[str, torch.Tensor],
                tgt: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        src, tgt = src['tokens'], tgt['tokens']
        encode_outputs = self.encode(src)
        out_logits, _ = self.decode(tgt[:,:-1], encode_outputs)

        targets = tgt[:, 1:].contiguous()
        seq_mask = (targets>0).float()

        self.accuracy(predictions=out_logits, gold_labels=targets, mask=seq_mask)
        loss = sequence_cross_entropy_with_logits(logits=out_logits, 
                                                  targets=targets,
                                                  weights=seq_mask,
                                                  average='token',
                                                  label_smoothing=self.label_smoothing)
        outputs = {'loss': loss}

        return outputs

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {'accuracy': self.accuracy.get_metric(reset)}


    def predict(self, 
                src: Dict[str, torch.Tensor], 
                tgt: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        with torch.no_grad(): 
            if self.beam_size == 1:
                return self.greedy_search(src, self.max_decoding_step)
            else:
                return self.beam_search(src, self.max_decoding_step)

    def greedy_search(self, 
                      src: Dict[str, torch.Tensor],
                      max_decoding_step: int) -> Dict[str, torch.Tensor]:

        src = src['tokens']
        ys = torch.ones(src.size(0), 1).long().fill_(self._bos).cuda()
        #self.decoder.init_cache()
        #output_ids, attns = [], []

        encode_outputs = self.encode(src)

        for i in range(max_decoding_step):
            outputs, attn = self.decode(ys, encode_outputs, step_wise=False)
            logits = outputs[:, -1]
            #logits = outputs
            next_id = logits.max(1, keepdim=True)[1]
            #ys = next_id
            #output_ids.append(ys)
            #attns.append(attn)
            ys = torch.cat([ys, next_id], dim=1)

        output_ids = ys[:, 1:]
        attns = attn
        #output_ids = torch.cat(output_ids, dim=1)
        #attns = torch.cat(attns, dim=1)
        alignments = attns.max(2)[1]
        outputs = {'output_ids': output_ids.tolist(), 'alignments': alignments.tolist()}

        return outputs


    def beam_search(self, 
                    src: Dict[str, torch.Tensor], 
                    max_decoding_step: int) -> Dict[str, torch.Tensor]:

        beam_size = self.beam_size
        src = src['tokens']
        batch_size = src.size(0)
        encode_outputs = self.encode(src)

        contexts = encode_outputs.repeat(beam_size, 1, 1)
        beam = [modules.beam.Beam(beam_size, bos = self._bos, eos = self._eos, n_best = 1, minimum_length=self.minimum_length)
                for _ in range(batch_size)]

        for i in range(max_decoding_step):

            if all((b.done() for b in beam)):
                break

            inp = torch.stack([b.getCurrentState() for b in beam]).t().contiguous().view(-1)

            outputs, attns = self.decode(inp, contexts)
            logits = outputs[:, -1]

            output = torch.nn.functional.log_softmax(logits, dim=-1).view(beam_size, batch_size, -1)
            attn = attn.view(beam_size, batch_size, -1)

            for j, b in enumerate(beam):
                b.advance(output.data[:, j], attn.data[:, j])
                b.beam_update(decState, j)

        allHyps, allScores, allAttn = [], [], []

        for j in rev_indices:
            b = beam[j]
            n_best = 1
            scores, ks = b.sortFinished(minimum=n_best)
            hyps, attn = [], []
            for i, (times, k) in enumerate(ks[:n_best]):
                hyp, att = b.getHyp(times, k)
                hyps.append(hyp)
                attn.append(att.max(1)[1])
            allHyps.append(hyps[0])
            allScores.append(scores[0])
            allAttn.append(attn[0])

        outputs = {'output_ids': allHyps, 'alignments': allAttn}
        return outputs