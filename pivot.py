from typing import Iterator, List, Dict
import torch
import torch.optim as optim
import numpy as np
import os
import time
import argparse
import json
from pathlib import Path
from allennlp.data import Instance
from allennlp.data.fields import TextField
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.training.learning_rate_schedulers import LearningRateWithMetricsWrapper
from allennlp.data.iterators import BucketIterator, BasicIterator
from trainer import Trainer
from allennlp.nn import util
from allennlp.common.tqdm import Tqdm
from metrics import SequenceAccuracy, calc_bleu_score
from models.pivot import Pivot
from datasets.pivot import Table2PivotCorpus, Pivot2TextCorpus
from datasets.sampler import BatchSampler, NoisySortedSampler
from datasets.collate import basic_collate
from predictor import Predictor

parser = argparse.ArgumentParser(description='train.py')
parser.add_argument('-emb_size', type=int, default=400, help="Embedding size")
parser.add_argument('-key_emb_size', type=int, default=50, help="Key Embedding size")
parser.add_argument('-pos_emb_size', type=int, default=5, help="Pos Embedding size")
parser.add_argument('-hidden_size', type=int, default=500, help="Hidden size")
parser.add_argument('-n_hidden', type=int, default=512, help="")
parser.add_argument('-ff_size', type=int, default=2048, help="")
parser.add_argument('-n_head', type=int, default=8, help="")
parser.add_argument('-n_block', type=int, default=6, help="")
parser.add_argument('-enc_layers', type=int, default=1, help="Number of encoder layer")
parser.add_argument('-dec_layers', type=int, default=1, help="Number of decoder layer")
parser.add_argument('-batch_size', type=int, default=64, help="Batch size")
parser.add_argument('-beam_size', type=int, default=1, help="Beam size")
parser.add_argument('-vocab_size', type=int, default=20000, help="Vocabulary size")
parser.add_argument('-epoch', type=int, default=50, help="Number of epoch")
parser.add_argument('-report', type=int, default=100000, help="Number of report interval")
parser.add_argument('-lr', type=float, default=3e-4, help="Learning rate")
parser.add_argument('-lr_decay', type=float, default=1.0, help="Learning rate Decay")
parser.add_argument('-ema_decay', type=float, default=1.000, help="Moving Average rate Decay")
parser.add_argument('-dropout', type=float, default=0.2, help="Dropout rate")
parser.add_argument('-noise_prob', type=float, default=0.2, help="Noise rate")
parser.add_argument('-grad_norm', type=float, default=5.0, help="Gradient Norm")
parser.add_argument('-label_smoothing', type=float, default=0.0, help="Dropout rate")
parser.add_argument('-restore', type=str, default='', help="Restoring model path")
parser.add_argument('-mode', type=str, default='train', help="Train or test")
parser.add_argument('-arch', type=str, default='s2s', help="")
parser.add_argument('-setting', type=str, default='pivot', help="")
parser.add_argument('-scale', type=int, default=10000, help="")
parser.add_argument('-part', type=str, default='', help="Part of the model")
parser.add_argument('-optimizer', type=str, default='adam', help="Optimizer options")
parser.add_argument('-dir', type=str, default='', help="Checkpoint directory")
parser.add_argument('-src_max_len', type=int, default=0, help="Limited length for text")
parser.add_argument('-tgt_max_len', type=int, default=0, help="Limited length for text")
parser.add_argument('-append_rate', type=float, default=0, help="")
parser.add_argument('-drop_rate', type=float, default=0, help="")
parser.add_argument('-blank_rate', type=float, default=0, help="")
parser.add_argument('-max_step', type=int, default=150, help="Max decoding step")
parser.add_argument('-minimum_length', type=int, default=0, help="Minimum length for beam decoding")
parser.add_argument('-gpu', type=int, default=0, help="GPU device")
parser.add_argument('-lazy', action='store_true', help="Lazyness of dataset")
parser.add_argument('-fp16', action='store_true', help="Whether to use 16-bit float precision instead of 32-bit")
parser.add_argument('-bidirectional', action='store_true', help="Bidirectional model")
parser.add_argument('-feature', action='store_true', help="")
parser.add_argument('-share', action='store_true', help="Shared Embeddings")
parser.add_argument('-loss_scale', type=float, default=128,
                    help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                         "0 (default value): dynamic loss scaling.\n"
                         "Positive power of 2: static loss scaling value.\n")

opt = parser.parse_args()
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
torch.cuda.set_device(opt.gpu)
device = torch.device('cuda')

def train():
    if opt.part == 'table2pivot':
        corpus = Table2PivotCorpus(vocab_size=opt.vocab_size, 
                                    max_len=opt.src_max_len, 
                                    batch_size=opt.batch_size,
                                    log_dir=opt.dir,
                                    scale=opt.scale,
                                    mode=opt.mode)
    else:
        corpus = Pivot2TextCorpus(vocab_size=opt.vocab_size, 
                                    src_max_len=opt.src_max_len, 
                                    tgt_max_len=opt.tgt_max_len, 
                                    batch_size=opt.batch_size,
                                    share=opt.share,
                                    log_dir=opt.dir,
                                    scale=opt.scale,
                                    append_rate=opt.append_rate,
                                    drop_rate=opt.drop_rate,
                                    blank_rate=opt.blank_rate,
                                    setting=opt.setting,
                                    mode=opt.mode,
                                    use_feature=opt.feature)

    model = Pivot(emb_size=opt.emb_size,
                    key_emb_size=opt.key_emb_size,
                    pos_emb_size=opt.pos_emb_size,
                    hidden_size=opt.hidden_size,
                    n_hidden=opt.n_hidden,
                    n_block=opt.n_block,
                    ff_size=opt.ff_size,
                    n_head=opt.n_head,
                    enc_layers=opt.enc_layers,
                    dec_layers=opt.dec_layers,
                    dropout=opt.dropout,
                    bidirectional=opt.bidirectional,
                    beam_size=opt.beam_size,
                    max_decoding_step=opt.max_step,
                    minimum_length=opt.minimum_length,
                    label_smoothing=opt.label_smoothing,
                    share=opt.share,
                    part=opt.part,
                    vocab=corpus.vocab,
                    use_feature=opt.feature,
                    arch=opt.arch)
    
    if opt.fp16:
        model.half()
        model.to(device)
        try:
            from apex.fp16_utils import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(model.parameters(),
                              lr=opt.lr,
                              bias_correction=False)
        if opt.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=opt.loss_scale)
    else:
        model.to(device)
        if opt.optimizer == 'adagrad':
            optimizer = optim.Adagrad(model.parameters(), lr=opt.lr, initial_accumulator_value=0.1)
        else:
            optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    learning_rate_scheduler = LearningRateWithMetricsWrapper(torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='max', patience=2))

    predictor = Predictor(dataset=corpus.test_dataset,
                          dataloader=corpus.test_loader,
                          corpus=corpus,
                          cuda_device=opt.gpu)

    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      learning_rate_scheduler=learning_rate_scheduler,
                      learning_rate_decay=opt.lr_decay,
                      ema_decay=opt.ema_decay,
                      predictor=predictor, 
                      train_loader=corpus.train_loader,
                      train_dataset=corpus.train_dataset,
                      validation_metric=corpus.metrics,
                      cuda_device=opt.gpu,
                      patience=4,
                      num_epochs=opt.epoch,
                      serialization_dir=corpus.log_dir,
                      num_serialized_models_to_keep=3,
                      summary_interval=opt.report,
                      should_log_parameter_statistics=False,
                      grad_norm=opt.grad_norm,
                      fp16=opt.fp16)

    trainer.train()


def evaluate():
    if opt.part == 'table2pivot':
        corpus = Table2PivotCorpus(vocab_size=opt.vocab_size, 
                                    max_len=opt.src_max_len, 
                                    batch_size=opt.batch_size,
                                    log_dir=opt.dir,
                                    scale=opt.scale,
                                    mode=opt.mode)
    else:
        corpus = Pivot2TextCorpus(vocab_size=opt.vocab_size, 
                                    src_max_len=opt.src_max_len, 
                                    tgt_max_len=opt.tgt_max_len, 
                                    batch_size=opt.batch_size,
                                    share=opt.share,
                                    log_dir=opt.dir,
                                    scale=opt.scale,
                                    append_rate=opt.append_rate,
                                    drop_rate=opt.drop_rate,
                                    blank_rate=opt.blank_rate,
                                    setting=opt.setting,
                                    mode=opt.mode,
                                    use_feature=opt.feature)

    model = Pivot(emb_size=opt.emb_size,
                    key_emb_size=opt.key_emb_size,
                    pos_emb_size=opt.pos_emb_size,
                    hidden_size=opt.hidden_size,
                    n_hidden=opt.n_hidden,
                    n_block=opt.n_block,
                    ff_size=opt.ff_size,
                    n_head=opt.n_head,
                    enc_layers=opt.enc_layers,
                    dec_layers=opt.dec_layers,
                    dropout=opt.dropout,
                    bidirectional=opt.bidirectional,
                    beam_size=opt.beam_size,
                    max_decoding_step=opt.max_step,
                    minimum_length=opt.minimum_length,
                    label_smoothing=opt.label_smoothing,
                    share=opt.share,
                    part=opt.part,
                    vocab=corpus.vocab,
                    use_feature=opt.feature,
                    arch=opt.arch)
    
    if opt.fp16:
        model.half()

    model = model.cuda(opt.gpu)
    model_state = torch.load(opt.restore, map_location=util.device_mapping(-1))
    model.load_state_dict(model_state)

    predictor = Predictor(dataset=corpus.test_dataset,
                          dataloader=corpus.test_loader,
                          corpus=corpus,
                          cuda_device=opt.gpu)
    
    predictor.evaluate(model)



if __name__ == '__main__':
    if opt.mode == 'train':
        train()
    else:
        evaluate()
