from allennlp.common import Registrable
from allennlp.models import Model
from allennlp.data import Instance
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.common.tqdm import Tqdm
from allennlp.nn import util
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.dataset_readers import DatasetReader
from torch.utils.data import Dataset, DataLoader
from typing import Iterator, List, Dict
from typing import Dict, Optional, List, Tuple, Union, Iterable, Any, Set
from metrics import calc_bleu_score
import json
import torch
class Predictor(Registrable):

    def __init__(self,
                 dataset: Dataset,
                 dataloader: DataLoader,
                 corpus: object,
                 cuda_device: Union[int, List] = -1) -> None:
    
        self.dataloader = dataloader
        self.dataset = dataset
        self.corpus = corpus
        self.cuda_device = cuda_device

    def predict(self, model: Model):
        model.eval()

        generator_tqdm = Tqdm.tqdm(self.dataloader, total=len(self.dataloader))
        model_outputs = {}

        for batch in generator_tqdm:
            with torch.no_grad():
                batch = util.move_to_device(batch, self.cuda_device)
                output_dict = model.back2table(**batch)
                for key in output_dict:
                    if key not in model_outputs:
                        model_outputs[key] = output_dict[key]
                    else:
                        model_outputs[key] += output_dict[key]

        predictions = self.corpus.predict(model_outputs, self.dataset)        
        model.train()
        return predictions


    def evaluate(self, model: Model):
        model.eval()

        generator_tqdm = Tqdm.tqdm(self.dataloader, total=len(self.dataloader))
        model_outputs = {}

        for batch in generator_tqdm:
            with torch.no_grad():
                batch = util.move_to_device(batch, self.cuda_device)
                output_dict = model.predict(**batch)
                for key in output_dict:
                    if key not in model_outputs:
                        model_outputs[key] = output_dict[key]
                    else:
                        model_outputs[key] += output_dict[key]

        evaluation_results = self.corpus.evaluate(model_outputs, self.dataset)
        
        print(evaluation_results['logging'])
        
        model.train()

        return evaluation_results['score']