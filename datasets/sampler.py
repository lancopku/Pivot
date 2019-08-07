import random
import torch.utils.data

class SortedSampler(torch.utils.data.Sampler):
    def __init__(self, data_source, sort_key):
        super().__init__(data_source)
        self.data_source = data_source
        self.sort_key = sort_key
        zip_ = [(i, self.sort_key(row))
                for i, row in enumerate(self.data_source)]
        zip_ = sorted(zip_, key=lambda r: r[1])
        self.sorted_indexes = [item[0] for item in zip_]

    def __iter__(self):
        return iter(self.sorted_indexes)

    def __len__(self):
        return len(self.sorted_indexes)


class NoisySortedSampler(torch.utils.data.Sampler):
    """Samples elements sequentially with noise.

    **Reference:**
    https://github.com/allenai/allennlp/blob/e125a490b71b21e914af01e70e9b00b165d64dcd/allennlp/data/iterators/bucket_iterator.py

    Args:
        data (iterable): Data to sample from.
        sort_key (callable): Specifies a function of one argument that is used to extract a
            numerical comparison key from each list element.
        sort_key_noise (float): Maximum noise added to the numerical ``sort_key``.

    Example:
        >>> list(NoisySortedSampler(range(10), sort_key=lambda i: i, sort_key_noise=0.25))
        [0, 1, 2, 3, 4, 5, 8, 6, 7, 9]
    """

    def __init__(self, data, sort_key, sort_key_noise=0.25):
        super().__init__(data)
        self.data_source = data
        self.sort_key = sort_key
        self.sort_key_noise = sort_key_noise

    def __iter__(self):
        zip_ = []
        sort_key_noise = self.sort_key_noise
        for i, row in enumerate(self.data_source):
            value = self.sort_key(row)
            noise_value = value * sort_key_noise
            noise = random.uniform(-noise_value, noise_value)
            value = noise + value
            zip_.append((i, value))
        zip_ = sorted(zip_, key=lambda r: r[1])
        self.sorted_indexes = [item[0] for item in zip_]

        return iter(self.sorted_indexes)

    def __len__(self):
        return len(self.data_source)


class BatchSampler(torch.utils.data.BatchSampler):
    def __init__(self,
                 sampler,
                 batch_size,
                 drop_last=False,
                 last_batch_first=True,
                 shuffle=True):

        super().__init__(sampler, batch_size, drop_last)

        self.last_batch_first = last_batch_first
        self.shuffle = shuffle

    def __iter__(self):
        batches = list(super().__iter__())
        if self.last_batch_first:
            last_batch = batches.pop()
        if self.shuffle:
            random.shuffle(batches)
        if self.last_batch_first:
            batches.insert(0, last_batch)
        return iter(batches)
