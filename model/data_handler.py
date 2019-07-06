from typing import List, Tuple, Dict

import os
import dill

import torchtext
from torchtext import data


class DataHandler(object):

    def __init__(self,
                 field: data.Field = None,
                 train_file: str = None,
                 val_file: str = None,
                 ) -> None:

        # Attributes
        self._train_file = train_file
        self._val_file   = val_file
        self._field      = field
        self._datasets   = None

    def build_vocab(self,
                    field: data.Field = None,
                    train_file: str = None,
                    val_file: str = None,
                    target_path: str = None,
                    max_size: int = 20000,
                    ) -> None:
        """ Build vocabulary requires 2 steps i.e. load_dataset() and build_vocab()
        """

        # Build vocab out of training dataset only
        self.load_dataset(field, train_file, val_file)
        self._field.build_vocab(self._datasets[0], max_size=max_size)

        # Save vocab if target_path is specified
        if target_path:
            with open(target_path, "wb") as f:
                dill.dump(field, f)

    def load_vocab(self,
                   file_path: str,
                   ) -> None:
        
        # Load "trained" vocab i.e. torchtext.data.Field
        with open(file_path, "rb") as f:
            self._field = dill.load(f)

    def load_dataset(self,
                     field: data.Field = None,
                     train_file: str = None,
                     val_file: str = None,
                     ) -> None:

        # Change attributes required to build vocab if specified
        if field: self._field = field
        if train_file: self._train_file = train_file
        if val_file: self._val_file = val_file

        # Construct datafields to be passed to TabularDataset
        datafields = [('input', self._field),
                      ('target', self._field)]

        self._datasets = data.TabularDataset.splits(path='',
                                                    train=self._train_file,
                                                    validation=self._val_file,
                                                    format="csv",
                                                    skip_header=True,
                                                    fields=datafields)

    def gen_iterator(self,
                     batch_size: int = 32,
                     ) -> Tuple[data.BucketIterator, ...]:

        data_iterators = [data.BucketIterator(dataset, batch_size) for dataset in self._datasets]
        return tuple(data_iterators)

    @property
    def data_size(self) -> Tuple[int, ...]:
        data_sizes = [len(dataset) for dataset in self._datasets]
        return tuple(data_sizes)

    @property
    def datasets(self):
        return self._datasets
    
    @property
    def vocab(self):
        return self._field.vocab