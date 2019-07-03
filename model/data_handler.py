from typing import List, Tuple, Dict

import os
import dill

import torchtext
from torchtext import data


class DataHandler(object):

	def __init__(self,
				 dir_path: str = '../data',
				 filenames: Dict[str, str] = {"train": "train.csv",
				 							  "validation": None,
				 							  "test": None},
				 ) -> None:

		# Check file existence
		for key in filenames:
			if filenames[key] is not None:
				file_path = os.path.join(dir_path, filenames[key])
				assert os.path.isfile(file_path), "No file found at {}".format(file_path)

		# Attributes
		self._dir_path  = dir_path
		self._filenames = filenames
		self._datasets  = None
		self._field		= None

	def build_vocab(self,
					field: data.Field,
					field_keys: List[str] = ['input', 'target'],
					target_path: str = None,
					max_size: int = 20000,
					) -> None:

		# Construct datafields to be passed to TabularDataset
		datafields = []
		for field_key in field_keys:
			datafields.append((field_key, field))

		self._field = field
		self._datasets = data.TabularDataset.splits(path=self._dir_path,
													train=self._filenames["train"],
													validation=self._filenames["validation"],
													test=self._filenames["test"],
													format="csv",
													skip_header=True,
													fields=datafields)

		# Build and Save vocab if target_path is specified
		field.build_vocab(self._datasets[0], max_size=max_size)

		if target_path:
			with open(target_path, "wb") as f:
				dill.dump(field, f)

	def load_vocab(self,
				   file_path: str,
				   field_keys: List[str] = ['input', 'target'],
				   ) -> None:
		
		# Load data.Field
		field = None
		with open(file_path, "r") as f:
			field = dill.load(f)

		# Construct datafields to be passed to TabularDataset
		datafields = []
		for field_key in field_keys:
			datafields.append((field_key, field))

		self._field = field
		self._datasets = data.TabularDataset.splits(path=self._dir_path,
													train=self._filenames["train"],
													validation=self._filenames["validation"],
													test=self._filenames["test"],
													format="csv",
													skip_header=True,
													fields=datafields)

	def gen_iterator(self,
					 batch_size: int = 32,
					 ) -> Tuple[data.BucketIterator, ...]:

		data_iterators = []
		for dataset in self._datasets:
			data_iterators.append(data.BucketIterator(dataset, batch_size))
		return tuple(data_iterators)

	@property
	def data_size(self) -> Tuple[int, ...]:

		data_sizes = []
		for dataset in self._datasets:
			data_sizes.append(len(dataset))
		return tuple(data_sizes)