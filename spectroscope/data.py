"""
Tools for restricting a classification dataset to a subset of classes,
and for reordering the presentation of classes to the model.

E.g. with MNIST: train first on 0, then 0 and 1, then 0, 1, and 2, etc. 
"""


from typing import Any, Tuple, Type

import torch
from torch.utils.data import DataLoader, Dataset
from torchtyping import TensorType


def as_subset(cls: Type[Dataset]):
    """
    Wrapper for a dataset class that adds a labels attribute to the dataset,
    and a __repr__ method that prints the labels.
    """
    class Subset(cls):
        def __init__(self, *args, labels: Tuple[int, ...], **kwargs):
            self.labels = labels
            super().__init__(*args, **kwargs)

        def __repr__(self):
            return f"Subset({self.__class__.__name__}, labels={self.labels})"
    
    return Subset


def filter_by_labels(dataset: Dataset, labels: Tuple[int, ...]):
    """
    Returns an iterator over the dataset, yielding only the images and labels in labels.
    """
    for image, label in dataset:
        if label in labels:
            yield image, label


def get_filtered_dataset(dataset: Dataset, labels: Tuple[int, ...]):
    """
    Returns a new dataset with only the images and labels in labels.
    """
    dataset_cls = as_subset(type(dataset))
    return dataset_cls(filter_by_labels(dataset, labels), labels=labels)


class SubsetsLoader(DataLoader):
    """
    A data loader that trains on different subsets of the dataset 
    depending on the active subset.

    Note: it's up to you to call next_subset() when you want to move to the next subset.
    """

    def __init__(
        self,
        subsets: Tuple[Dataset],
        *args,
        **kwargs,
    ):
        self.subsets = subsets
        self.subset_idx = 0
        self.dataset = subsets[0]

        super().__init__(self.dataset, *args, **kwargs)

    def next_subset(self):
        """Moves to the next subset."""
        if self.subset_idx < len(self.subsets):
            self.subset_idx += 1
            self.dataset = self.subsets[self.subset_idx]     
        else:
            raise StopIteration

    @classmethod
    def from_filters(cls, dataset: Dataset, labels_per_subset: Tuple[Tuple[int, ...]]):
        """
        Returns a SubsetsLoader that trains on different subsets of the dataset 
        depending on the active subset.
        """
        subsets = tuple(get_filtered_dataset(dataset, labels) for labels in labels_per_subset)
        return cls(subsets, shuffle=True)

        
    def __repr__(self):
        return f"SubsetsLoader({self.subsets}, batch_size={self.batch_size}, shuffle={self.shuffle})"