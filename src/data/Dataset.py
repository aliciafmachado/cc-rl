import numpy as np
import os
import pandas as pd
import pathlib
import sys
from skmultilearn.dataset import load_dataset, load_from_arff


class Dataset:
    """
    Downloads and provides a dataset from a set of available datasets.
    Available datasets: 'bibtex', 'corel5k', 'emotions', 'enron', 'mediamill', 'medical', 'scene', 'yeast'.
    """
    data_path = os.path.dirname(__file__) + '/../../data'
    skmultilearn_datasets = {'bibtex', 'corel5k', 'emotions',
                             'enron', 'mediamill', 'medical', 'scene', 'yeast'}
    other_datasets = {'flags': 'http://www.uco.es/grupos/kdis/MLLResources/ucobigfiles/Datasets/Original/Flags.zip',
                      'image': '',
                      'reuters': '',
                      'slashdot': 'http://www.uco.es/grupos/kdis/MLLResources/ucobigfiles/Datasets/Original/Slashdot.zip'}
    dataset_types = {'bibtex': 'Text', 'corel5k': 'Image', 'emotions': 'Music', 'enron': 'Text', 'flags': 'Image',
                     'mediamill': 'Video', 'medical': 'Text', 'reuters': 'Text', 'scene': 'Image', 'slashdot': 'Text', 'yeast': 'Biology'}

    def __init__(self, name):
        """
        Downloads the dataset given its name.

        Args:
            name (str): Name of the dataset.
        """
        if name not in Dataset.skmultilearn_datasets and name not in Dataset.other_datasets:
            raise NotImplementedError('Dataset <{}> not available. Available datasets: {}'.format(
                name, Dataset.skmultilearn_datasets.union(set(Dataset.other_datasets.keys()))))

        self.name = name

        if not os.path.isdir(Dataset.data_path):
            os.mkdir(Dataset.data_path)

        if name in Dataset.skmultilearn_datasets:
            self.__load_skmultilearn_dataset(name)
        else:
            self.__load_other_dataset(name)

    def info(self):
        """
        Prints information about this dataset.
        """

        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.expand_frame_repr', False):
            card = (self.Y_train.sum() + self.Y_test.sum()) / \
                (self.Y_train.shape[0] + self.Y_test.shape[0])
            print(pd.DataFrame({'Type': Dataset.dataset_types[self.name],
                                'Train Instances': self.X_train.shape[0],
                                'Test Instances': self.X_test.shape[0],
                                'Attributes': self.X_train.shape[1],
                                'Labels': self.Y_train.shape[1],
                                'Cardinality': card}, index=[self.name]))

    def __load_skmultilearn_dataset(self, name):
        if name == 'corel5k':
            name = 'Corel5k'

        # Supress print
        orig_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        self.X_train, self.Y_train, _, _ = load_dataset(
            name, 'train', data_home=Dataset.data_path)
        self.X_test, self.Y_test, _, _ = load_dataset(
            name, 'test', data_home=Dataset.data_path)
        sys.stdout.close()
        sys.stdout = orig_stdout

        # Store data
        self.X_train = self.X_train.toarray()
        self.Y_train = self.Y_train.toarray()
        self.X_test = self.X_test.toarray()
        self.Y_test = self.Y_test.toarray()

        # mediamill should be reduced to 5000 instances total.
        # Using 4500 train and 500 test
        if name == 'mediamill':
            shuffled = np.arange(self.X_train.shape[0])
            np.random.shuffle(shuffled)
            train_idx = shuffled[:4500]
            self.X_train = self.X_train[train_idx]
            self.Y_train = self.Y_train[train_idx]

            shuffled = np.arange(self.X_test.shape[0])
            np.random.shuffle(shuffled)
            test_idx = shuffled[:500]
            self.X_test = self.X_test[test_idx]
            self.Y_test = self.Y_test[test_idx]

    def __load_other_dataset(self, name):
        pass
