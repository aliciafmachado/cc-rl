from io import BytesIO
from urllib.request import urlopen, urlretrieve
from skmultilearn.dataset import load_dataset, load_from_arff, available_data_sets
import numpy as np
import os
import pandas as pd
import pathlib
import sys
from zipfile import ZipFile


class Dataset:
    """
    Downloads and provides a dataset from a set of available datasets.
    Available datasets: 'bibtex', 'birds', 'corel5k', 'delicious', 'emotions', 'enron', 'genbase', 'mediamill', 'medical',
                        'rcv1subset1', 'rcv1subset2', 'rcv1subset3', 'rcv1subset4', 'rcv1subset5', 'scene', 'tmc2007_500',
                        'yeast', 'flags', 'image'.
    """
    data_path = os.path.dirname(__file__) + '/../../data/'
    skmultilearn_datasets = set(
        [x[0] for x in available_data_sets().keys()] + ['corel5k'])
    other_datasets = {'flags': (7, 'http://www.uco.es/grupos/kdis/MLLResources/ucobigfiles/Datasets/Original/Flags.zip'),
                      'image': (5, 'https://www.openml.org/data/download/21241890/file203856ce269f.arff')}
    dataset_types = {'bibtex': 'Text',
                     'birds': 'Audio',
                     'corel5k': 'Image',
                     'delicious': 'Text',
                     'emotions': 'Music',
                     'enron': 'Text',
                     'genbase': 'Biology',
                     'mediamill': 'Video',
                     'medical': 'Text',
                     'rcv1subset1': 'Text',
                     'rcv1subset2': 'Text',
                     'rcv1subset3': 'Text',
                     'rcv1subset4': 'Text',
                     'rcv1subset5': 'Text',
                     'scene': 'Image',
                     'tmc2007_500': 'Text',
                     'yeast': 'Biology',
                     'flags': 'Image',
                     'image': 'Image'}

    def __init__(self, name):
        """
        Downloads the dataset given its name.

        Args:
            name (str): One of: 'bibtex', 'birds', 'corel5k', 'delicious', 'emotions', 'enron', 'genbase', 'mediamill',
                        'medical', 'rcv1subset1', 'rcv1subset2', 'rcv1subset3', 'rcv1subset4', 'rcv1subset5', 'scene',
                        'tmc2007_500', 'yeast', 'flags', 'image'.
        """
        if name not in Dataset.skmultilearn_datasets and name not in Dataset.other_datasets:
            raise NotImplementedError('Dataset <{}> not available. Available datasets: {}'.format(
                name, Dataset.skmultilearn_datasets.union(set(Dataset.other_datasets.keys()))))

        if name == 'Corel5k':
            name = 'corel5k'
        self.name = name

        if not os.path.isdir(Dataset.data_path):
            os.mkdir(Dataset.data_path)

        if name in Dataset.skmultilearn_datasets:
            self.__load_skmultilearn_dataset(name)
        else:
            self.__load_other_dataset(name)

    def info(self):
        """
        Gives information about this dataset.

        Returns:
            [pd.DataFrame]: 1x6 dataframe with information.
        """

        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.expand_frame_repr', False):
            card = (self.Y_train.sum() + self.Y_test.sum()) / \
                (self.Y_train.shape[0] + self.Y_test.shape[0])
            return pd.DataFrame({'Type': Dataset.dataset_types[self.name],
                                 'Train Instances': self.X_train.shape[0],
                                 'Test Instances': self.X_test.shape[0],
                                 'Attributes': self.X_train.shape[1],
                                 'Labels': self.Y_train.shape[1],
                                 'Cardinality': card}, index=[self.name])

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

        # Convert to np.array
        self.X_train = self.X_train.toarray()
        self.Y_train = np.array(self.Y_train.toarray(), dtype=bool)
        self.X_test = self.X_test.toarray()
        self.Y_test = np.array(self.Y_test.toarray(), dtype=bool)

    def __load_other_dataset(self, name):
        label_count, url = Dataset.other_datasets[name]

        if name == 'flags':
            filenames = [name + '-' + t + '.arff' for t in ['train', 'test']]
            paths = [Dataset.data_path + f for f in filenames]

            # Download if not already
            if not os.path.isfile(paths[0]) or not os.path.isfile(paths[1]):
                response = urlopen(url)
                zf = ZipFile(BytesIO(response.read()))
                for f in filenames:
                    zf.extract(member=f, path=Dataset.data_path)
                zf.close()

            self.X_train, self.Y_train = load_from_arff(paths[0], label_count)
            self.X_test, self.Y_test = load_from_arff(paths[1], label_count)

            # Convert to np.array
            self.X_train = self.X_train.toarray()
            self.Y_train = np.array(self.Y_train.toarray(), dtype=bool)
            self.X_test = self.X_test.toarray()
            self.Y_test = np.array(self.Y_test.toarray(), dtype=bool)
        elif name == 'image':
            path = Dataset.data_path + 'image.arff'

            # Download if not already
            if not os.path.isfile(path):
                urlretrieve(url, path)

            x, y = load_from_arff(path, label_count)
            x = x.toarray()
            y = np.array(self.Y_train.toarray(), dtype=bool)

            # Separate train, test in 1800, 200
            shuffled = np.arange(x.shape[0])
            np.random.shuffle(shuffled)
            train_idx = shuffled[:1800]
            test_idx = shuffled[1800:]

            self.X_train = x[train_idx]
            self.Y_train = y[train_idx]
            self.X_test = x[test_idx]
            self.Y_test = y[test_idx]
