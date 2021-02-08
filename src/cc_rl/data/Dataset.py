from io import BytesIO
from urllib.request import urlopen, urlretrieve
from sklearn.preprocessing import MinMaxScaler
from skmultilearn.dataset import load_dataset, load_from_arff, available_data_sets
import numpy as np
import os
import pandas as pd
import pathlib
import sys
from zipfile import ZipFile


class Dataset:
    """Downloads and provides a dataset from a set of available datasets.

    Available datasets: 'bibtex', 'birds', 'Corel5k', 'delicious', 'emotions', 'enron', 
        'genbase', 'mediamill', 'medical', 'scene', 'tmc2007_500', 'yeast', 'flags',
        'image'.
    """
    data_path = os.path.dirname(__file__) + '/../../../data/'
    skmultilearn_datasets = {'bibtex', 'birds', 'Corel5k', 'delicious', 'emotions',
                             'enron',  'genbase', 'mediamill', 'medical', 'scene',
                             'tmc2007_500', 'yeast'}

    other_datasets = {
        'flags': (7, 'http://www.uco.es/grupos/kdis/MLLResources/ucobigfiles/Datasets/'
                  'Original/Flags.zip'),
        'image': (5,
                  'https://www.openml.org/data/download/21241890/file203856ce269f.arff')
    }
    available_datasets = skmultilearn_datasets.union(
        set(other_datasets.keys()))
    dataset_types = {
        'bibtex': 'Text',
        'birds': 'Audio',
        'Corel5k': 'Image',
        'delicious': 'Text',
        'emotions': 'Music',
        'enron': 'Text',
        'genbase': 'Biology',
        'mediamill': 'Video',
        'medical': 'Text',
        'scene': 'Image',
        'tmc2007_500': 'Text',
        'yeast': 'Biology',
        'flags': 'Image',
        'image': 'Image',
    }

    def __init__(self, name):
        """Downloads the dataset given its name.

        Args:
            name (str): One of: 'bibtex', 'birds', 'Corel5k', 'delicious', 'emotions', 
                'enron', 'genbase', 'mediamill', 'medical', 'scene', 'tmc2007_500', 
                'yeast', 'flags', 'image'.
        """
        # FIXME: skmultilearn temporary down
        assert(name == 'flags' or name == 'image')

        if name not in Dataset.available_datasets:
            raise ValueError('Dataset <{}> not available. Available datasets: {}'.format(
                name, Dataset.available_datasets))

        self.name = name

        if not os.path.isdir(Dataset.data_path):
            os.mkdir(Dataset.data_path)

        if name in Dataset.skmultilearn_datasets:
            self.__load_skmultilearn_dataset(name)
        else:
            self.__load_other_dataset(name)

        scaler = MinMaxScaler()
        scaler.fit(np.concatenate([self.train_x, self.test_x]))
        self.train_x = scaler.transform(self.train_x)
        self.test_x = scaler.transform(self.test_x)

        # FIXME: Limiting dataset sizes
        self.train_x = self.train_x[:5000]
        self.train_y = self.train_y[:5000]
        self.test_x = self.test_x[:5000]
        self.test_y = self.test_y[:5000]

    def info(self):
        """Gives information about this dataset.

        Returns:
            pd.DataFrame: 1x6 dataframe with information.
        """

        with pd.option_context('display.max_rows', None, 'display.max_columns', None,
                               'display.expand_frame_repr', False):
            card = (self.train_y.sum() + self.test_y.sum()) / \
                (self.train_y.shape[0] + self.test_y.shape[0])
            return pd.DataFrame({'Type': Dataset.dataset_types[self.name],
                                 'Train Instances': self.train_x.shape[0],
                                 'Test Instances': self.test_x.shape[0],
                                 'Attributes': self.train_x.shape[1],
                                 'Labels': self.train_y.shape[1],
                                 'Cardinality': card}, index=[self.name])

    def __load_skmultilearn_dataset(self, name):
        # Supress print
        orig_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        self.train_x, self.train_y, _, _ = load_dataset(
            name, 'train', data_home=Dataset.data_path)
        self.test_x, self.test_y, _, _ = load_dataset(
            name, 'test', data_home=Dataset.data_path)
        sys.stdout.close()
        sys.stdout = orig_stdout

        # Convert to np.array
        self.train_x = self.train_x.toarray()
        self.train_y = np.array(self.train_y.toarray(), dtype=bool)
        self.test_x = self.test_x.toarray()
        self.test_y = np.array(self.test_y.toarray(), dtype=bool)

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

            self.train_x, self.train_y = load_from_arff(paths[0], label_count)
            self.test_x, self.test_y = load_from_arff(paths[1], label_count)

            # Convert to np.array
            self.train_x = self.train_x.toarray()
            self.train_y = np.array(self.train_y.toarray(), dtype=bool)
            self.test_x = self.test_x.toarray()
            self.test_y = np.array(self.test_y.toarray(), dtype=bool)
            
        elif name == 'image':
            path = Dataset.data_path + 'image.arff'

            # Download if not already
            if not os.path.isfile(path):
                urlretrieve(url, path)

            x, y = load_from_arff(path, label_count)
            x = x.toarray()
            y = np.array(y.toarray(), dtype=bool)

            # Separate train, test in 1800, 200
            shuffled = np.arange(x.shape[0])
            np.random.shuffle(shuffled)
            train_idx = shuffled[:1800]
            test_idx = shuffled[1800:]

            self.train_x = x[train_idx]
            self.train_y = y[train_idx]
            self.test_x = x[test_idx]
            self.test_y = y[test_idx]
