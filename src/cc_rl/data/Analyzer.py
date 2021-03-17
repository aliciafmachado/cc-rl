import numpy as np
import pandas as pd
import time
from sklearn.metrics import zero_one_loss, hamming_loss

from cc_rl.classifier_chain.ClassifierChain import ClassifierChain
from cc_rl.data.Dataset import Dataset


class Analyzer:
    available_methods = ['greedy', 'exhaustive_search', 'epsilon_approximation',
                         'beam_search', 'monte_carlo', 'efficient_monte_carlo',
                         'qlearning', 'mcts']
    available_losses = ['exact_match', 'hamming']
    available_datasets = list(Dataset.available_datasets)
    all_params = {'greedy': [{}],
                  'exhaustive_search': [{}],
                  'epsilon_approximation': [{'epsilon': 0.25}],
                  'beam_search': [{'b': 2}, {'b': 3}, {'b': 10}],
                  'monte_carlo': [{'q': 10}, {'q': 50}, {'q': 100}],
                  'efficient_monte_carlo': [{'q': 10}, {'q': 50}, {'q': 100}],
                  'qlearning': [{'nb_sim': 100, 'nb_paths': 3, 'epochs': 10}],
                  'mcts': [{'nb_sim': 60, 'nb_paths': 1, 'epochs': 10, 'mcts_passes': 3}]}

    def __init__(self, methods='all', losses='all', datasets='all', params=all_params):
        """Default constructor. It checks if all inputs are correct and transforms them
            into lists.

        Args:
            methods (str or list, optional): one of 'all', name of a method or lists of
                names of methods. Defaults to 'all'.
            losses (str or list, optional): one of 'all', 'exact_match', 'humming'. 
                Defaults to 'all'.
            datasets (str or list, optional): one of 'all', name of a dataset or lists of
                names of datasets. Defaults to 'all'.
            params (dict, optional): Dict with keys as the names of each method and values
                as lists of dicts of keyword arguments that method receives. Defaults to
                all_params.
        """
        pd.set_option('display.max_columns', 100)
        pd.set_option('display.max_rows', 100)

        self.methods = self.__parse(methods, Analyzer.available_methods)
        self.losses = self.__parse(losses, Analyzer.available_losses)
        self.datasets = self.__parse(datasets, Analyzer.available_datasets)
        self.__assert_params(params)
        self.params = params

    def analyze(self):
        """Predicts values in each dataset for each method, storing the results in
        dataframes.

        Returns:
            dict of pd.DataFrame: Returns dict with four keys: 'time', 'n_nodes', 
                'exact_search' and 'humming', with a dataframe of results for each dataset
                and each method.
        """

        dfs = {}
        for l in self.losses + [l + '_reward' for l in self.losses] + ['time', 'n_nodes']:
            dfs[l] = pd.DataFrame(index=self.datasets)

        for i in range(len(self.datasets)):
            ds = self.datasets[i]

            print('Dataset {}/{}: {} - Loading'.format(
                i+1, len(self.datasets), ds), end='... ', flush=True)
            dataset = Dataset(ds)

            print('Fitting', end='... ', flush=True)
            cc = ClassifierChain()
            cc.fit(dataset)

            for method in self.methods:
                for p in self.params[method]:
                    column = method + ' ' + str(p)
                    print(column, end='... ', flush=True)

                    # Limiting exhaustive search, it has exponential complexity
                    if dataset.train_y.shape[1] > 15 and method == 'exhaustive_search':
                        dfs['time'].loc[ds, column] = np.nan
                        dfs['n_nodes'].loc[ds, column] = np.nan
                        dfs['exact_match'].loc[ds, column] = np.nan
                        dfs['hamming'].loc[ds, column] = np.nan
                        dfs['exact_match_reward'].loc[ds, column] = np.nan
                        dfs['hamming_reward'].loc[ds, column] = np.nan
                        break

                    # Predict and calculate metrics
                    if len(self.losses) == 2:
                        t = 0
                        for l in self.losses:
                            t1 = time.time()
                            pred, n_nodes, reward = cc.predict(
                                dataset, method, loss=l, return_num_nodes=True,
                                return_reward=True, **p)
                            t += time.time() - t1

                            if l == 'exact_match':
                                loss = zero_one_loss(dataset.test_y, pred)
                            else:
                                loss = hamming_loss(dataset.test_y, pred)

                            dfs[l].loc[ds, column] = loss
                            dfs[l + '_reward'].loc[ds, column] = reward
                        t /= 2 * len(dataset.test_x)
                    else:
                        l = self.losses[0]
                        t = time.time()
                        pred, n_nodes, reward = cc.predict(
                            dataset, method, loss=l, return_num_nodes=True,
                            return_reward=True, **p)
                        t = (time.time() - t) / len(dataset.test_x)

                        dfs[l].loc[ds, column] = zero_one_loss(
                            dataset.test_y, pred)
                        dfs[l + '_reward'].loc[ds, column] = reward

                    dfs['time'].loc[ds, column] = t
                    dfs['n_nodes'].loc[ds, column] = n_nodes
            
            print()

        return dfs

    def __parse(self, params, available_params):
        """Converts input into a list, checking if the input is correct.

        Args:
            params (str or list): 'all', name of a param or list of params.
            available_params (iterable): List of acceptable values for those params.

        Returns:
            list: List with the chosen params.
        """

        if isinstance(params, str):
            if params == 'all':
                return available_params
            elif params in available_params:
                return [params]
        elif isinstance(params, list):
            for p in params:
                assert(p in available_params)
            return params
        else:
            raise ValueError

    def __assert_params(self, params):
        """Checks if the parameters for each method are correct.

        Args:
            params (dict): Dict with keys as the names of each method and values as lists
                of dicts of keyword arguments that method receives.
        """
        for method in self.methods:
            assert(method in params)
            assert(isinstance(params[method], list))
