from nptyping import NDArray
import numpy as np

from cc_rl.classifier_chain.BaseInferer import BaseInferer
from cc_rl.gym_cc.Env import Env
from cc_rl.rl.MCTSAgent import MCTSAgent
from cc_rl.rl.QAgent import QAgent


class RLInferer(BaseInferer):
    """
    Uses RL agents to learn the best inference.
    """

    def __init__(self, classifier_chain, loss: str,
                 agent_type: str, nb_sim: int, nb_paths: int, epochs: int,
                 batch_size: int = 64, learning_rate: float = 1e-3,
                 mcts_passes: int = 10):
        super().__init__(classifier_chain.cc, loss)
        self.cc = classifier_chain
        assert agent_type == 'qlearning' or agent_type == 'mcts'
        self.__agent_type = agent_type
        self.__nb_sim = nb_sim
        self.__nb_paths = nb_paths
        self.__epochs = epochs
        self.__batch_size = batch_size if batch_size is not None else 64
        self.__learning_rate = learning_rate if learning_rate is not None else 1e-3
        self.__mcts_passes = mcts_passes if mcts_passes is not None else 10

    def _infer(self, x: NDArray[float]):
        y_pred = []
        n_nodes = 0
        env = Env(self.cc, x)

        for i in range(len(x)):
            if self.__agent_type == 'qlearning':
                agent = QAgent(env)
            elif self.__agent_type == 'mcts':
                agent = MCTSAgent(env, mcts_passes=self.__mcts_passes)
            else:
                raise ValueError

            agent.train(self.__nb_sim, self.__nb_paths, self.__epochs,
                        self.__batch_size, self.__learning_rate)

            pred = agent.predict(return_num_nodes=True)
            y_pred.append(pred[0])
            n_nodes += pred[1]

            if i < len(x) - 1:
                env.next_sample()

        return np.array(y_pred, dtype=bool), n_nodes / len(x)
