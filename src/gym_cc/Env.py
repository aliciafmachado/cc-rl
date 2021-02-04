import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym.envs.toy_text import discrete
import numpy as np

from src.gym_cc.Renderer import Renderer


class Env(gym.Env):
  '''
  Classifier chains are there and you have to defeat them by 
  finding the greatest joint probability among all the possible ones
  The environment receives a classifier chain in the constructor which is
  used as the predictor
  The actions corresponds to the choice to go left (0) or right(1) in the
  tree incurred by the classifier chain
  You only receive a reward in the end and it corresponds to the final 
  joint probability
  '''
  def __init__(self, classifier_chain, dataset, random_seed=42):
    '''
    Environment constructor
    Args:
      classifier_chain : Classifier Chain used in the environment
      dataset : the dataset that we are working with
      random_seed : a random_seed for reproducibility
    '''
    # Passing the seed
    np.random.seed(random_seed)

    self.classifier_chain = classifier_chain
    self.action_space = spaces.Discrete(2)
    self.observation_path_space = spaces.MultiDiscrete(np.ones((classifier_chain.n_labels,), dtype=int) * 2)
    self.observation_probabilities_space = spaces.Box(low=0, high=1, shape=(classifier_chain.n_labels,), dtype=np.float16)

    self.path = np.zeros((classifier_chain.n_labels,), dtype=int)
    self.probabilities = np.zeros((classifier_chain.n_labels,), dtype=float)
    self.obs = None
    self.current_estimator = 0
    self.current_probability = 1
    self.dataset = dataset
    self.x = self.dataset.train_x[np.random.randint(0, len(self.dataset.train_x))]

    self.renderer = Renderer('print', self.observation_path_space, self.observation_probabilities_space)


  def _next_observation(self, action):
    '''
    Return the new observation
    '''
    if self.current_estimator > 0:
      xy = np.append(self.x, self.path[:self.current_estimator])

    else:
      xy = self.x 

    obs = self.classifier_chain.cc.estimators_[self.current_estimator].predict_proba(xy.reshape(1,-1)).flatten()
    
    self.current_estimator += 1

    return obs


  def step(self, action):
    '''
    Step in the environment
    Args:
      action : Classifier Chain used in the environment
    Returns:
      Next left probability
      The action history
      The chosen probabilities history
      Reward
      If the environment is done: if we arrived in the end of the
      classifier chain
    '''
    # Execute the action
    if self.current_estimator == self.classifier_chain.n_labels - 1:
      self.current_probability *= self.obs[action]
      return self.obs, self.path, self.probabilities, self.current_probability, True 

    else:

      self.obs = self._next_observation(action)
      
      # append last observation
      if self.current_estimator > 0:
        self.path[self.current_estimator - 1] = action

        # Passing left probability
        self.probabilities[self.current_estimator - 1] = self.obs[0]
        self.current_probability *= self.obs[action]

      return self.obs, self.path, self.probabilities, 0, False


  def reset(self):
    '''
    Resets the environment
    '''
    self.current_probability = 1
    self.current_estimator = 0
    self.path = np.zeros((classifier_chain.n_labels,), dtype=int)
    self.probabilities = np.zeros((classifier_chain.n_labels,), dtype=float)
    self.renderer.reset()

    # We reset x as well
    self.x = self.dataset.train_x(np.random.randint(0, len(self.dataset.train_x)))

  def render(self):
    self.renderer.render()
