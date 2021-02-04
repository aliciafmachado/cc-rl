import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym.envs.toy_text import discrete
import numpy as np

from src.gym_cc.Renderer import Renderer

def import_tree():
  '''

  '''
  pass

def import_classifier_chain():
  '''
  '''
  pass

def generate_random_tree():
  '''
  '''
  pass

class Env(gym.Env):
  '''
  '''
  def __init__(self, classifier_chain):
    self.classifier_chain = classifier_chain
    self.action_space = spaces.Discrete(2)
    self.observation_path_space = spaces.MultiDiscrete(np.ones((classifier_chain.n_labels,), dtype=int) * 2)
    self.observation_probabilities_space = spaces.Box(low=0, high=1, shape=(classifier_chain.n_labels,), dtype=np.float16)

    self.path = np.zeros((classifier_chain.n_labels,), dtype=bool)
    self.probabilities = np.zeros((classifier_chain.n_labels,), dtype=float)
    self.current_estimator = 0
    self.current_probability = 1

    self.renderer = Renderer('print', self.observation_path_space, self.observation_probabilities_space)

  def _take_action(self, action):
    '''
    Take the action in the environment
    '''
    # TODO

  def _next_observation(self):
    '''
    Return the observation
    '''
    # TODO
    return obs

  def step(self, action):
    # Execute the action
    self._take_action(action)

    if self.current_step == self.classifier_chain.n_labels:
      return self.next_observation(), self.current_probability, True, {} 

    else:
      return None, 0, False, {}  

  def reset(self):
    self.current_probability = 1
    self.current_step = 0
    # TODO

  def render(self):
    self.renderer.render()
