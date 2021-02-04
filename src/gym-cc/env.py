import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym.envs.toy_text import discrete

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
    self.current_step = 0
    self.current_probability = 1
    self.action_space = spaces.Discrete(2)
    self.observation_path_space = [spaces.Discrete(2) for e in classifier_chain.n_estimators]
    self.observation_probabilities_space = spaces.Box(low=0, high=1, shape=(classifier_chain.n_estimators,), dtype=np.float16)

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


  def render(self, mode='human'):
    pass
