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

class ClEnv(discrete.DiscreteEnv):
  '''
  '''
  metadata = {'render.modes': ['human']}

  def __init__(self, classifier_chain=None, raw_tree=None):
    if classifier_chain != None:
      desc = import_classifier_chain()
    else:
      if raw_tree != None:
        desc = import_tree()
      else:
        desc = generate_random_tree()

    
    ...
  def step(self, action):
    ...
  def reset(self):
    ...
  def render(self, mode='human'):
    ...