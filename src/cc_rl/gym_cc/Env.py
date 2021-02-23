import gym
from gym import spaces
import numpy as np

from cc_rl.gym_cc.Renderer import Renderer


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
  def __init__(self, classifier_chain, x, display='none', random_seed=42):
    '''
    Environment constructor
    Args:
      classifier_chain : Classifier Chain used in the environment
      x : the dataset that we are working with
      random_seed : a random_seed for reproducibility
    '''
    # Passing the seed
    np.random.seed(random_seed)

    self.classifier_chain = classifier_chain
    self.action_space = [-1, 1]

    self.path = np.zeros((classifier_chain.n_labels,), dtype=int)
    self.probabilities = np.zeros((classifier_chain.n_labels,), dtype=float)
    self.obs = None
    self.current_estimator = 0
    self.current_probability = 1
    self.x = x
    self.cur_sample = 0
    self.cur_x = self.x[self.cur_sample]

    self.renderer = Renderer(display, classifier_chain.n_labels + 1)

  def _next_observation(self):
    '''
    Return the new observation
    '''
    self.current_estimator += 1

    xy = np.append(self.cur_x, self.path[:self.current_estimator])

    return self.classifier_chain.cc.estimators_[self.current_estimator].predict_proba(xy.reshape(1,-1)).flatten()

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

    # append last observation
    self.path[self.current_estimator] = action

    # We append the last chosen probability
    
    #self.probabilities[self.current_estimator] = self.obs[0]
    self.probabilities[self.current_estimator] = self.obs[(action + 1) // 2]
    self.current_probability *= self.obs[(action + 1) // 2]

    self.renderer.step(action, self.obs[(action + 1) // 2])

    if self.current_estimator == self.classifier_chain.n_labels - 1:
      self.current_probability *= self.obs[(action + 1) // 2]
      return self.obs, self.path, self.probabilities, self.current_probability, True 

    else:
      # Take new observation
      self.obs = self._next_observation()
      return self.obs, self.path, self.probabilities, 0, False

  def reset(self, label=0):
    '''
    Resets the environment
    Args:
      label: optional parameter, which is passed in order to return to a specific label 
              by undoing all decisions done after this label
    Returns:
      Next left probability
      The action history
      The chosen probabilities history
    '''
    self.current_estimator = label

    # self.current_probability = np.prod(np.abs(((1 + self.path) // 2 - self.probabilities))[:label])
    self.current_probability = np.prod(self.probabilities[:label])
    
    # Update path and probabilities
    self.path = np.append(self.path[:label], 
        np.zeros((self.classifier_chain.n_labels - label,), dtype=int))
    self.probabilities = np.append(self.probabilities[:label],
        np.zeros((self.classifier_chain.n_labels - label,), dtype=float))

    self.renderer.reset(label)

    # Get observation
    xy = np.append(self.cur_x, self.path[:label])
    self.obs = self.classifier_chain.cc.estimators_[self.current_estimator].predict_proba(xy.reshape(1,-1)).flatten()

    return self.obs, self.path, self.probabilities  
  
  def next_sample(self):
    self.cur_sample += 1
    self.cur_x = self.x[self.cur_sample]
    self.reset()
    self.renderer.next_sample()
