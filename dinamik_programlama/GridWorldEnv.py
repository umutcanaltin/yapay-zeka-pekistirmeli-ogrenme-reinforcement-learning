# https://www.udemy.com/course/yapay-zeka-pekistirmeli-ogrenme-reinforcement-learning/ 
# Udemy AR AiTech Pekiştirmeli Öğrenme Kursu Kodlama Örnekleri -1 Dinamik Programlama   / Politika İterayonu   /  Değer İterayonu

import random
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

class GridworldEnv():
  def __init__(self):
    self.shape = [4,4]
    self.state_space = 16
    self.action_space = 4


  def next_state_probabilities(self):
    next_state_probs = []
    for state_number in range(self.state_space):
      state_prob = []
      if(state_number == 0 or state_number == 15):
        done = True
        reward = 0
        for action_pos in range(self.action_space):
          state_prob.append((1,state_number, reward, done))
        next_state_probs.append(state_prob)
      else:
        for action_pos in range(self.action_space):
          done = False
          reward = -1
          if(action_pos == 3):
            if(state_number % 4 == 0):
              ns = state_number
            else:
              ns = state_number -1 
          if(action_pos == 1):
            if(state_number%4 == 3):
              ns = state_number
            else:
              ns = state_number +1
          if(action_pos == 0 ):
            if(int(state_number / 4) == 0):
              ns = state_number
            else :
              ns = state_number - 4
          if (action_pos == 2):
            if(int(state_number/4)==3):
              ns = state_number
            else:
              ns = state_number +4
          state_prob.append((1,ns,reward,done))
        next_state_probs.append(state_prob)
    return next_state_probs
