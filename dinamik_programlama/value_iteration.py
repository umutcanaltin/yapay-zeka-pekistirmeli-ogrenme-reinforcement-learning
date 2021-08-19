# https://www.udemy.com/course/yapay-zeka-pekistirmeli-ogrenme-reinforcement-learning/ 
# Udemy AR AiTech Pekiştirmeli Öğrenme Kursu Kodlama Örnekleri -1 Dinamik Programlama   / Politika İterayonu   /  Değer İterayonu

import random
import numpy as np
def value_iteration(env,theta=0.001, discount_factor=1):
  V = np.zeros(env.state_space)
  while True:
    delta = 0
    for s in range(env.state_space):
      A = np.zeros(env.action_space)
      for a in range(env.action_space):
        prob, next_state, reward , done = env.next_state_probabilities()[s][a]
        A[a] += prob * (reward + discount_factor*V[next_state])
      best_action = max(A)
      delta = max(delta, np.abs(best_action - V[s]))
      V[s] = best_action
    if(delta< theta):
      break
  
  policy = np.zeros([env.state_space,env.action_space])
  for s in range(env.state_space):
    A = np.zeros(env.action_space)
    for a in range(env.action_space):
      prob, next_state, reward , done = env.next_state_probabilities()[s][a]
      A[a] += prob * (reward + discount_factor*V[next_state])
    best_a = np.argmax(A)
    policy[s][best_a] = 1
  return policy, V
env = GridworldEnv()
print(value_iteration(env))
