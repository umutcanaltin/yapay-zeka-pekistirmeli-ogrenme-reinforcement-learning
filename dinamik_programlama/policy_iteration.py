# https://www.udemy.com/course/yapay-zeka-pekistirmeli-ogrenme-reinforcement-learning/ 
# Udemy AR AiTech Pekiştirmeli Öğrenme Kursu Kodlama Örnekleri -1 Dinamik Programlama   / Politika İterayonu   /  Değer İterayonu

import random
import numpy as np


def policy_evaluation(policy, env, discount_factor = 1 , theta = 0.0001):
  V = np.zeros(env.state_space)
  while True:
    delta = 0
    for s in range(env.state_space):
      v = 0
      for a, action_prob in enumerate(policy[s]):
        prob, next_state, reward, done = env.next_state_probabilities()[s][a]
        v += action_prob * prob * (reward + discount_factor*V[next_state])
      delta = max(delta, np.abs(v - V[s]))
      V[s]= v
    if(delta < theta):
      break
  return V

#env = GridworldEnv()
#policy = np.ones([env.state_space, env.action_space])/env.action_space
#print(policy_evaluation(policy,env))


def policy_improvement(env,discount_factor=1):
  policy = np.ones([env.state_space,env.action_space]) / env.action_space
  while True:
    V = policy_evaluation(policy, env)
    policy_stable = True
    for s in range(env.state_space):
      action_values = np.zeros(env.action_space)
      for a in range(env.action_space):
        prob, next_state , reward , done = env.next_state_probabilities()[s][a]
        action_values[a] += prob * (reward + discount_factor*V[next_state])
      choosen_a = np.argmax(policy[s])
      best_action = np.argmax(action_values)
      if(choosen_a != best_action):
        policy_stable = False
      policy[s] = np.eye(env.action_space)[best_action]
    if(policy_stable == True):
      return policy, V

env = GridworldEnv()
print(policy_improvement(env))
