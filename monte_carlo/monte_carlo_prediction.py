# https://www.udemy.com/course/yapay-zeka-pekistirmeli-ogrenme-reinforcement-learning/ 
# Udemy AR AiTech Pekiştirmeli Öğrenme Kursu Kodlama Örnekleri -1 Dinamik Programlama   / Politika İterayonu   /  Değer İterayonu
import numpy as np
import gym
env_frozen = gym.make("FrozenLake-v0")
def sample_policy(env):
  return  env.action_space.sample()

def mc_prediction(policy , env, num_episode):
  sum_returns = np.zeros(env.observation_space.n)
  count_returns = np.zeros(env.observation_space.n)
  V = np.zeros(env.observation_space.n)
  for number_of_eps in range(num_episode):
    episode = []
    state =  env.reset()
    for t in range(50):
      action = policy(env)
      next_state , reward , done , _ = env.step(action)
      episode.append((state,reward))

      if done:
        break
      state=next_state
    state_visit = np.zeros(env.observation_space.n)
    for state_num in range(len(episode)):
      state_number = episode[state_num][0]
      if(state_visit[state_number] == 0):
        G = sum([i[1] for i in episode[state_num:]])
        sum_returns[state_number] += G
        count_returns += 1
        V[state_number] = sum_returns[state_number] / count_returns[state_number]
        state_visit[state_number] = 1
        
  return V


print(mc_prediction(policy = sample_policy, env= env_frozen , num_episode = 10000 ))
