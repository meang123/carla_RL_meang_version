

"""

** for test **  : evaluate policy then caculate total rewarld GT

**linear schedule** : beta, alpha, epsilon have to convergence at the end of training
so Need adjust parameter scheduler



"""

import gym
import numpy as np

import collections

_field_names = [
    "state",
    "action",
    "reward",
    "next_state",
    "done"
]
Experience = collections.namedtuple("Experience", field_names=_field_names)

def eval_policy(policy, env,eval_episodes=10):


    avg_reward = 0.
    avg_cost=0
    for _ in range(eval_episodes):
        state =env.reset()
        done = False

        while True:

            eval_action = policy.select_action(state)
            next_state, reward, done, info = env.step(eval_action)
            env.display()
            cost = info['cost']  # collision & invasion cost


            avg_reward += reward
            avg_cost += cost
            if done:
                break

    avg_reward /= eval_episodes
    avg_cost /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: AVG reward : {avg_reward:.3f} AVG cost : {avg_cost:.3f}")
    print("---------------------------------------")
    #eval_env.close()
    return avg_reward,avg_cost

"""
td mean - td std 가 클때 알파 높이고 
작을때 알파를 낮추는 전략으로 할것이다 
constant C =5로 설정하겠다 
alpha max 0.9
alpha min 0.3

"""
class time_base_schedule(object):
    def __init__(self,alpha_max,alpha_min,total_timestep):
        self.C = 0.05
        self.total_timestep = total_timestep
        self.alpha_max = alpha_max
        self.alpha_min = alpha_min

    def value(self,cur_timestep):

        #differ = td_mean-td_std
        progress = cur_timestep / self.total_timestep
        # sigmoid 함수 입력: 진행도가 0.5일 때 중간값이 나오도록 변환
        # 진행도가 낮으면 음수, 높으면 양수가 되어 sigmoid가 0에서 1로 변환됨
        logistic_input = (progress - 0.5) / self.C
        alpha = self.alpha_min+(self.alpha_max-self.alpha_min)*(1/(1+np.exp(-logistic_input)))
        return alpha









class LinearSchedule(object):
    def __init__(self,schedule_timesteps,initial_p,final_p):
        self.schedule_timesteps = schedule_timesteps
        self.initial_p = initial_p
        self.final_p = final_p

    # 조금식 조금식 final_p로 설정한 값으로 가까워 지도록 스케줄링 하는 함수
    def value(self,t):

        fraction = min(float(t)/self.schedule_timesteps,1.0)
        return self.initial_p+fraction*(self.final_p-self.initial_p)

