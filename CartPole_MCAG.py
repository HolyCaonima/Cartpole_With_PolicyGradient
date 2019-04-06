import gym
import Agent
import random
import numpy as np

def RandomActionDiscrete(ActionProp):
    ActionSize = len(ActionProp)
    Action = np.random.choice(np.arange(ActionSize), p=ActionProp)
    ActionOut = np.zeros(ActionSize)
    ActionOut[Action] = 1
    return Action, ActionOut

env = gym.make("CartPole-v1")
Ag = Agent.MCPGAg(2, 4)

EpisodeTime = 200
CurrentEpisode = 0
DoneTimes=0
while True:
    Observation = env.reset()
    O_array = []
    R_array = []
    A_array = []
    AccReward = 0
    CurrentEpisode += 1
    if CurrentEpisode %50 == 0:
        print(DoneTimes)
        DoneTimes=0
    for t in range(EpisodeTime):
        if CurrentEpisode%50 == 0:
            env.render()
        ActionProp = Ag.Predict([Observation])
        action, action_list = RandomActionDiscrete(ActionProp[0])
        O_array.append(Observation)
        A_array.append(action_list)
        NewObservation, reward, done, info = env.step(action)
        AccReward += reward
        Observation = NewObservation
        R_array.append([reward])
        if done:
            break
    if AccReward != EpisodeTime:
        Ag.Train(O_array, A_array, R_array, Observation)
    else:
        print("Done")
        DoneTimes+=1
        
env.close()


