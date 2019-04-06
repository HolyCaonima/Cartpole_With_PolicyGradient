import gym
import Agent
import random

env = gym.make("CartPole-v1")
Ag = Agent.PolicyBasedNaiveAg(2, 4)

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
        action = 0 if random.uniform(0,1) < ActionProp[0][0] else 1
        O_array.append(Observation)
        actionOut = [0, 0]
        actionOut[action] = 1
        A_array.append(actionOut)
        NewObservation, reward, done, info = env.step(action)
        AccReward += reward
        Observation = NewObservation
        R_array.append([reward])
        if done:
            break
    if AccReward != EpisodeTime:
        Ag.Train(O_array, A_array, R_array)
    else:
        print("Done")
        DoneTimes+=1
        
env.close()


