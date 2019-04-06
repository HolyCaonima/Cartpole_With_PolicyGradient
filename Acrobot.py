import gym
import Agent
import random
import numpy as np

def RandomActionDiscrete(ActionProp):
    ActionSize = len(ActionProp)
    ActionOut = np.zeros(ActionSize)
    rand = random.uniform(0,1)
    for ActionIndex in range(ActionSize):
        if rand < ActionProp[ActionIndex]:
            ActionOut[ActionIndex] = 1
            return ActionIndex, ActionOut
    ActionOut[ActionSize-1] = 1
    return ActionSize-1, ActionOut

env = gym.make("Acrobot-v1")
Ag = Agent.ACPGAg(3, 6)

CurrentEpisode = 0
while True:
    Observation = env.reset()
    O_array = []
    R_array = []
    A_array = []
    CurrentEpisode += 1
    rememberTimeStep = 0
    StepTime = 0
    score = 0
    while True:
        StepTime += 1
        if CurrentEpisode%100 == 0:
            env.render()
        rememberTimeStep += 1
        ActionProp = Ag.Predict([Observation])
        action, action_list = RandomActionDiscrete(ActionProp[0])
        O_array.append(Observation)
        A_array.append(action_list)
        NewObservation, reward, done, info = env.step(action)

        score += reward
        reward += score / 100 # Reward will be the accumulative score divied by 100
        print(reward)
        if done and StepTime < 1000:
            reward = 1000 # If make it, send a big reward
            rememberTimeStep = 200

        Observation = NewObservation
        R_array.append([reward])
        if rememberTimeStep > 200:
            print("Train")
            Ag.Train(O_array, A_array, R_array, Observation)
            O_array = []
            R_array = []
            A_array = []
            rememberTimeStep = 0
        if done:
            break
        
env.close()
