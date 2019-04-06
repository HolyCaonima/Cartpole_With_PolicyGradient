import tensorflow as tf
import numpy as np

#J(Theta) = V(S1) = E[v1]

class PolicyBasedNaiveAg:

    def __init__(self, ActionSpace, StateSpace):
        self.sess = tf.Session()
        self.Gamma = 0.97
        self.StateSpace = StateSpace
        self.ActionSpace = ActionSpace
        self.StateInput = tf.placeholder(tf.float32, shape=[None, StateSpace])
        self.AdvantageInput = tf.placeholder(tf.float32, shape=[None, 1])
        self.ActionInput = tf.placeholder(tf.float32, shape=[None, ActionSpace])
        self.ActionValueInput = tf.placeholder(tf.float32, shape=[None, 1])
        self.Policy = self.MakePolicy()
        self.ValueEst = self.MakeStateValue()
        self.NegJ = self.MakeObject()
        self.OptPolicy = self.OptPolicyTarget()
        self.OptValue = self.OptValueTarget()
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def MakePolicy(self):
        FC_0 = tf.layers.dense(self.StateInput, units=100, activation=tf.nn.relu)
        FC_1 = tf.layers.dense(FC_0, units=self.ActionSpace, activation=None)
        Out = tf.nn.softmax(FC_1)
        return Out

    def MakeStateValue(self):
        FC_0 = tf.layers.dense(self.StateInput, units=100, activation=tf.nn.relu)
        FC_1 = tf.layers.dense(FC_0, units=1, activation=None)
        return FC_1

    def MakeObject(self):
        ActionPropInPolicy = tf.reduce_sum(tf.multiply(self.Policy, self.ActionInput), reduction_indices=[1])
        RewardInAction = tf.log(ActionPropInPolicy) * self.AdvantageInput
        return tf.reduce_sum(-RewardInAction)
    
    def OptValueTarget(self):
        loss = tf.nn.l2_loss(self.ValueEst - self.ActionValueInput)
        return tf.train.AdamOptimizer(0.01).minimize(loss)

    def OptPolicyTarget(self):
        return tf.train.AdamOptimizer(0.01).minimize(self.NegJ)

    def Train(self, State, Action, Reward):
        EpisodeSize = len(State)
        Advangets = []
        NewActionValue = []
        for TimeStep in range(EpisodeSize):
            FutureReward = 0
            DecreaseFactor = 1
            FutureSteps = EpisodeSize - TimeStep
            # calculate discounted monte-carlo return
            for FutureTimeStep in range(FutureSteps):
                FutureReward += Reward[TimeStep + FutureTimeStep][0] * DecreaseFactor
                DecreaseFactor *= self.Gamma
            
            EstimateReward = self.sess.run(self.ValueEst, feed_dict={self.StateInput:[State[TimeStep]]})[0][0]
            Advangets.append([FutureReward - EstimateReward])
            NewActionValue.append([FutureReward])

        self.sess.run(self.OptPolicy, feed_dict={self.StateInput:State, self.ActionInput:Action, self.AdvantageInput:Advangets})
        self.sess.run(self.OptValue, feed_dict={self.StateInput:State, self.ActionValueInput:NewActionValue})

    def Predict(self, State):
        return self.sess.run(self.Policy, feed_dict={self.StateInput:State})

class PolicyBasedAg:

    def __init__(self):
        pass

    def InitModel(self):
        pass

class ActorCriticAg:

    def __init__(self):
        pass



class PGAgent:

    def __init__(self):
        pass

    def InitModel(self):
        pass