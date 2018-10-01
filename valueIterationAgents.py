# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections
import logging
logger = logging.getLogger('agentLogger')

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        # print('===================== in runValueIteration===================')
        current_iteration = 1
        states = self.mdp.getStates()
        values_dict = util.Counter()
        # initialize all values to 0
        for state in states:
            values_dict[str(state)+'0'] = 0
        # print(values_dict)
        # run value iteration
        while current_iteration <= self.iterations:
            # print('===================== iteration number ', current_iteration ,'===================')
            for state in states:
                # print('state is',state)
                if state == 'TERMINAL_STATE': continue
                temp_values = util.Counter()
                # print ('possible actions are', self.mdp.getPossibleActions(state))
                for action in self.mdp.getPossibleActions(state):
                    # print('action is',action)
                    temp_state_value = 0
                    for nextState,prob in self.mdp.getTransitionStatesAndProbs(state,action):
                        # print('next state is', nextState, 'and prob is', prob)
                        # print('values_dict is ',str(state)+str(current_iteration-1),
                        #       'and values inside is', values_dict[str(state)+str(current_iteration-1)])
                        # print('reward function is', self.mdp.getReward(state, action,nextState))
                        if self.mdp.isTerminal(nextState):
                            temp_state_value += prob * (self.mdp.getReward(state, action, nextState))
                        else:
                            temp_state_value+= prob*(self.mdp.getReward(state, action,nextState)
                                               + self.discount*values_dict[str(nextState)+str(current_iteration-1)])
                        # print('temp_state_value is',temp_state_value)

                    temp_values[action] = temp_state_value
                # print('dictionary is ', temp_values)
                # if no action, return value to be 0
                max_action = max(temp_values, default = None, key=temp_values.get)
                max_value = temp_values[max_action]
                values_dict[str(state)+str(current_iteration)] = max_value
            current_iteration +=1
        for state in states:
            self.values[state] = values_dict[str(state)+str(self.iterations)]


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        # print('===================== in computeQValueFromValues ===================')
        # print('state and action are', state,action)
        qValue = 0
        for nextState, nextProb in self.mdp.getTransitionStatesAndProbs(state,action):
            # print('next state and prob is',nextState,nextProb)
            qValue += nextProb*(self.mdp.getReward(state,action,nextState) + self.discount * self.values[nextState])
        return qValue

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        legal_actions = self.mdp.getPossibleActions(state)
        if len(legal_actions) == 0: return None
        values_dict = util.Counter()
        for action in legal_actions:
            temp_state_value = 0
            for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                temp_state_value += prob * (self.mdp.getReward(state,action,nextState)
                                            + self.discount * self.getValue(nextState))
            values_dict[action] = temp_state_value
        return values_dict.argMax()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)
        self.runValueIteration()

    def runValueIteration(self):
        current_iteration = 0
        states = self.mdp.getStates()
        # run value iteration
        print(self.iterations)
        while current_iteration < self.iterations:
            temp_values = self.values.copy()
            state = states[current_iteration%len(states)]
            if state == 'TERMINAL_STATE': continue
            # print ('possible actions are', self.mdp.getPossibleActions(state))
            next_state_dict = {}
            for action in self.mdp.getPossibleActions(state):
                # print('action is',action)
                temp_state_value = 0
                for next_state,prob in self.mdp.getTransitionStatesAndProbs(state,action):
                    # print('next state is', nextState, 'and prob is', prob)
                    # print('values_dict is ',str(state)+str(current_iteration-1),
                    #       'and values inside is', values_dict[str(state)+str(current_iteration-1)])
                    # print('reward function is', self.mdp.getReward(state, action,nextState))
                    if self.mdp.isTerminal(next_state):
                        temp_state_value += prob * (self.mdp.getReward(state, action, next_state))
                    else:
                        temp_state_value += prob*(self.mdp.getReward(state, action,next_state)
                                           + self.discount*temp_values[next_state])
                    # print('temp_state_value is',temp_state_value)
                next_state_dict[action] = temp_state_value
            # if no action, return value to be 0
            max_value = max(next_state_dict.values())
            self.values[state] = max_value
            current_iteration +=1


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

