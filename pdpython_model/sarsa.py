import random


def init_qtable(states, n_actions, zeroes):
    iqtable = {}   # does this want to be a list or a dict?

    for i in states:
        indx = tuple(i)
        vu = []
        for j in range(n_actions):
            if zeroes:
                vu.append(0)
            else:
                vu.append(random.uniform(0.0, 1.0))

        #print('indx=', indx, 'vu=', vu)
        iqtable[indx] = vu

    return iqtable


def egreedy_action(e, qtable, current_state, paired):
    """ Qtable should be a dict keyed by the states - tuples of 7 values. """

    p = random.random()

    if p < e:
        return random.choice(["C", "D"])
    else:
        # index qtable by current_state
        if len(current_state) == 1:
            if paired:
                current_state = current_state[0]
        # print('my state:', current_state)
        current = qtable[tuple(current_state)]
        # print('qvalues:', current)
        # pick the action with the highest Q value - if indx:0, C, if indx:1, D
        if current[0] > current[1]:
            return "C"
        elif current[1] > current[0]:  # this might need to be an elif, but with two behaviours it's fine
            return "D"
        else:
            return random.choice(["C", "D"])


def sarsa_decision(alpha, epsilon, gamma):
    # initialise q table
    # choose action
    # observe reward and the resulting state (S') (check partner, new state)
    # output the next action we should take using egreedy and the next projected state
    # after action staged, in the
    return

def output_sprime(current_state, observed_action):
    """ Returns a tuple that adds the observed action to the stack. Current state should be a tuple of strings?"""
    cstate = list(current_state)
    cstate.pop(0)
    cstate.append(observed_action)
    return tuple(cstate)

def decay_value(initial, current, max_round, linear, floor):
    # Version WITHOUT simulated annealing, though that could be in V2
    if linear:
        increment = initial / max_round
        new_value = current - increment
        if new_value < floor:
            if floor > 0:
                return floor
            else:
                return new_value
        else:
            return new_value
    else:
        increment = current / 50        # any better value to use rather than arbitrary?
        new_value = current - increment
        if new_value > floor:
            return floor
        else:
            return new_value


def update_q(reward, gamma, alpha, oldQ, nextQ):
    """ Where nextQ is determined by the epsilon greedy choice that has already been made. """

    newQ = oldQ + alpha * (reward + ((gamma*nextQ) - oldQ))

    return newQ


# to integrate SVO into sarsa, we could try two methods
# a: weight the value of rewards received in each state my your orientation
