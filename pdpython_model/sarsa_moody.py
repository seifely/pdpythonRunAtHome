import random
import math

def observe_state(obsAction, oppID, oppMood, stateMode):
    """Keeping this as a separate method in case we want to manipulate the observation somehow,
    like with noise (x chance we make an observational mistake, etc)."""
    # print('oppmood', oppMood)
    state = []
    if stateMode == 'stateless':
        state.append(obsAction)
    elif stateMode == 'agentstate':
        state.append(obsAction)
        state.append(oppID)
    elif stateMode == 'moodstate':
        state.append(obsAction)
        state.append(oppID)
        state.append(oppMood)
    # Returns a list, but this should be utilised as a tuple when used to key a Q value
    # print('state:', state)
    return state

def init_qtable(states, n_actions, zeroes):
    """First value is for cooperate, second value is for defect."""
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

def init_statememory(states, n_actions, delta):
    iqtable = {}   # does this want to be a list or a dict?
    vv = []
    for j in range(0, delta):
        vv.append(0)

    for i in states:
        indx = tuple(i)
        vu = []
        for j in range(n_actions):
            vu.append(vv)

        #print('indx=', indx, 'vu=', vu)
        iqtable[indx] = vu

    return iqtable

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

# def moody_action_three(mood, state, qtable, moodAffectMode, epsilon, moodAffect, turn, startingBehav):
#     """ IF USING THIS FUNCTION, USE A SMALL STARTING EPSILON AS IT USES A LARGER EPSILON VALUE SELECTIVELY """
#     change = epsilon
#     epsChange = 0  # this should stay at no change if mood isn't high or low
#     current = qtable[tuple(state)]
#
#     r = random.randint(1, 100) / 100  # Still not sure about this line and below, it might need to be if r > 50?
#     if turn == 1:
#         print("I did my starting behaviour")
#         todo = startingBehav
#     # Inititally starts with cooperation
#
#     else:
#         print("It isn't turn one, so I will pick a move")
#         # we pick a move. if that move fits the behavioural condition, then we instead perhaps explore
#         if r < (1 - change):
#             print("r (", r, ") is less than", (1-change), "so I will pick optimally")
#             # pick the optimal move
#             if current[0] > current[1]:
#                 todo = 'C'
#             else:
#                 todo = 'D'
#
#         else:
#             print("r (", r, ") is greater than", (1 - change), "so I will explore")
#             # explore
#             todo = random.choice(['C', 'D'])
#
#         print("I am originally gonna do ", todo)
#         # then we see if either of the behavioural conditions are met - i.e., were we high or low mood and D/C-ing
#         if mood > 70:
#             print("my mood is high")
#             if todo == 'D':
#                 print("and I am currently gonna defect, so let us decide again")
#                 change = moodAffect
#                 print("my epsilon is now", change)
#                 if r < (1 - change):
#                     print("r (", r, ") is smaller than the new", (1 - change), "so I will pick optimally")
#                     # pick the optimal move
#                     if current[0] > current[1]:
#                         todo = 'C'
#                     else:
#                         todo = 'D'
#
#                 else:
#                     # explore
#                     print("r (", r, ") is greater than the new", (1 - change), "so I will explore")
#                     todo = random.choice(['C', 'D'])
#                 print("I am now gonna do ", todo)
#                 return todo, epsilon
#             else:
#                 print("I didn't do the move condition even though my mood was low, so I will continue to do ", todo)
#                 return todo, epsilon
#         elif mood < 30:
#             print("my mood is low")
#             if todo == 'C':
#                 print("and I chose to cooperate so let us decide again")
#                 change = moodAffect
#                 print("my epsilon is now ", change)
#                 if r < (1 - change):
#                     print("r (", r, ") is smaller than the new", (1 - change), "so I will pick optimally")
#                     # pick the optimal move
#                     if current[0] > current[1]:
#                         todo = 'C'
#                     else:
#                         todo = 'D'
#
#                 else:
#                     # explore
#                     print("r (", r, ") is greater than the new", (1 - change), "so I will explore")
#                     todo = random.choice(['C', 'D'])
#                 print("I am now gonna do ", todo)
#                 return todo, epsilon
#             else:
#                 print("I didn't do the move condition even though my mood was low, so I will continue to do ", todo)
#                 return todo, epsilon
#         # if we weren't in either of those behavioural conditions, then we can just return the original selection
#         else:
#             print("none of these conditions applied to me, so I will continue to do ", todo)
#             return todo, epsilon


def moody_action_three(mood, state, qtable, moodAffectMode, epsilon, moodAffect):
    """ IF USING THIS FUNCTION, USE A SMALL STARTING EPSILON AS IT USES A LARGER EPSILON VALUE SELECTIVELY """
    # print(mood, state, qtable, epsilon, moodAffect, turn, startingBehav)
    change = epsilon
    epsChange = 0  # this should stay at no change if mood isn't high or low
    current = qtable[tuple(state)]

    r = random.randint(1, 100) / 100  # Still not sure about this line and below, it might need to be if r > 50?

    # Inititally starts with StartingBehav outside of this function


    # we pick a move. if that move fits the behavioural condition, then we instead perhaps explore
    if r < (1 - change):
        # pick the optimal move
        if current[0] > current[1]:
            todo = 'C'
        else:
            todo = 'D'
    else:
        # explore
        todo = random.choice(['C', 'D'])

    # then we see if either of the behavioural conditions are met - i.e., were we high or low mood and D/C-ing
    if mood > 70:
        if todo == 'D':
            change = moodAffect

            if r < (1 - change):
                # pick the optimal move
                if current[0] > current[1]:
                    todo = 'C'
                else:
                    todo = 'D'

            else:
                # explore
                todo = random.choice(['C', 'D'])
            return todo
    elif mood < 30:
        if todo == 'C':
            change = moodAffect

            if r < (1 - change):
                # pick the optimal move
                if current[0] > current[1]:
                    todo = 'C'
                else:
                    todo = 'D'

            else:
                # explore
                todo = random.choice(['C', 'D'])
            return todo
        # if we weren't in either of those behavioural conditions, then we can just return the original selection

    return todo

def moody_action_alt(mood, state, qtable, moodAffectMode, epsilon, moodAffect, turn, startingBehav):
    """ The essence of this should be that we pick an action through epsilon greedy, then
        depending on which one we picked and our mood, we change epsilon. """
    """ IF USING THIS FUNCTION, USE A LARGE EPSILON AND THEN IT DECAYS """
    r = random.randint(1, 100) / 100
    if type(state[0]) == tuple:
        state = state[0]
    else:
        state = tuple(state)
    current = qtable[state]
    todo = 'C'  # just in case the next steps mess up, let's cooperate as default
    change = 0
    # print("My initial action is:", todo)

    if turn == 1:
        todo = startingBehav
    else:
        # In the original paper, we don't explore a lot to start and then explore more as we go on (I think this is wrong
        # and should be the other way around but ah well)
        if r > epsilon:
            # Do the optimal action
            if current[1] > current[0]:
                todo = 'D'
            else:
                todo = 'C'
            # print("One was higher, so my action is now:", todo)
        elif r <= epsilon:
            todo = random.choice(['C', 'D'])
            # print("I picked randomly and my action is now:", todo)

        if mood >= 70:
            # print("My mood is high")
            if todo == 'D':
                # print("And I chose D, so I will change my epsilon")
                change = moodAffect
        if mood <= 30:
            # print("My mood is low")
            if todo == 'C':
                # print("And I chose C, so I will change my epsilon")
                change = moodAffect

    # print("Amount to change epsilon by:", change)
    newEps = epsilon - change
    # print("So my new epsilon is:", newEps)
    newEps = min(99.999, newEps)
    newEps = max(0.0001, newEps)

    return todo, newEps

def moody_action(mood, state, qtable, moodAffectMode, epsilon, moodAffect):
    """ Fixed Amount should be in the model as a test parameter MA """
    change = epsilon  # not sure why change is set to epsilon to start with --> in case none of the loops take effect
    # TODO: part of this function needs to produce an altered epsilon value
    epsChange = 0  # this should stay at no change if mood isn't high or low

    r = random.randint(1, 100) / 100  # Still not sure about this line and below, it might need to be if r > 50?
    if r > 0:
        todo = 'C'
    else:
        todo = 'D'
    # Inititally starts with cooperation

    # index qtable by current state
    current = qtable[tuple(state)]
    if current[1] > current[0]:

        if mood > 70 and moodAffectMode == 'Fixed':
            change = moodAffect
            epsChange = change
        elif mood > 70 and moodAffectMode == 'Mood':
            change = ((mood - 50) / 100)
            epsChange = change

        if r > (1-change):  # because change is epsilon, this section is the e-greedy choice making
            todo = 'D'
        else:
            todo = 'C'

    elif current[0] > current[1]:
        if mood < 30 and moodAffectMode == 'Fixed':
            change = moodAffect
            epsChange = change
        elif mood < 30 and moodAffectMode == 'Mood':
            change = ((50 - mood) / 100)
            epsChange = change

        if r < (1-change):
            todo = 'C'
        else:
            todo = 'D'

    newEps = epsilon + epsChange
    # print(newEps)
    return todo, newEps

def getMoodType(mood):
    if mood > 70:
        return 'HIGH'
    elif mood < 30:
        return 'LOW'
    else:
        return 'NEUTRAL'

# def egreedy_action(e, qtable, current_state, paired):
#     """ Qtable should be a dict keyed by the states - tuples of 7 values. """
#
#     p = random.random()
#
#     if p < e:
#         return random.choice(["C", "D"])
#     else:
#         # index qtable by current_state
#         if len(current_state) == 1:
#             if paired:
#                 current_state = current_state[0]
#         # print('my state:', current_state)
#         # print("q", qtable)
#         current = qtable[tuple(current_state)]
#         # print('qvalues:', current)
#         # pick the action with the highest Q value - if indx:0, C, if indx:1, D
#         if current[0] > current[1]:
#             return "C"
#         elif current[1] > current[0]:  # this might need to be an elif, but with two behaviours it's fine
#             return "D"
#         else:
#             return random.choice(["C", "D"])


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

def update_q(step_number, actionTaken, state, qtable, payoff, memory, currMood, alpha, gamma):
    # TODO: Need to manage memory outside of this function, as we do with working memory /
    # TODO: Needs to be a NEW memory of payoffs gained instead of moves observed
    """THIS RETURNS UPDATED Q VALUES FROM OLD Q VALUES"""
    current = qtable[tuple(state)]
    if step_number is not None:
        if step_number is not 0:
            if actionTaken == 'C':
                current[0] = learn(current[0], payoff, memory, currMood, alpha, gamma)
                return current[0], current[1]
            else:
                current[1] = learn(current[1], payoff, memory, currMood, alpha, gamma)
                return current[0], current[1]
        else:
            if actionTaken == 'C':
                current[0] = payoff
                current[1] = 0
                return current[0], current[1]
            else:
                current[0] = 0
                current[1] = payoff
                return current[0], current[1]
    else:
        if actionTaken == 'C':
            current[0] = payoff
            current[1] = 0
            return current[0], current[1]
        else:
            current[0] = 0
            current[1] = payoff
            return current[0], current[1]


# def update_q(reward, gamma, alpha, oldQ, nextQ):
#     """ Where nextQ is determined by the epsilon greedy choice that has already been made. """
#
#     newQ = oldQ + alpha * (reward + ((gamma*nextQ) - oldQ))
#     return newQ
#

def learn(oldQ, reward, memory, mood, alpha, gamma):
    newQ = oldQ + alpha * (reward + (gamma*estimateFutureRewards(mood, memory)) - oldQ)
    return newQ

def estimateFutureRewards(mood, memory):
    percentToLookAt = (100 - mood) / 100
    actualAmount = math.ceil((len(memory) * percentToLookAt))

    tot = sum(memory[0:actualAmount+1])
    return tot/actualAmount


# def update_mood(currentmood, score, averageScore, oppScore, oppAverage):
#     ab = (100 - currentmood) / 100
#     u = averageScore - ((ab * max((oppAverage - averageScore), 0)) - (ab * max((averageScore - oppAverage), 0)))
#     dif = score - u
#
#     newMood = min(99.999, (currentmood + dif))
#     newMood = max(0.0001, newMood)
#     print('Mood:', newMood)
#     return newMood

def update_mood_old(currentmood, score, averageScore, oppScore, oppAverage, sensitive, sensitivity):
    ab = (100 - currentmood) / 100
    omega = averageScore - ((ab * max((oppAverage - averageScore), 0)) - (ab * max((averageScore - oppAverage), 0)))  # perceived payoff
    dif = score - omega  # difference between the score and the perceived payoff
    adjustment = (score - averageScore) + omega
    if sensitive:
        if adjustment < 0:
            """Sensitivity should be used, which has been building up on every turn since a negative outcome.
                            Once used, it then resets the clock. """
            sensitivity = min(20.00, sensitivity)
            adjustment = adjustment * sensitivity
            sensitivity = 0
        else:
            """This linearly increases the sensitivity for each turn it isn't activated."""
            sensitivity += 1
    newMood = currentmood + adjustment
    newMood = min(99.999, newMood)
    newMood = max(0.0001, newMood)
    # print('Mood:', newMood)
    return newMood, sensitivity

def update_mood_new(currentmood, score, averageScore, oppScore, oppAverage, sensitive, sensitivity):
    ab = (100 - currentmood) / 100
    omega = averageScore - ((ab * max((oppAverage - averageScore), 0)) - (ab * max((averageScore - oppAverage), 0)))  # perceived payoff
    dif = score - omega  # difference between the score and the perceived payoff
    adjustment = (score - omega)
    if sensitive:
        if adjustment < 0:
            """Sensitivity should be used, which has been building up on every turn since a negative outcome.
                Once used, it then resets the clock. """
            sensitivity = min(20.00, sensitivity)
            adjustment = adjustment * sensitivity
            sensitivity = 0
        else:
            """This linearly increases the sensitivity for each turn it isn't activated."""
            sensitivity += 1
    newMood = currentmood + adjustment
    newMood = min(99.999, newMood)
    newMood = max(0.0001, newMood)
    # print('Mood:', newMood)
    return newMood, sensitivity


def get_payoff(myMove, oppMove, CCpayoff, DDpayoff, CDpayoff, DCpayoff):
    outcome = [myMove, oppMove]

    if outcome == ['C', 'C']:
        return CCpayoff
    elif outcome == ['C', 'D']:
        return CDpayoff
    elif outcome == ['D', 'C']:
        return DCpayoff
    else:
        return DDpayoff


def payoff_matrix_generator(value_range, direction, critical_ci):
    for i in value_range:
        for j in value_range:
            for k in value_range:
                for n in value_range:
                    T = i
                    R = j
                    P = k
                    S = n

                    if T > R:
                        if R > P:
                            if P > S:
                                # then we can begin

                                if 2*R > (T+S):
                                    cooperation_index = (R-P)/(T-S)
                                    if direction == "above":
                                        if cooperation_index >= critical_ci:
                                            print("T:", T, "R:", R, "P:", P, "S:", S, "and the CI = ",
                                                  cooperation_index)
                                    elif direction == "below":
                                        if cooperation_index <= critical_ci:
                                            print("T:", T, "R:", R, "P:", P, "S:", S, "and the CI = ", cooperation_index)
                                    elif direction == "equals":
                                        if cooperation_index == critical_ci:
                                            print("T:", T, "R:", R, "P:", P, "S:", S, "and the CI = ", cooperation_index)

# to integrate SVO into sarsa, we could try two methods
# a: weight the value of rewards received in each state my your orientation
