from mesa import Agent
import random
import csv
import numpy as np
import statistics
import pickle
from scipy.spatial import distance as dst
import copy
import sarsa
import sarsa_moody
import math
import random_network_functions as rnf

"""Note on Strategies:
    RANDOM - Does what it says on the tin, each turn a random move is selected.
    EV - Expected Value. Has a static probability of prediction of what other partner
        will do, and picks the highest expected VALUE from those.
    VEV - Variable Expected Value. Agent reacts to partner's previous move, through altering EV probabilities.
    VPP - Variable Personal Probability. Agent changes own likelihood that it will defect in response to defection.
    ANGEL - Always co-operate.
    DEVIL - Always defect.
    
    LEARN - SARSA strategy for learning.
    MOODYLEARN - A moody SARSA variant for learning.

    TFT - The classic Tit for Tat strategy.
    WSLS - Win Stay Lose Switch.
    iWSLS - An alternative implementation of WSLS for comparison."""


class PDAgent(Agent):
    def __init__(self, pos, model,
                 stepcount=0,
                 pick_strat="RDISTRO",
                 strategy=None,
                 starting_move=None,
                 checkerboard=False,
                 lineplace=False,
                 ):
        super().__init__(pos, model)
        """ To set a heterogeneous strategy for all agents to follow, use strategy. If agents 
            are to spawn along a distribution, set number of strategy types, or with
            random strategies, use pick_strat and set strategy to None """

        self.pos = pos
        self.stepCount = stepcount
        self.ID = self.model.agentIDs.pop(0)
        self.score = 0
        self.strategy = strategy
        self.filename = ('%s agent %d.csv' % (self.model.exp_n, self.ID), "a")
        self.previous_moves = []
        self.pickstrat = pick_strat
        self.checkerboard = checkerboard
        self.lineplace = lineplace

        self.update_values = {}
        self.update_value = 0.015  # this is the value that will change each turn
        self.theta = 0.015  # uv we manipulate, stays static

        self.delta = self.model.msize  # max memory size
        self.moody_delta = self.model.moody_msize  # max memory size for moody
        if self.model.memoryPaired:
            if self.model.msize == 1:
                self.delta = 1
            if self.model.msize == 7:
                self.delta = 4
            #if self.model.msize == 2:
                #self.delta = 1
            #TODO: Figure out a consistent memory size metric
        if self.model.moody_memoryPaired:
            if self.model.moody_msize == 1:
                self.moody_delta = 1
            if self.model.moody_msize == 7:
                self.moody_delta = 4
            #if self.model.moody_msize == 2:
                #self.moody_delta = 1
            #TODO: Figure out a consistent memory size metric

        #print('delta:', self.delta)

        self.init_uv = self.model.theta
        # self.init_ppD = model.init_ppD  # this isn't actually used

        self.move = None
        self.next_move = None
        self.printing = self.model.agent_printing
        if starting_move:
            self.move = starting_move
        else:
            self.move = self.random.choice(["C", "D"])

        self.payoffs = self.model.payoffs

        # ------------------------ LOCAL MEMORY --------------------------
        self.partner_IDs = []
        self.partner_moves = {}
        self.ppD_partner = 0
        self.per_partner_payoffs = {}  # this should be list of all prev payoffs from my partner, only used fr averaging
        self.partner_latest_move = {}  # this is a popped list
        #self.my_latest_move = {} # this is a popped list
        self.partner_scores = {}
        self.default_ppds = {}
        self.training_data = []
        self.per_partner_utility = {}
        self.per_partner_mcoops = {}  # MY cooperations that I perform
        self.per_partner_tcoops = {}  # THEIR cooperations that THEY perform
        self.per_partner_mutc = {}   # cumulative number of mutual cooperations me and each partner did
        self.per_partner_strategies = {}
        self.similar_partners = 0
        self.outcome_list = {}
        self.itermove_result = {}
        self.common_move = ""
        self.last_round = False
        self.wsls_failed = False

        self.globalAvPayoff = 0
        self.globalHighPayoff = 0  # effectively the highscore
        self.indivAvPayoff = {}
        self.oppAvPayoff = {}  # a store of our opponents' average scored against us
        self.proportional_score = 0  # this is used by the visualiser

        # self.average_payoff = 0  # should this be across partners or between them?

        self.working_memory = {}  # this is a popped list of size self.delta

        # ----------------------- SARSA GLOBALS ---------------------------

        self.states = []
        self.pp_sprime = {}
        self.pp_aprime = {}
        self.pp_payoff = {}
        self.oldstates = {}

        self.epsilon = copy.deepcopy(self.model.epsilon)
        self.alpha = copy.deepcopy(self.model.alpha)
        self.gamma = copy.deepcopy(self.model.gamma)

        # Per Partner Q Tables
        self.qtable = []

        # ----------------------- MOODY SARSA GLOBALS ---------------------------
        self.mood = self.model.moody_startmood  # A value between 0 and 100

        self.moody_states = []
        self.moody_pp_sprime = {}
        self.moody_pp_aprime = {}
        self.moody_pp_payoff = {}
        self.moody_pp_oppPayoff = {}
        self.pp_oppPayoff = {}  #  a version of the above (history of what my opponent scored against me) for normies to use  # I think this is just one round long...
        self.moody_oldstates = {}

        self.moody_epsilon = copy.deepcopy(self.model.moody_epsilon)
        self.moody_alpha = copy.deepcopy(self.model.moody_alpha)
        self.moody_gamma = copy.deepcopy(self.model.moody_gamma)

        self.moody_qtable = []
        self.partner_moods = {}
        self.statemode = self.model.moody_statemode
        self.partner_states = {}

        self.state_working_memory = {}  # The paired (s,a) memory used to calculate psi for estimated payoff


        # ------------------------ SENSITIVITY ----------------------------

        self.sensitivity_mod = 20  # Initialise this at full, naive and not expecting betrayal
        if self.ID in list(self.model.sensitive_agents):
            self.sensitive = True
        else:
            self.sensitive = False

        # ----------------------- SVO GLOBALS ------------------------

        # -------------------- RANDOM NETWORK GLOBALS --------------------

        self.current_partner_list = []  # List of current partners we are playing against
        self.current_partner_reputations = {}  # Mean scores against the agents we are currently playing against

        self.potential_partner_list = []  # Agents we are not currently playing against
        self.potential_partner_reputations = {}  # A dict of all agent's average past scores against us?

        self.all_possible_partners = []  # List of all agent IDs except my own
        self.all_possible_partners = list(range(1, (self.model.number_of_agents + 1)))
        if self.ID in self.all_possible_partners:
            self.all_possible_partners.remove(self.ID)

        self.rejected_partner_list = []  # Agents we have played against in the past and dropped
        self.rejected_partner_reputations = {}  # Their reputations/scores

        for possible in self.model.agentIDs:
            self.potential_partner_reputations[possible] = 0  # Initialise the reps at zero

        self.actorDegreeCentrality = 0
        self.normalizedActorDegreeCentrality = 0
        self.utilityRatio = 0  # ratio of utility earned to number of partners
        self.payoffRatio = 0   # average payoff that round
        self.avPayoff = 0
        self.pp_UR = {}  # Each of my Partner's Utility Ratios (utility / n_partners)
        self.pp_PR = {}  # Each of my Partner's Payoff Ratios (payoff / n partners)
        self.pp_CON = {} # Each of my Partner's connectednesses
        self.average_ppUR = 0
        self.average_ppPR = 0
        self.average_ppCON = 0

        self.betrayals = 0
        # the number of times I have betrayed someone else who was cooperating with me
        self.partnerSelectionStrat = self.model.selectionStrategy
        # Options for the above should be "DEFAULT", "SCORE", "REP",
        # Default is random partner changes
        self.currentForgiveness = copy.deepcopy(self.model.forgivenessPeriod)
        # initialises and then decreases as each round goes by
        #self.agentsToRestructure = 0  # TODO: This needs to be a random number within some parameters
        self.Switcher = False
        # check for this value from the model each round, and see if we are allowed to switch partners or not


        # ----------------------- DATA TO OUTPUT --------------------------
        self.number_of_c = 0
        self.number_of_d = 0
        self.mutual_c_outcome = 0
        self.n_partners = 0

        self.default_attempt = 0
        self.attempts_taken = 0

        # ----------------------- INTERACTIVE VARIABLES ----------------------
        # these values are increased if partner defects. ppC for each is 1 - ppD
        self.ppD_partner = {}
        self.rounds_left = self.model.rounds - self.stepCount

    def get_IDs(self, current_partners):
        x, y = self.pos
        neighbouring_agents = current_partners
        neighbouring_cells = []

        # neighbouring_cells = [(x, y + 1), (x + 1, y), (x, y - 1), (x - 1, y)]  # N, E, S, W
        for partner in neighbouring_agents:
            neighbouring_cells.append(self.model.agent_positions[partner])

        # print("model agent pos", self.model.agent_positions)
        # print("model agent connects", self.model.updated_graphD)
        # print("my parts", self.current_partner_list)
        # print("neigh agents", neighbouring_agents)
        # print("neigh cells", neighbouring_cells)

        # First, get the neighbours
        for i in neighbouring_cells:
            bound_checker = self.model.grid.out_of_bounds(i)
            if not bound_checker:
                this_cell = self.model.grid.get_cell_list_contents([i])
                # print("This cell", this_cell)

                if len(this_cell) > 0:
                    partner = [obj for obj in this_cell
                               if isinstance(obj, PDAgent)][0]

                    partner_ID = partner.ID

                    if partner_ID not in self.partner_IDs:
                        self.partner_IDs.append(partner_ID)

                    # self.ppD_partner[partner_ID] = 0.5

    def set_defaults(self, ids):
        if self.model.dynamic:
            # I believe we just want a list of a starting value * number_of_agents
            my_pickle = [0 for i in range(self.model.number_of_agents)]
            for i in ids:
                index = ids.index(i)
                self.ppD_partner[i] = my_pickle[index]
                self.default_ppds[i] = my_pickle[index]
                self.itermove_result[i] = self.model.startingBehav
                self.pp_aprime[i] = self.model.startingBehav
                self.moody_pp_aprime[i] = self.model.startingBehav
        else:
            # open the ppD pickle
            # with open("agent_ppds.p", "rb") as f:
            #     agent_ppds = pickle.load(f)
            agent_ppds = copy.deepcopy(self.model.agent_ppds)
            # print("agent ppds are,", agent_ppds)
            my_pickle = agent_ppds[self.ID]
            # print("my defaults are", my_pickle)
            # j = 0
            for i in ids:
                # print("eeyore", i)
                index = ids.index(i)
                self.ppD_partner[i] = my_pickle[index]
                # print("this ppd was", self.ppD_partner[i])
                # print("this partner's pickled ppd is ", my_pickle[index])
                self.default_ppds[i] = my_pickle[index]
                self.itermove_result[i] = self.model.startingBehav
                self.pp_aprime[i] = self.model.startingBehav
                self.moody_pp_aprime[i] = self.model.startingBehav


    def export_training_data(self):
        # print("the ppds are", self.default_ppds)
        my_data = []
        for i in self.partner_IDs:
            temp_data = []
            temp_data.append(self.per_partner_utility[i])
            temp_data.append(self.per_partner_mcoops[i])
            temp_data.append(self.per_partner_tcoops[i])
            temp_data.append(self.per_partner_mutc[i])
            temp_data.append(self.default_ppds[i])

            if self.per_partner_strategies[i] == 'VPP':
                temp_data.append(1)
            elif self.per_partner_strategies[i] == 'ANGEL':
                temp_data.append(2)
            elif self.per_partner_strategies[i] == 'DEVIL':
                temp_data.append(3)
            elif self.per_partner_strategies[i] == 'TFT':
                temp_data.append(4)
            elif self.per_partner_strategies[i] == 'WSLS':
                temp_data.append(5)
            elif self.per_partner_strategies[i] == 'iWSLS':
                temp_data.append(6)

            # print("It's the last turn, and this train_data is", temp_data)
            my_data.append(temp_data)

        # print("my_data", my_data)

        return my_data

    def pick_strategy(self):
        """ This is an initial strategy selector for agents """

        if self.model.kNN_spawn:
            if not self.model.kNN_testing:
                # print("My id is", self.ID)
                strat = "RANDOM"
                strat = self.model.kNN_strategies[self.ID]
                self.model.agent_strategies[self.ID] = str(strat)
                return str(strat)
            elif self.model.kNN_testing:
                # print("My id is", self.ID)
                strat = "RANDOM"
                strat = self.model.kNN_strategies[self.ID]
                self.model.agent_strategies[self.ID] = str(strat)
                return str(strat)

        elif self.model.sarsa_spawn:
            choices = ["LEARN", self.model.sarsa_oppo]
            if self.model.sarsa_distro > 0:
                weights = [self.model.sarsa_distro, 1-self.model.sarsa_distro]
                strat = np.random.choice(choices, 1, replace=False, p=weights)
                self.model.agent_strategies[self.ID] = str(strat[0])
                return str(strat[0])
            else:
                if len(choices) == 2:
                    if self.model.width == 8:
                        check_a = [1, 3, 5, 7, 10, 12, 14, 16, 17, 19, 21, 23, 26, 28, 30, 32, 33, 35, 37, 39,
                               42, 44, 46, 48, 49, 51, 53, 55, 58, 60, 62, 64]
                        check_b = [2, 4, 6, 8, 9, 11, 13, 15, 18, 20, 22, 24, 25, 27, 29, 31, 34, 36, 38, 40, 41,
                               43, 45, 47, 50, 52, 54, 56, 57, 59, 61, 63]
                    elif self.model.width == 3:
                        check_a = [1, 3, 5, 7, 9]
                        check_b = [2, 4, 6, 8]
                    elif self.model.width == 4:
                        check_a = [1, 3, 6, 8, 9, 11, 14, 16]
                        check_b = [2, 4, 5, 7, 10, 12, 13, 15]
                    elif self.model.width == 5:
                        check_a = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]
                        check_b = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
                    elif self.model.width == 6:
                        check_a = [1, 3, 5, 8, 10, 12, 13, 15, 17, 20, 22, 24, 25, 27, 29, 32, 34, 36]
                        check_b = [2, 4, 6, 7, 9, 11, 14, 16, 18, 19, 21, 23, 26, 28, 30, 31, 33, 35]
                    elif self.model.width == 7:
                        check_a = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35,
                                   37, 39, 41, 43, 45, 47, 49]
                        check_b = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36,
                                   38, 40, 42, 44, 46, 48]
                    elif self.model.width == 2:
                        check_a = [1, 4]
                        check_b = [2, 3]

                    # print("My ID is:", self.ID, "and my coordinates are", self)
                    if self.ID in check_a:
                        strat = choices[0]
                        self.model.agent_strategies[self.ID] = str(strat)
                        return str(strat)
                    elif self.ID in check_b:
                        strat = choices[1]
                        if strat == 'MIXED':
                            # Then we should randomly pick from this list without weighting
                            strat = random.choice(self.model.oppoList)
                            self.model.agent_strategies[self.ID] = str(strat)
                            return str(strat)
                        # if type(strat) == list:
                        #     # Then we should randomly pick from this list without weighting
                        #     strat = random.choice(strat)
                        #     return str(strat)
                        self.model.agent_strategies[self.ID] = str(strat)
                        return str(strat)

        elif self.model.moody_sarsa_spawn:
            initChoices = ["MOODYLEARN", self.model.moody_sarsa_oppo]
            choices = []
            choices.append("MOODYLEARN")
            if len(self.model.moody_sarsa_oppo) > 0:
                for i in self.model.moody_sarsa_oppo:
                    if i == "MIXED":
                        strat = random.choice(self.model.oppoList)
                        # self.model.agent_strategies[self.ID] = str(strat)
                        choices.append(strat)
                    else:
                        choices.append(i)
            else:
                choices.append(self.model.moody_sarsa_oppo)
            if self.model.dynamic:
                weighter = 1 / len(choices)
                weights = []
                for n in choices:
                    weights.append(weighter)
                strat = np.random.choice(choices, 1, replace=False, p=weights)
                self.model.agent_strategies[self.ID] = str(strat[0])
                return str(strat[0])
            else:
                if self.model.sarsa_distro > 0:                                                               # THIS SECTION ISN'T SET TO MOODY_ --> might need changing in future
                    weights = [self.model.sarsa_distro, 1-self.model.sarsa_distro]
                    strat = np.random.choice(choices, 1, replace=False, p=weights)
                    self.model.agent_strategies[self.ID] = str(strat[0])
                    return str(strat[0])
                else:
                    if len(choices) == 2:
                        if self.model.width == 8:                                                             # THESE COORDINATES SHOULD BE CORRECT, JUST SPAWN NORMAL SARSA IN AS AN OPPONENT TYPE
                            check_a = [1, 3, 5, 7, 10, 12, 14, 16, 17, 19, 21, 23, 26, 28, 30, 32, 33, 35, 37, 39,
                                   42, 44, 46, 48, 49, 51, 53, 55, 58, 60, 62, 64]
                            check_b = [2, 4, 6, 8, 9, 11, 13, 15, 18, 20, 22, 24, 25, 27, 29, 31, 34, 36, 38, 40, 41,
                                   43, 45, 47, 50, 52, 54, 56, 57, 59, 61, 63]
                        elif self.model.width == 3:
                            check_a = [1, 3, 5, 7, 9]
                            check_b = [2, 4, 6, 8]
                        elif self.model.width == 4:
                            check_a = [1, 3, 6, 8, 9, 11, 14, 16]
                            check_b = [2, 4, 5, 7, 10, 12, 13, 15]
                        elif self.model.width == 5:
                            check_a = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]
                            check_b = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
                        # elif self.model.width == 5:
                        #     check_a = [1, 5, 21, 25, 9, 7, 17, 19, 6, 16, 20, 10, ]  # These are test-related coordinates used to spawn agents for the Sarsa Vs. Moody Paper Experiments
                        #     check_b = [13, 8, 12, 14, 18, 22, 24, 2, 4, 15, 23, 3, 11,]
                        elif self.model.width == 6:
                            check_a = [1, 3, 5, 8, 10, 12, 13, 15, 17, 20, 22, 24, 25, 27, 29, 32, 34, 36]
                            check_b = [2, 4, 6, 7, 9, 11, 14, 16, 18, 19, 21, 23, 26, 28, 30, 31, 33, 35]
                        elif self.model.width == 7:
                            check_a = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35,
                                       37, 39, 41, 43, 45, 47, 49]
                            check_b = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36,
                                       38, 40, 42, 44, 46, 48]
                        elif self.model.width == 2:
                            check_a = [1, 4]
                            check_b = [2, 3]

                        # print("My ID is:", self.ID, "and my coordinates are", self)
                        if self.ID in check_a:
                            strat = choices[0]
                            # print("I am in check_a and my strategy is now", strat)
                            self.model.agent_strategies[self.ID] = str(strat)
                            return str(strat)
                        elif self.ID in check_b:
                            strat = choices[1]
                            if strat == ['MIXED']:
                                # Then we should randomly pick from this list without weighting
                                strat = random.choice(self.model.oppoList)
                                self.model.agent_strategies[self.ID] = str(strat)
                                return str(strat)
                            # if type(strat) == list:
                            #     # Then we should randomly pick from this list without weighting
                            #     strat = random.choice(strat)
                            #     return str(strat)
                            self.model.agent_strategies[self.ID] = str(strat)
                            return str(strat)

        else:
            if self.pickstrat == "RANDOM":
                choices = ["EV", "ANGEL", "RANDOM", "DEVIL", "VEV", "TFT", "WSLS", "VPP", "iWSLS"]
                strat = random.choice(choices)
                # print("strat is", strat)
                self.model.agent_strategies[self.ID] = str(strat)
                return str(strat)
            elif self.pickstrat == "DISTRIBUTION":
                """ This is for having x agents start on y strategy and the remaining p agents
                    start on q strategy """

            elif self.pickstrat == "RDISTRO":  # Random Distribution of the selected strategies
                choices = ["iWSLS", "VPP"]
                if not self.checkerboard:
                    if not self.lineplace:
                        strat = random.choice(choices)
                        self.model.agent_strategies[self.ID] = str(strat)
                        return str(strat)
                    elif self.lineplace:
                        if len(choices) == 2:
                            if (self.ID % 2) == 0:
                                strat = choices[0]
                                self.model.agent_strategies[self.ID] = str(strat)
                                return str(strat)
                            else:
                                strat = choices[1]
                                self.model.agent_strategies[self.ID] = str(strat)
                                return str(strat)
                        elif len(choices) == 3:
                            # make choices into a popped queue, take the front most and then add it in at the back after
                            # choosing
                            return
                elif self.checkerboard:
                    # print("My ID is...", self.ID)
                    if len(choices) == 2:
                        check_a = [1, 3, 5, 7, 10, 12, 14, 16, 17, 19, 21, 23, 26, 28, 30, 32, 33, 35, 37, 39,
                                   42, 44, 46, 48, 49, 51, 53, 55, 58, 60, 62, 64]
                        check_b = [2, 4, 6, 8, 9, 11, 13, 15, 18, 20, 22, 24, 25, 27, 29, 31, 34, 36, 38, 40, 41,
                                   43, 45, 47, 50, 52, 54, 56, 57, 59, 61, 63]
                        if self.ID in check_a:
                            strat = choices[0]
                            self.model.agent_strategies[self.ID] = str(strat)
                            return str(strat)
                        elif self.ID in check_b:
                            strat = choices[1]
                            self.model.agent_strategies[self.ID] = str(strat)
                            return str(strat)

    def change_strategy(self):
        return

    def compare_score(self):
        """ Compares own score to current highest agent score in network for visualisation purposes"""
        if self.stepCount > 1:
            highscore = self.model.highest_score
            myscore = self.score
            # what percentage is my score of the highest score?
            self.proportional_score = ((myscore / highscore) * 100)

    def iter_pick_move(self, strategy, payoffs, current_partners, new_partners):
        """ Iterative move selection uses the pick_move function PER PARTNER, then stores this in a dictionary
        keyed by the partner it picked that move for. We can then cycle through these for iter. score incrementing"""
        versus_moves = {}
        x, y = self.pos

        # Current Partners will be a vector of ID numbers
        # Find, from the model storage, each partner's XY coordinates, and then access the agent in that cell
        neighbouring_agents = current_partners
        neighbouring_cells = []

        # neighbouring_cells = [(x, y + 1), (x + 1, y), (x, y - 1), (x - 1, y)]  # N, E, S, W
        for partner in neighbouring_agents:
            neighbouring_cells.append(self.model.agent_positions[partner])

        # First, get the neighbours
        for i in neighbouring_cells:
            bound_checker = self.model.grid.out_of_bounds(i)
            if not bound_checker:
                this_cell = self.model.grid.get_cell_list_contents([i])
                # print("This cell", this_cell)

                if len(this_cell) > 0:
                    partner = [obj for obj in this_cell
                               if isinstance(obj, PDAgent)][0]

                    partner_ID = partner.ID
                    partner_mood = sarsa_moody.getMoodType(partner.mood)
                    # print("partner itermove was", partner.itermove_result)
                    # print("but my ID is", self.ID)
                    if partner.itermove_result.get(self.ID) is None:
                        partner_move = 0
                    else:
                        partner_move = partner.itermove_result[self.ID]

                    if self.partner_states.get(partner_ID) is None:
                        self.partner_states[partner_ID] = sarsa_moody.observe_state(partner_move, partner_ID,
                                                                                    partner_mood,
                                                                                    self.statemode)
                    else:
                        self.partner_states[partner_ID] = sarsa_moody.observe_state(partner_move, partner_ID,
                                                                                    partner_mood,
                                                                                    self.statemode)

                    if partner_ID not in new_partners:
                        # pick a move
                        if strategy == "MOODYLEARN":
                            move = self.pick_move(strategy, payoffs, partner_ID, self.partner_states)
                            # move = self.pick_move(strategy, payoffs, partner_ID, self.working_memory)
                        else:
                            move = self.pick_move(strategy, payoffs, partner_ID, self.working_memory)
                            # move = self.pick_move(strategy, payoffs, partner_ID, self.partner_states)
                        # add that move, with partner ID, to the versus choice dictionary
                        versus_moves[partner_ID] = move
                    else:
                        versus_moves[partner_ID] = self.model.startingBehav
        # print("agent", self.ID,"versus moves:", versus_moves)
        return versus_moves

    # def iter_pick_move(self, strategy, payoffs):
    #     """ Iterative move selection uses the pick_move function PER PARTNER, then stores this in a dictionary
    #     keyed by the partner it picked that move for. We can then cycle through these for iter. score incrementing"""
    #     versus_moves = {}
    #     x, y = self.pos
    #     neighbouring_cells = [(x, y + 1), (x + 1, y), (x, y - 1), (x - 1, y)]  # N, E, S, W
    #
    #     # First, get the neighbours
    #     for i in neighbouring_cells:
    #         bound_checker = self.model.grid.out_of_bounds(i)
    #         if not bound_checker:
    #             this_cell = self.model.grid.get_cell_list_contents([i])
    #             # print("This cell", this_cell)
    #
    #             if len(this_cell) > 0:
    #                 partner = [obj for obj in this_cell
    #                            if isinstance(obj, PDAgent)][0]
    #
    #                 partner_ID = partner.ID
    #                 partner_mood = sarsa_moody.getMoodType(partner.mood)
    #                 partner_move = 0
    #                 if partner.itermove_result.get(self.ID) is None:
    #                     partner_move = 0
    #                 else:
    #                     partner_move = partner.itermove_result[self.ID]
    #
    #                 if self.partner_states.get(partner_ID) is None:
    #                     self.partner_states[partner_ID] = sarsa_moody.observe_state(partner_move, partner_ID, partner_mood,
    #                                                                                                    self.statemode)
    #
    #                 # pick a move
    #                 if strategy is not "MOODYLEARN":
    #                     move = self.pick_move(strategy, payoffs, partner_ID, self.working_memory)
    #                 else:
    #                     move = self.pick_move(strategy, payoffs, partner_ID, self.partner_states)
    #                 # add that move, with partner ID, to the versus choice dictionary
    #                 versus_moves[partner_ID] = move
    #     # print("agent", self.ID,"versus moves:", versus_moves)
    #     return versus_moves

    def iter_pick_nextmove(self, strategy, payoffs, nextstates, current_partners):
        """ Iterative move selection uses the pick_move function PER PARTNER, then stores this in a dictionary
        keyed by the partner it picked that move for. We can then cycle through these for iter. score incrementing"""
        versus_moves = {}
        x, y = self.pos

        neighbouring_agents = current_partners
        neighbouring_cells = []

        # neighbouring_cells = [(x, y + 1), (x + 1, y), (x, y - 1), (x - 1, y)]  # N, E, S, W
        for partner in neighbouring_agents:
            neighbouring_cells.append(self.model.agent_positions[partner])

        # First, get the neighbours
        for i in neighbouring_cells:
            bound_checker = self.model.grid.out_of_bounds(i)
            if not bound_checker:
                this_cell = self.model.grid.get_cell_list_contents([i])
                # print("This cell", this_cell)

                if len(this_cell) > 0:
                    partner = [obj for obj in this_cell
                               if isinstance(obj, PDAgent)][0]

                    partner_ID = partner.ID

                    # pick a move
                    move = self.pick_move(strategy, payoffs, partner_ID, nextstates)
                    # add that move, with partner ID, to the versus choice dictionary
                    versus_moves[partner_ID] = move
        # print("agent", self.ID,"versus moves:", versus_moves)
        return versus_moves

    def pick_move(self, strategy, payoffs, id, learning_state):
        """ given the payoff matrix, the strategy, and any other inputs (communication, trust perception etc.)
            calculate the expected utility of each move, and then pick the highest"""
        """AT THE MOMENT, THIS IS A GENERAL ONCE-A-ROUND FUNCTION, AND ISN'T PER PARTNER - THIS NEEDS TO CHANGE """

        if strategy is None or [] or 0:
            self.pick_strategy()

        elif strategy == "ANGEL":
            # print("I'm an angel, so I'll cooperate")
            self.number_of_c += 1
            return "C"

        elif strategy == "DEVIL":
            # print("I'm a devil, so I'll defect")
            self.number_of_d += 1
            return "D"

        elif strategy == "EV":  # this is under assumption of heterogeneity of agents
            """ EV is designed as a strategy not based on 'trust' (though it can reflect that), but instead on 
            the logic; 'I know they know defect is the best strategy usually, just as they know I know that'. """
            # Current set-up: We assume partner will defect

            ppD = 0.5  # probability of partner's defection
            ppC = 1 - ppD  # probability of partner's cooperation

            evCC = (payoffs["C", "C"] * ppC)
            evCD = (payoffs["C", "D"] * ppD)
            evDC = (payoffs["D", "C"] * ppC)
            evDD = (payoffs["D", "D"] * ppD)

            exp_util = (evCC, evCD, evDC, evDD)
            highest_ev = exp_util.index(max(exp_util))
            if highest_ev == 0:
                # print("Cooperate is best")
                self.number_of_c += 1
                return "C"
            elif highest_ev == 1:
                # print("Cooperate is best")
                self.number_of_c += 1
                return "C"
            elif highest_ev == 2:
                # print("Defect is best")
                self.number_of_d += 1
                return "D"
            elif highest_ev == 3:
                # print("Defect is best")
                self.number_of_d += 1
                return "D"

        elif strategy == "RANDOM":
            choice = self.random.choice(["C", "D"])
            if choice == "C":
                self.number_of_c += 1
            elif choice == "D":
                self.number_of_d += 1
            return choice

        elif strategy == "VEV":
            ppD = self.ppD_partner[id]
            ppC = 1 - self.ppD_partner[id]

            evCC = (payoffs["C", "C"] * ppC)
            evCD = (payoffs["C", "D"] * ppD)
            evDC = (payoffs["D", "C"] * ppC)
            evDD = (payoffs["D", "D"] * ppD)

            exp_value = (evCC, evCD, evDC, evDD)
            highest_ev = exp_value.index(max(exp_value))
            if highest_ev == 0:
                self.number_of_c += 1
                return "C"
            elif highest_ev == 1:
                self.number_of_c += 1
                return "C"
            elif highest_ev == 2:
                self.number_of_d += 1
                return "D"
            elif highest_ev == 3:
                self.number_of_d += 1
                return "D"

        elif strategy == "TFT":
            if self.stepCount == 1:
                self.number_of_c += 1
                return "C"
            else:
                if self.partner_latest_move.get(id) is None:
                    self.number_of_c += 1
                    return "C"

                if self.partner_latest_move[id] == 'C':
                    self.number_of_c += 1
                elif self.partner_latest_move[id] == 'D':
                    self.number_of_d += 1
                return self.partner_latest_move[id]

        elif strategy == "WSLS":
            """ This strategy picks C in the first turn, and then changes its move only if it 'loses' 
                - e.g. if it gets pwned or if both defect. """
            # if it's turn one, cooperate
            # after this, if the outcome for this partner was a winning one (either C-D or C-C?) then play the
            # same move again, if not, play the opposite move.

            if self.stepCount == 1:
                self.number_of_c += 1
                return "C"

            if self.partner_latest_move.get(id) is None:
                self.number_of_c += 1
                return "C"

            my_move = self.itermove_result[id]

            this_partner_move = self.partner_latest_move[id]
            outcome = [my_move, this_partner_move]

            failure_outcomes = [["C", "D"], ["D", "D"]]

            self.wsls_failed = False

            if outcome == ['C', 'C']:
                self.number_of_c += 1
                self.wsls_failed = False
            elif outcome == ['D', 'C']:
                self.number_of_d += 1
                self.wsls_failed = False
            elif outcome == ['C', 'D']:
                self.number_of_c += 1
                self.wsls_failed = True
                # print("I failed! Switching")
            elif outcome == ['D', 'D']:
                self.number_of_d += 1
                self.wsls_failed = True
                # print("I failed! Switching")

            if self.wsls_failed == True:
                if my_move == "C":
                    # self.number_of_c += 1
                    # print("Outcome was", outcome, "so Failure = ", self.wsls_failed, "So I will pick D")
                    self.wsls_failed = False
                    return "D"
                if my_move == "D":
                    # self.number_of_d += 1
                    # print("Outcome was", outcome, "so Failure = ", self.wsls_failed, "So I will pick C")
                    self.wsls_failed = False
                    return "C"
            else:
                # print("Outcome was", outcome, "so Failure = ", self.wsls_failed, "So I picked the same as last time")
                self.wsls_failed = False
                return my_move

        elif strategy == "iWSLS":
            """ This strategy picks C in the first turn, and then changes its move only if it 'loses'. 
                            - this is as alternative implementation of the previous WSLS strategy to 
                             check if it was performing as the lit suggests."""

            if self.stepCount == 1:
                self.number_of_c += 1
                return "C"

            if self.partner_latest_move.get(id) is None:
                self.number_of_c += 1
                return "C"

            my_move = self.itermove_result[id]
            this_partner_move = self.partner_latest_move[id]
            outcome = [my_move, this_partner_move]

            payoffs = self.payoffs
            outcome_payoff = payoffs[my_move, this_partner_move]

            aspiration_level = 1
            # print("My outcome was:", outcome)
            # print("My outcome payoff last turn was:", outcome_payoff, "whilst my aspiration level is", aspiration_level)
            if outcome_payoff <= aspiration_level:
                # print("My outcome was worse than my aspiration, sO I'll switch")
                if my_move == "C":
                    self.number_of_d += 1
                    return "D"
                if my_move == "D":
                    self.number_of_c += 1
                    return "C"
            else:
                # print("I'm doing good! I won't switch")
                if my_move == "C":
                    self.number_of_c += 1
                if my_move == "D":
                    self.number_of_d += 1
                return my_move

        elif strategy == "VPP":
            ppD = self.ppD_partner[id]
            ppC = 1 - self.ppD_partner[id]

            # Instead of using these values to calculate expected values/expected utilities, we use them to pick
            # our own move. This is a stochastic move selection weighted to respond to our partner's moves

            choices = ["C", "D"]
            weights = [ppC, ppD]

            choice = random.choices(population=choices, weights=weights, k=1)
            if choice == ["C"]:
                self.number_of_c += 1
                # print("number_of_c increased by 1, is now", self.number_of_c)
            elif choice == ["D"]:
                self.number_of_d += 1
                # print("number_of_d increased by 1, is now", self.number_of_d)
            return choice[0]

        # elif strategy == "SWITCH":
            # if self.stepCount <= 100:
            #     self.number_of_c += 1
            #     return "C"
            # elif self.stepCount > 100 < 200:
            #     self.number_of_d += 1
            #     return "D"
            # else:
            #     self.number_of_c += 1
            #     return "C"

        elif strategy == "LEARN":
            """ Use the epsilon-greedy algorithm to select a move to play. """
            if not learning_state:
                for j in self.partner_IDs:
                    blank_list = []
                    if self.model.memoryPaired:
                        # need to now vary the length of this depending on msize

                        for n in range(self.delta):
                            blank_list.append((0, 0))
                    elif not self.model.memoryPaired:
                        for n in range(self.delta):
                            blank_list.append(0)

                    learning_state[j] = blank_list
            elif len(learning_state) == 0:
                for j in self.partner_IDs:
                    blank_list = []
                    if self.model.memoryPaired:
                        # need to now vary the length of this depending on msize

                        for n in range(self.delta):
                            blank_list.append((0, 0))
                    elif not self.model.memoryPaired:
                        for n in range(self.delta):
                            blank_list.append(0)

            if self.delta > 1:
                egreedy = sarsa.egreedy_action(self.epsilon, self.qtable, tuple(learning_state[id]), self.model.memoryPaired)
            else:
                # print("my id is ", self.ID, "partner ids", self.partner_IDs)
                # print("eps:", self.epsilon)
                # print("qtable", len(self.qtable))
                # print("my id is ", self.ID, " and the current partner is id", id, "because my partners are", self.current_partner_list)
                # print("learn states", learning_state)
                # if learning_state.get(id) is None:

                # print("learn state", learning_state[id])
                egreedy = sarsa.egreedy_action(self.epsilon, self.qtable, learning_state[id], self.model.memoryPaired)
            if egreedy == "C":
                self.number_of_c += 1
            elif egreedy == "D":
                self.number_of_d += 1
            return egreedy

        elif strategy == "MOODYLEARN":
            """ Use the epsilon-greedy algorithm to select a move to play. """
            if not learning_state:
                for j in self.partner_IDs:
                    blank_list = []
                    if self.statemode == 'stateless':
                        blank_list.append((0))
                    if self.statemode == 'agentstate':
                        blank_list.append(((0, 0)))
                    if self.statemode == 'moodstate':
                        blank_list.append(((0, 0, 0)))
                    learning_state[j] = blank_list

            # elif learning_state[id]:
            #     learning_state[id] = sarsa_moody.observe_state(self.partner_latest_move[id], id, self.partner_moods[id],
            #                                                                                self.statemode)

            # print("my learning state is", learning_state)
            # print("my statemode is:", self.statemode)
            # if self.moody_delta > 1:
            #     egreedy = sarsa_moody.egreedy_action(self.moody_epsilon, self.moody_qtable, tuple(learning_state[id]), self.model.moody_memoryPaired)
            # else:
            #     egreedy = sarsa_moody.egreedy_action(self.moody_epsilon, self.moody_qtable, learning_state[id], self.model.moody_memoryPaired)

            if self.model.moody_MA is not 'v':
                moodAffectMode = 'Fixed'
            else:
                moodAffectMode = 'Mood'

            if self.stepCount == 1:
                moodyBehav = self.model.startingBehav
                self.moody_epsilon = self.moody_epsilon
            else:

                # print(self.partner_moods, type(self.partner_moods))
                # print(self.partner_latest_move, type(self.partner_latest_move))
                # print('latest move:', self.partner_latest_move[id], type(self.partner_latest_move[id]))
                # print('id:', id, type(id))
                # print('partner_mood', self.partner_moods[id], type(self.partner_moods[id]))
                # print('statemode', self.statemode, type(self.statemode))

                if self.moody_delta > 1:
                    # print("Mood,", self.mood, type(self.mood))
                    # print(' State:', sarsa_moody.observe_state(self.partner_latest_move[id],
                    #                                         id, self.partner_moods[id],
                    #                                         self.statemode))
                    # print(' Len Qtable:', len(self.moody_qtable))
                    # print(self.moody_qtable, type(self.moody_qtable))
                    # print(' MAM:', moodAffectMode, type(moodAffectMode))
                    # print(' e:', self.moody_epsilon, type(self.moody_epsilon))
                    # print(' MA:', self.model.moody_MA, type(self.model.moody_MA))
                    # print(' Starting Behav', self.model.startingBehav, type(self.model.startingBehav))
                    # print(' Stepcount', self.stepCount, type(self.stepCount))
                    #
                    # print('Learning state ', learning_state[id], type(learning_state[id]))

                    # moodyBehav, self.moody_epsilon = sarsa_moody.moody_action_alt(self.mood, learning_state[id],
                    #                                       self.moody_qtable, moodAffectMode, self.moody_epsilon, self.model.moody_MA,
                    #                                                               self.stepCount, self.model.startingBehav)

                    moodyBehav = sarsa_moody.moody_action_three(self.mood, learning_state[id],
                                                                                  self.moody_qtable, moodAffectMode,
                                                                                  self.moody_epsilon,
                                                                                  self.model.moody_MA,
                                                                                  )

                    # print('mbehav A', moodyBehav)
                else:
                    # this doesn't need to be here, not sure why the original version it was copied from was like this
                    # but I will leave it for now because it works
                    # moodyBehav, self.moody_epsilon = sarsa_moody.moody_action_alt(self.mood, learning_state[id],
                    #                                       self.moody_qtable, moodAffectMode, self.moody_epsilon, self.model.moody_MA,
                    #                                                               self.stepCount, self.model.startingBehav)

                    moodyBehav = sarsa_moody.moody_action_three(self.mood, learning_state[id],
                                                                                  self.moody_qtable, moodAffectMode,
                                                                                  self.moody_epsilon,
                                                                                  self.model.moody_MA,)
                    # print('mbehav B', moodyBehav)

            self.moody_epsilon = self.model.moody_epsilon

            #TODO: below, this should be conditional on NOT using the batchrunner (possibly)
            if moodyBehav == "C":
                if self.stepCount == 1:
                    self.number_of_c += 0.5
                else:
                    self.number_of_c += 1
            elif moodyBehav == "D":
                if self.stepCount == 1:
                    self.number_of_d += 0.5
                else:
                    self.number_of_d += 1

            # print('mbehav C', moodyBehav)
            return moodyBehav

    def change_update_value(self, partner_behaviour):
        """ Produce a [new update value] VALUE BY WHICH TO ALTER THE CURRENT UV given the current uv and the
        behaviour that partner has shown.
        Partner behaviour should be a list of self.delta size, ordered by eldest behaviour observed to most recent.
        current_uv should be a singular value """
        # let's start with a very simple lookup table version of behaviour comparison - probably only usable if
        # delta is fairly small, as we have to outline the specific behavioural patterns we are comparing
        """ NEW NOTE: Shaheen wants to use unordered lists, so the value judgements are just made on quantity of 
            recent good or bad interactions. This reduces the options down to 'do we have a hat trick' or 
            'is behaviour more one way or another'
            NEW NOTE mk. II: I have decided to disregard the above. Unordered lists that are only three long
            don't allow for any variability, so I'm just going to hard code it for now."""
        # THESE CONDITIONS BELOW ARE ONLY USABLE FOR A DELTA OF 3 EXACTLY

        # ** PROG NOTE: behaviour strings are capital letters
        # ** PROG NOTE: we never want the update value to be zero...
        # new_uv = current_uv
        # default
        # uv_modifier = 0
        theta = self.theta

        numberC = partner_behaviour.count('C')
        numberD = partner_behaviour.count('D')

        # # print("My partner did:", partner_behaviour)
        # if partner_behaviour == ['C', 'D', 'C']:  # Higher Value to Break Potential Cycles
        #     # print("I used behavioural rule 1, and I'm gonna return update value", theta * 3)
        #     return theta * 6
        #
        # elif partner_behaviour == ['D', 'C', 'D']:  # Higher Value to Break Potential Cycles
        #     # print("I used behavioural rule 1, and I'm gonna return update value", theta * 3)
        #     return theta * 6
        #
        # elif partner_behaviour == ['C', 'C', 'D']:  # Low Confidence due to New Behaviour
        #     # print("I used behavioural rule 2, and I'm gonna return update value", theta)
        #     return theta
        #
        # elif partner_behaviour == ['D', 'D', 'C']:  # Low Confidence due to New Behaviour
        #     # print("I used behavioural rule 2, and I'm gonna return update value", theta)
        #     return theta
        #
        # elif partner_behaviour == ['C', 'D', 'D']:  # Gaining Confidence/Trust
        #     # print("I used behavioural rule 3, and I'm gonna return update value", theta * 2)
        #     return theta * 4
        #
        # elif partner_behaviour == ['D', 'C', 'C']:  # Gaining Confidence/Trust
        #     # print("I used behavioural rule 3, and I'm gonna return update value", theta * 2)
        #     return theta * 4
        #
        # elif numberC or numberD == self.delta:  # High Value due to High Confidence
        #     # print("I used behavioural rule 4, and I'm gonna return update value", theta * 3)
        #     return theta * 6

        """ Acquire our state, then compare it to the list of all possible states generated by the 
            model. """

        all_states = copy.deepcopy(self.model.memory_states)
        state_values = copy.deepcopy(self.model.state_values)

        index = 0

        for i in all_states:
            if i == partner_behaviour:
                index = all_states.index(i)

        state_value = state_values[index]

        # TODO: This section is probably going to break to all hell when the new statemaker is used
        # I don't think it did?

        """" Now need to decide what the boundaries are for changing update value
            based on this state value that is returned... """
        # State values exist between values of 21 and -21, with a normal distribution of state values (i.e.
        # there are lower numbers of SUPER GOOD and SUPER BAD states, and where the numbers of C and D equal
        # out a bit there are more of those states). The value of middling states is zero and there is never more than
        # 16 of those states in those categories

        bound_a = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
        bound_b = [-6, -7, -8, -9, 6, 7, 8, 9]
        bound_c = [-10, -11, -12, 10, 11, 12]
        bound_d = [-13, -14, -15, 13, 14, 15]
        bound_e = [-16, -17, -18, 16, 17, 18]
        bound_f = [-19, -20, -21, 19, 20, 21]

        if state_value in bound_a:
            return theta * 1
        if state_value in bound_b:
            return theta * 2
        if state_value in bound_c:
            return theta * 3
        if state_value in bound_d:
            return theta * 4
        if state_value in bound_e:
            return theta * 5
        if state_value in bound_f:
            return theta * 6

    def check_item(self, partnerID, type):

        neighbouring_agents = [partnerID]
        neighbouring_cells = []
        # partner = 0

        # neighbouring_cells = [(x, y + 1), (x + 1, y), (x, y - 1), (x - 1, y)]  # N, E, S, W
        for partner in neighbouring_agents:
            neighbouring_cells.append(self.model.agent_positions[partner])

        # print("My ID is", self.ID, "and my current partners are", current_partners)
        # First, get the neighbours
        for i in neighbouring_cells:
            bound_checker = self.model.grid.out_of_bounds(i)
            if not bound_checker:
                this_cell = self.model.grid.get_cell_list_contents([i])
                # print("This cell", this_cell)
                if self.stepCount == 2:
                    self.n_partners += 1

                if len(this_cell) > 0:
                    partner = [obj for obj in this_cell
                               if isinstance(obj, PDAgent)][0]

        if type == "rep":
            return partner.betrayals

    def check_partner(self, current_partners):
        """ Check Partner looks at all the partner's current move selections and adds them to relevant memory spaces"""
        x, y = self.pos
        neighbouring_agents = current_partners
        neighbouring_cells = []

        # neighbouring_cells = [(x, y + 1), (x + 1, y), (x, y - 1), (x - 1, y)]  # N, E, S, W
        for partner in neighbouring_agents:
            neighbouring_cells.append(self.model.agent_positions[partner])

        # print("My ID is", self.ID, "and my current partners are", current_partners)
        # First, get the neighbours
        for i in neighbouring_cells:
            bound_checker = self.model.grid.out_of_bounds(i)
            if not bound_checker:
                this_cell = self.model.grid.get_cell_list_contents([i])
                # print("This cell", this_cell)
                if self.stepCount == 2:
                    self.n_partners += 1

                if len(this_cell) > 0:
                    partner = [obj for obj in this_cell
                               if isinstance(obj, PDAgent)][0]

                    partner_ID = partner.ID
                    partner_score = partner.score
                    partner_strategy = partner.strategy

                    # print("my id is", self.ID, "my partner is", partner_ID, "and their moves are", partner.itermove_result)
                    # print("graph says", self.model.updated_graphD)
                    # TODO: If a partner drops you, it should be mutual, and you shouldn't be looking for their move against you anymore
                    if partner.itermove_result.get(self.ID) is None:
                        partner_move = "C"  # THIS IS A HOPEFUL ESTIMATE
                    else:
                        partner_move = partner.itermove_result[self.ID]
                    partner_moves = partner.previous_moves
                    partner_mood = sarsa_moody.getMoodType(partner.mood)
                    partner_UR = partner.utilityRatio
                    partner_PR = partner.payoffRatio
                    if partner.indivAvPayoff.get(self.ID) is None:  # check if my opponent has an average payoff for me as a partner
                        partner_avAgainstMe = 0  # if they don't, use zero as the placeholder for now
                    else:
                        partner_avAgainstMe = partner.indivAvPayoff[self.ID]  # if it's available, please use it

                    if self.itermove_result.get(partner_ID) is None:
                        my_move = self.model.startingBehav
                    else:
                        my_move = self.itermove_result[partner_ID]

                    # Wanna add each neighbour's move, score etc. to the respective memory banks
                    if self.partner_latest_move.get(partner_ID) is None:
                        self.partner_latest_move[partner_ID] = partner_move
                    else:
                        self.partner_latest_move[partner_ID] = partner_move
                        # this is stupidly redundant but I don't have the current brain energy to fix it

                    if self.per_partner_utility.get(partner_ID) is None:
                        self.per_partner_utility[partner_ID] = 0

                    if self.per_partner_payoffs.get(partner_ID) is None:
                        self.per_partner_payoffs[partner_ID] = [0]

                    if self.pp_payoff.get(partner_ID) is None:
                        self.pp_payoff[partner_ID] = 0

                    if self.pp_UR.get(partner_ID) is None:
                        self.pp_UR[partner_ID] = 0

                    if self.pp_PR.get(partner_ID) is None:
                        self.pp_PR[partner_ID] = 0

                    if self.pp_CON.get(partner_ID) is None:
                        self.pp_CON[partner_ID] = 0

                    if self.per_partner_mcoops.get(partner_ID) is None:
                        self.per_partner_mcoops[partner_ID] = 0

                    if self.per_partner_tcoops.get(partner_ID) is None:
                        self.per_partner_tcoops[partner_ID] = 0

                    if self.oppAvPayoff.get(partner_ID) is None:
                        self.oppAvPayoff[partner_ID] = 0
                    else:
                        self.oppAvPayoff[partner_ID] = partner_avAgainstMe  # todo: this hopefully shouldn't break as i checked this up above

                    if self.per_partner_mutc.get(partner_ID) is None:
                        self.per_partner_mutc[partner_ID] = 0

                    if self.indivAvPayoff.get(partner_ID) is None:
                        self.indivAvPayoff[partner_ID] = 0

                    if self.partner_moods.get(partner_ID) is None:
                        self.partner_moods[partner_ID] = "NEUTRAL"

                    if self.per_partner_strategies.get(partner_ID) is None:
                        self.per_partner_strategies[partner_ID] = partner_strategy

                    if self.update_values.get(partner_ID) is None:  # add in default update value per partner
                        self.update_values[partner_ID] = self.init_uv  # this has to happen before change update value occurs!!

                    # if self.oppAvPayoff.get(partner_ID) is None:
                    #     self.indivAvPayoff[partner_ID] = 0
                    #
                    # if self.myAvPayoff.get(partner_ID) is None:
                    #     self.indivAvPayoff[partner_ID] = 0

                    """ Below is the code for adding to and/or updating self.working_memory.
                     if WM does not have a key for current partner's ID in it, we open one
                     if it does, we extract it to a local variable by popping it
                     boobly boo, mess about with it and check what it means for us here
                     after it is updated and checked, we send it back to working memory
                    """
                    current_uv = self.update_value
                    self.pp_UR[partner_ID] = partner.utilityRatio
                    self.pp_PR[partner_ID] = partner.payoffRatio
                    self.pp_CON[partner_ID] = partner.normalizedActorDegreeCentrality


                    if self.strategy == "VPP" or "LEARN":
                        if self.strategy is not "MOODYLEARN":
                            if self.model.learnFrom != "us":
                                if self.working_memory.get(partner_ID) is None:
                                    zeroes = []
                                    for j in range(self.delta-1):
                                        zeroes.append(0)
                                    zeroes.append(partner_move)
                                    self.working_memory[partner_ID] = zeroes  # initialise with first value if doesn't exist
                                else:
                                    current_state = self.working_memory.pop(partner_ID)

                                    # first, check if it has more than three values
                                    if len(current_state) < self.delta:  # if list hasn't hit delta, add in new move
                                        if self.model.learnFrom == "them":
                                            current_state.append(partner_move)
                                        elif self.model.learnFrom == "me":
                                            current_state.append(my_move)
                                    elif len(current_state) == self.delta:
                                        current_state.pop(0)
                                        if self.model.learnFrom == "them":
                                            current_state.append(partner_move)  # we have the updated move list for that partner here
                                            current_uv = self.update_values[partner_ID]

                                            self.update_value = self.update_value + self.change_update_value(current_state)
                                        elif self.model.learnFrom == "me":
                                            current_state.append(my_move)

                                    # print('My current partner history is now:', current_state)
                                    self.working_memory[partner_ID] = current_state  # re-instantiate the memory to the bank

                            elif self.model.learnFrom == "us":
                                if self.working_memory.get(partner_ID) is None:
                                    zeroes = []
                                    if self.delta > 1:
                                        for j in range(self.delta-1):
                                            zeroes.append((0,0))
                                    # print('mm', my_move, 'pm', partner_move)
                                    zeroes.append((my_move, partner_move))
                                    self.working_memory[partner_ID] = zeroes
                                else:
                                    current_state = self.working_memory.pop(partner_ID)
                                    # print('mmm', my_move, 'pmm', partner_move)
                                    # print('len cs:', len(current_state), 'del', self.delta)
                                    if len(current_state) < self.delta:
                                        current_state.append((my_move, partner_move))
                                    elif len(current_state) == self.delta:
                                        current_state.pop(0)
                                        current_state.append((my_move, partner_move))
                                    self.working_memory[partner_ID] = current_state

                                    # self.update_value = self.update_value + self.change_update_value(current_state)
                                    # TODO: Change the above so it doesn't need to work on just 7-count opponent values

                    if self.strategy == "MOODYLEARN":
                        self.partner_moods[partner_ID] = partner_mood
                        if self.model.moody_learnFrom != "us":
                            if self.working_memory.get(partner_ID) is None:
                                zeroes = []
                                for j in range(self.moody_delta-1):
                                    zeroes.append(0)
                                zeroes.append(sarsa_moody.get_payoff(my_move, partner_move, self.model.CC, self.model.DD, self.model.CD, self.model.DC))
                                self.working_memory[partner_ID] = zeroes  # initialise with first value if doesn't exist
                            else:
                                current_state = self.working_memory.pop(partner_ID)

                                # first, check if it has more than three values
                                if len(current_state) < self.moody_delta:  # if list hasn't hit delta, add in new move
                                    if self.model.moody_learnFrom == "them":
                                        current_state.append(sarsa_moody.get_payoff(my_move, partner_move, self.model.CC, self.model.DD, self.model.CD, self.model.DC))
                                    # elif self.model.moody_learnFrom == "me":
                                    #     current_state.append(my_move)
                                elif len(current_state) == self.moody_delta:
                                    current_state.pop(0)
                                    if self.model.moody_learnFrom == "them":
                                        current_state.append(sarsa_moody.get_payoff(my_move, partner_move, self.model.CC, self.model.DD, self.model.CD, self.model.DC))
                                        current_uv = self.update_values[partner_ID]

                                        self.update_value = self.update_value + self.change_update_value(current_state)
                                    # elif self.model.moody_learnFrom == "me":
                                    #     current_state.append(my_move)

                                # print('My current partner history is now:', current_state)
                                self.working_memory[partner_ID] = current_state  # re-instantiate the memory to the bank
                                self.partner_states[partner_ID] = sarsa_moody.observe_state(partner_move,
                                                                                                       partner_ID,
                                                                                                       partner_mood,
                                                                                                       self.statemode)
                        # get the current state, action
                        # check the payoff for this round, add it to the state-action list if it isn't over delta

                        state = sarsa_moody.observe_state(partner_move, partner_ID, partner_mood, self.model.moody_statemode)
                        action = my_move
                        stateMem = self.state_working_memory[tuple(state)]
                        # print('stateMem', stateMem)
                        if action == 'C':
                            stateActionMem = stateMem[0]
                        else:
                            stateActionMem = stateMem[1]

                        # At this point (after initialisation) we have the list we want to edit
                        if stateActionMem is None:
                            zeroes = []
                            for j in range(self.moody_delta - 1):
                                zeroes.append(0)
                            zeroes.append(sarsa_moody.get_payoff(my_move, partner_move, self.model.CC, self.model.DD,
                                                                 self.model.CD, self.model.DC))

                            stateActionMem = zeroes  # initialise with first value if doesn't exist
                            if action == 'C':
                                stateMem[0] = stateActionMem
                            else:
                                stateMem[1] = stateActionMem

                            self.state_working_memory[tuple(state)] = stateMem

                        else:
                            current_state = stateActionMem
                            # print('stateActionMem', stateActionMem)
                            # first, check if it has more than three values
                            if len(current_state) < self.moody_delta:  # if list hasn't hit delta, add in new move
                                if self.model.moody_learnFrom == "them":
                                    current_state.append(
                                        sarsa_moody.get_payoff(my_move, partner_move, self.model.CC, self.model.DD,
                                                               self.model.CD, self.model.DC))
                                # elif self.model.moody_learnFrom == "me":
                                #     current_state.append(my_move)
                            elif len(current_state) == self.moody_delta:
                                current_state.pop(0)
                                if self.model.moody_learnFrom == "them":
                                    current_state.append(
                                        sarsa_moody.get_payoff(my_move, partner_move, self.model.CC, self.model.DD,
                                                               self.model.CD, self.model.DC))
                                    current_uv = self.update_values[partner_ID]

                                    self.update_value = self.update_value + self.change_update_value(current_state)
                                # elif self.model.moody_learnFrom == "me":
                                #     current_state.append(my_move)

                            # print('My current partner history is now:', current_state)
                            stateActionMem = current_state
                            if action == 'C':
                                stateMem[0] = stateActionMem
                            else:
                                stateMem[1] = stateActionMem

                            self.state_working_memory[tuple(state)] = stateMem
                            # print('mem for that (s,a) is:', self.state_working_memory[tuple(state)])

                        # elif self.model.moody_learnFrom == "us":
                        #     if self.working_memory.get(partner_ID) is None:
                        #         zeroes = []
                        #         if self.moody_delta > 1:
                        #             for j in range(self.moody_delta-1):
                        #                 zeroes.append((0,0))
                        #         # print('mm', my_move, 'pm', partner_move)
                        #         zeroes.append((my_move, partner_move))
                        #         self.working_memory[partner_ID] = zeroes
                        #     else:
                        #         current_state = self.working_memory.pop(partner_ID)
                        #         # print('mmm', my_move, 'pmm', partner_move)
                        #         # print('len cs:', len(current_state), 'del', self.delta)
                        #         if len(current_state) < self.moody_delta:
                        #             current_state.append((my_move, partner_move))
                        #         elif len(current_state) == self.moody_delta:
                        #             current_state.pop(0)
                        #             current_state.append((my_move, partner_move))
                        #         self.working_memory[partner_ID] = current_state

                                # self.update_value = self.update_value + self.change_update_value(current_state)

                    """" ======================================================================================== """

                    # First, check if we have a case file on them in each memory slot
                    if self.partner_moves.get(partner_ID) is None:  # if we don't have one for this partner, make one
                        self.partner_moves[partner_ID] = []
                        self.partner_moves[partner_ID].append(partner_move)
                    else:
                        self.partner_moves[partner_ID].append(partner_move)
                        """ We should repeat the above process for the other memory fields too, like 
                        partner's gathered utility """

                    if partner_ID not in self.partner_IDs:
                        self.partner_IDs.append(partner_ID)

    def increment_score(self, payoffs):
        total_utility = 0
        outcome_listicle = {}
        for i in self.partner_IDs:
            if self.itermove_result.get(i) is None:
                my_move = self.model.startingBehav
            else:
                my_move = self.itermove_result[i]


            this_partner_move = self.partner_latest_move[i]
            outcome = [my_move, this_partner_move]
            if my_move == 'D':
                if this_partner_move == 'C':
                    #self.model.reputationBlackboard[self.ID] += 1  # Here we increment our model rep each time we betray
                    self.betrayals += 1

            if my_move == 'C':
                self.per_partner_mcoops[i] += 1
                # print("I cooperated with my partner so my total C with them is,", self.per_partner_coops[i])
                # print("My score with them is", self.per_partner_utility[i])

            if this_partner_move == 'C':
                self.per_partner_tcoops[i] += 1

            if outcome == ['C', 'C']:
                self.mutual_c_outcome += 1
                self.per_partner_mutc[i] += 1

            outcome_listicle[i] = outcome
            outcome_payoff = payoffs[my_move, this_partner_move]
            # print("Outcome with partner %i was:" % i, outcome)

            self.per_partner_payoffs[i].append(outcome_payoff)
            if self.strategy == "LEARN":
                self.pp_payoff[i] = outcome_payoff
            elif self.strategy == "MOODYLEARN":
                self.moody_pp_payoff[i] = outcome_payoff
                self.moody_pp_oppPayoff[i] = payoffs[this_partner_move, my_move]

            self.pp_oppPayoff[i] = payoffs[this_partner_move, my_move]
            self.indivAvPayoff[i] = statistics.mean(self.per_partner_payoffs[i])
            # print("My individual average payoff for partner", i, "is ", self.indivAvPayoff[i])

            # ------- Here is where we change variables based on the outcome -------
            if self.strategy == "VEV" or "RANDOM" or "VPP" or "LEARN" or "MOODYLEARN":
                if self.ppD_partner[i] < 1 and self.ppD_partner[i] > 0:

                    # if this_partner_move == "D":
                    #     self.ppD_partner[i] += 0.05
                    # elif this_partner_move == "C":
                    #     self.ppD_partner[i] -= 0.05
                    if this_partner_move == "D":
                        self.ppD_partner[i] += abs((outcome_payoff * self.update_value))  # self.update_values
                    elif this_partner_move == "C":
                        self.ppD_partner[i] -= abs((outcome_payoff * self.update_value))  # self.update_values

                if self.ppD_partner[i] > 1:
                    self.ppD_partner[i] = 1
                elif self.ppD_partner[i] < 0:
                    self.ppD_partner[i] = 0.001
                elif self.ppD_partner[i] == 6.938893903907228e-17:
                    self.ppD_partner[i] = 0.001

            outcome_payoff = payoffs[my_move, this_partner_move]
            current_partner_payoff = self.per_partner_utility[i]
            new_partner_payoff = current_partner_payoff + outcome_payoff
            self.per_partner_utility[i] = new_partner_payoff
            total_utility += outcome_payoff
            if self.printing:
                print("I am agent", self.ID, " I chose", my_move, " my partner is:", i, " they picked ",
                      this_partner_move, " so my payoff is ", outcome_payoff, " The p I will defect is now,",
                      self.ppD_partner)

        # self.score = self.score + total_utility
        self.outcome_list = outcome_listicle

        """ Here, we want to increment the GLOBAL, across-partner average payoff for the round """
        round_average = []
        for j in self.indivAvPayoff:
            item = self.indivAvPayoff[j]
            round_average.append(item)
        self.globalAvPayoff = statistics.mean(round_average)

        if self.globalAvPayoff > self.globalHighPayoff:
            self.globalHighPayoff = self.globalAvPayoff
        # print("My round average was ", self.globalAvPayoff, "and my highscore is ", self.globalHighPayoff)

        return total_utility

    def output_data_to_model(self):
        """ This sends the data to the model so the model can output it (I HOPE) """
        # print("Common move", self.common_move)
        if self.common_move == ['C']:
            self.model.agents_cooperating += 1
        elif self.common_move == ['D']:
            self.model.agents_defecting += 1
        # No line for Eq because the agent hasn't got a preference either way

        self.model.number_of_defects += self.number_of_d
        self.model.number_of_coops += self.number_of_c

        self.model.agent_list.append('{}, {}'.format(self.ID, self.strategy))

        # and also time each agent's step to create a total time thingybob

    def fake_data_to_file(self, outcomes):
        """ Outputs Zero Data on a Partner Switch Round"""

        # for m in self.per_partner_strategies:
        #     if self.per_partner_strategies[m] == self.strategy:
        #         self.similar_partners += 1
        #
        #
        # # List to add:
        # # Number of Partners
        # numbPartners = len(self.current_partner_list)
        #
        # # Who Each Partner Is
        # partnerList = self.current_partner_list
        #
        # # My Average and Median Total Utility -
        # utils = []
        # print("my partners are", self.current_partner_list, "and my per partner utilities are", self.per_partner_utility)
        # for i in self.current_partner_list:
        #     utils.append(self.per_partner_utility[i])
        # # print("My per partner utility is: ", utils)
        # if len(utils) == 0:
        #     utils = [0]
        # avUtility = sum(utils)/len(utils)
        # self.utilityRatio = sum(utils)/len(utils)
        # medianUtility = statistics.median(utils)
        #
        # # My Average and Median Round Score Per Partner
        # pays = []
        # for i in self.current_partner_list:
        #     end = len(self.per_partner_payoffs[i]) - 1
        #     item = self.per_partner_payoffs[i]
        #     pays.append(item[end])
        # if len(pays) == 0:
        #     pays = [0]
        # # print("My per partner payoff is: ", pays)
        # avPayoff = sum(pays)/len(pays)
        # self.payoffRatio = sum(pays)/len(pays)
        # medianPayoff = statistics.median(pays)
        #
        # # My Total Utility
        # totalUtility = self.score
        #
        # # ============ Centrality Measures ===============
        # # The Normalised Actor Degree Centrality is actually the connectedness measure below
        # """ The Index of Group Degree Centralisation is the normalised (I think?) version of a measure that  measures
        # the extent to which actors in the network differ from one another in their degree centralities. """
        #
        #
        # # How Connected I am (out of max partners ratio)
        # # print("my id is, ", self.ID, "and I have ", len(self.current_partner_list), "partners")
        # self.actorDegreeCentrality = len(self.current_partner_list)
        # self.normalizedActorDegreeCentrality = len(self.current_partner_list) / (self.model.number_of_agents - 1)
        # # print("my id is", self.ID, "and my centralities are", self.actorDegreeCentrality, self.normalizedActorDegreeCentrality, "my partners are", self.current_partner_list)
        # self.model.groupDegreeCentralities[self.ID] = self.actorDegreeCentrality
        # IGDC = self.model.group_degree_centralization
        #
        # """ When changing over to random network agents, there was some kind of background bug in the data
        # output where each round's number of c/number of d was correct for the round AFTER (some misalignment
        # in the csv) - this below is a temporary fix for that to save time. """
        # dCount = 0
        # cCount = 0
        # for n in self.current_partner_list:
        #     move = self.itermove_result[n]
        #     if move == 'D':
        #         dCount += 1
        #     elif move == 'C':
        #         cCount += 1
        # # Number of C
        # numbC = cCount
        #
        # # Number of D
        # numbD = dCount
        #
        # # Number of Mutual C
        # numbMutC = self.mutual_c_outcome
        #
        # # Mood
        # mood = self.mood
        #
        # # Coops / N Partners
        # if numbPartners == 0:
        #     cooperationRatio = 0
        # else:
        #     cooperationRatio = cCount / numbPartners
        #
        # # Number of Similar Partners
        # similarPartners = self.similar_partners
        #
        # # Average Scores of my Partners? Because we can't track who partners are directly
        # # Average Connectedness of my Partners? Because we can't track who partners are directly

        #
        #
        #
        # partnerURs = []
        # partnerPRs = []
        # partnerCONs = []
        #
        # if numbPartners == 0:
        #     partnerURs = [0]
        #     partnerPRs = [0]
        #     partnerCONs = [0]

        # for i in self.current_partner_list:
        #     partnerURs.append(self.pp_UR[i])
        #     partnerPRs.append(self.pp_PR[i])
        #     partnerCONs.append(self.pp_CON[i])
        #
        # averagePartnerUtilityRatio = statistics.mean(partnerURs)
        # averagePartnerPayoffRatio = statistics.mean(partnerPRs)
        # averagePartnerConnectedness = statistics.mean(partnerCONs)
        # self.average_ppUR = averagePartnerUtilityRatio
        # self.average_ppPR = averagePartnerPayoffRatio
        # self.average_ppCON = averagePartnerConnectedness
        #
        # # Model Connectedness
        # graphConnectedness = self.model.graph_connectedness
        #
        # prob_list = []
        # util_list = []
        # move_list = []
        # average_list = []
        #
        # for i in self.indivAvPayoff:
        #     average_list.append(self.indivAvPayoff[i])
        #
        # for i in self.ppD_partner:
        #     prob_list.append(self.ppD_partner[i])
        #
        # for i in self.per_partner_utility:
        #     util_list.append(self.per_partner_utility[i])
        #
        # for i in self.itermove_result:  # This encoding is per move type, allows graphing trends in move selection
        #     if self.itermove_result[i] == 'C':
        #         move_list.append(1)
        #     elif self.itermove_result[i] == 'D':
        #         move_list.append(2)

        strategy_code = 'None'

        if self.strategy == 'RANDOM':
            strategy_code = 0
        elif self.strategy == 'ANGEL':
            strategy_code = 1
        elif self.strategy == 'DEVIL':
            strategy_code = 2
        elif self.strategy == 'EV':
            strategy_code = 3
        elif self.strategy == 'VEV':
            strategy_code = 4
        elif self.strategy == 'TFT':
            strategy_code = 5
        elif self.strategy == 'VPP':
            strategy_code = 6
        elif self.strategy == 'WSLS':
            strategy_code = 7
        elif self.strategy == "LEARN":
            strategy_code = 8
        elif self.strategy == "MOODYLEARN":
            strategy_code = 9

        """ The above will error catch for when agents don't have those values, and will still let us print 
            to csv. **** WOULD ALSO LIKE TO DO THIS FOR MOVE PER PARTNER """


        # Here are the false values we should use for outputting this round
        # stepCount we can keep the same?
        # strategy we keep the same
        # strategy code we keep the same
        # itermove result can stay the same, as it's just the dict
        totalUtility = self.score
        # common move can stay the same, unless it breaks?
        numbC = 0
        numbD = 0
        numbMutC = 0
        # outcomes we can keep the same
        numbPartners = 0
        partnerList = {'SWAP'}
        avUtility = 0
        medianUtility = 0
        avPayoff = 0
        medianPayoff = 0
        mood = self.mood
        cooperationRatio = 0
        similarPartners = 0
        averagePartnerUtilityRatio = 0
        averagePartnerPayoffRatio = 0
        averagePartnerConnectedness = 0
        graphConnectedness = self.model.graph_connectedness
        IGDC = 0


        if self.strategy == "MOODYLEARN":
            try:
                self.attempts_taken += 1
                with open('{}.csv'.format(self.filename), 'a', newline='') as csvfile:
                    fieldnames = ['stepcount_%d' % self.ID,
                                  'strategy_%d' % self.ID,
                                  'strat_code_%d' % self.ID,
                                  'move_%d' % self.ID,
                                  'utility_%d' % self.ID,
                                  'common_move_%d' % self.ID,
                                  'number_coop_%d' % self.ID,
                                  'number_defect_%d' % self.ID,
                                  'mutualC_%d' % self.ID,
                                  'outcomes_%d' % self.ID,
                                  'n_partners_%d' % self.ID,
                                  'partner_list_%d' % self.ID,
                                  'av_utility_%d' % self.ID,
                                  'median_utility_%d' % self.ID,
                                  'av_payoff_%d' % self.ID,
                                  'median_payoff_%d' % self.ID,
                                  'mood_%d' % self.ID,
                                  'coop_ratio_%d' % self.ID,
                                  'similarPartners_%d' % self.ID,
                                  'avPartnerUR_%d' % self.ID,
                                  'avPartnerPR_%d' % self.ID,
                                  'avPartnerConnected_%d' % self.ID,
                                  'graphConnectedness_%d' % self.ID,
                                  'degree_centrality_%d' % self.ID,
                                  'normalized_centrality_%d' % self.ID,
                                  'groupDegreeCent_%d' % self.ID,
                                  'globav_%d' % self.ID,
                                  'sensitivity_%d' % self.ID,
                                  'epsilon_%d' % self.ID,
                                  #'rejected_%d' % self.ID
                                  ]

                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                    # moves = []
                    # for i in self.move:
                    #     moves.append(self.move[i])

                    if self.stepCount == 1:
                        writer.writeheader()

                    writer.writerow(
                        {'stepcount_%d' % self.ID: self.stepCount,
                                  'strategy_%d' % self.ID: self.strategy,
                                  'strat_code_%d' % self.ID: strategy_code,
                                  'move_%d' % self.ID: self.itermove_result,
                                  'utility_%d' % self.ID: totalUtility,
                                  'common_move_%d' % self.ID: self.common_move,
                                  'number_coop_%d' % self.ID: numbC,
                                  'number_defect_%d' % self.ID: numbD,
                                  'mutualC_%d' % self.ID: numbMutC,
                                  'outcomes_%d' % self.ID: outcomes,
                                  'n_partners_%d' % self.ID: numbPartners,
                                  'partner_list_%d' % self.ID: partnerList,
                                  'av_utility_%d' % self.ID: avUtility,
                                  'median_utility_%d' % self.ID: medianUtility,
                                  'av_payoff_%d' % self.ID: avPayoff,
                                  'median_payoff_%d' % self.ID: medianPayoff,
                                  'mood_%d' % self.ID: mood,
                                  'coop_ratio_%d' % self.ID: cooperationRatio,
                                  'similarPartners_%d' % self.ID: similarPartners,
                                  'avPartnerUR_%d' % self.ID: averagePartnerUtilityRatio,
                                  'avPartnerPR_%d' % self.ID: averagePartnerPayoffRatio,
                                  'avPartnerConnected_%d' % self.ID: averagePartnerConnectedness,
                                  'graphConnectedness_%d' % self.ID: graphConnectedness,
                                  'degree_centrality_%d' % self.ID: self.actorDegreeCentrality,
                                  'normalized_centrality_%d' % self.ID: self.normalizedActorDegreeCentrality,
                                  'groupDegreeCent_%d' % self.ID: IGDC,
                                  'globav_%d' % self.ID: self.globalAvPayoff,
                                  'sensitivity_%d' % self.ID: self.sensitivity_mod,
                                  'epsilon_%d' % self.ID: self.epsilon,
                                  #'rejected_%d' % self.ID: self.rejected_partner_list
                                   })
            except PermissionError:
                self.fake_data_to_file(self.outcome_list)

        else:
            try:
                with open('{}.csv'.format(self.filename), 'a', newline='') as csvfile:
                    fieldnames = ['stepcount_%d' % self.ID,
                                  'strategy_%d' % self.ID,
                                  'strat_code_%d' % self.ID,
                                  'move_%d' % self.ID,
                                  'utility_%d' % self.ID,
                                  'common_move_%d' % self.ID,
                                  'number_coop_%d' % self.ID,
                                  'number_defect_%d' % self.ID,
                                  'mutualC_%d' % self.ID,
                                  'outcomes_%d' % self.ID,
                                  'n_partners_%d' % self.ID,
                                  'partner_list_%d' % self.ID,
                                  'av_utility_%d' % self.ID,
                                  'median_utility_%d' % self.ID,
                                  'av_payoff_%d' % self.ID,
                                  'median_payoff_%d' % self.ID,
                                  'mood_%d' % self.ID,
                                  'coop_ratio_%d' % self.ID,
                                  'similarPartners_%d' % self.ID,
                                  'avPartnerUR_%d' % self.ID,
                                  'avPartnerPR_%d' % self.ID,
                                  'avPartnerConnected_%d' % self.ID,
                                  'graphConnectedness_%d' % self.ID,
                                  'degree_centrality_%d' % self.ID,
                                  'normalized_centrality_%d' % self.ID,
                                  'groupDegreeCent_%d' % self.ID,
                                  'globav_%d' % self.ID,
                                  'sensitivity_%d' % self.ID,
                                  'epsilon_%d' % self.ID,
                                  #'rejected_%d' % self.ID
                                  ]

                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                    # moves = []
                    # for i in self.move:
                    #     moves.append(self.move[i])

                    if self.stepCount == 1:
                        writer.writeheader()

                    writer.writerow(
                        {'stepcount_%d' % self.ID: self.stepCount,
                         'strategy_%d' % self.ID: self.strategy,
                         'strat_code_%d' % self.ID: strategy_code,
                         'move_%d' % self.ID: self.itermove_result,
                         'utility_%d' % self.ID: totalUtility,
                         'common_move_%d' % self.ID: self.common_move,
                         'number_coop_%d' % self.ID: numbC,
                         'number_defect_%d' % self.ID: numbD,
                         'mutualC_%d' % self.ID: numbMutC,
                         'outcomes_%d' % self.ID: outcomes,
                         'n_partners_%d' % self.ID: numbPartners,
                         'partner_list_%d' % self.ID: partnerList,
                         'av_utility_%d' % self.ID: avUtility,
                         'median_utility_%d' % self.ID: medianUtility,
                         'av_payoff_%d' % self.ID: avPayoff,
                         'median_payoff_%d' % self.ID: medianPayoff,
                         'mood_%d' % self.ID: mood,
                         'coop_ratio_%d' % self.ID: cooperationRatio,
                         'similarPartners_%d' % self.ID: similarPartners,
                         'avPartnerUR_%d' % self.ID: averagePartnerUtilityRatio,
                         'avPartnerPR_%d' % self.ID: averagePartnerPayoffRatio,
                         'avPartnerConnected_%d' % self.ID: averagePartnerConnectedness,
                         'graphConnectedness_%d' % self.ID: graphConnectedness,
                         'degree_centrality_%d' % self.ID: self.actorDegreeCentrality,
                         'normalized_centrality_%d' % self.ID: self.normalizedActorDegreeCentrality,
                         'groupDegreeCent_%d' % self.ID: IGDC,
                         'globav_%d' % self.ID: self.globalAvPayoff,
                         'sensitivity_%d' % self.ID: self.sensitivity_mod,
                         'epsilon_%d' % self.ID: self.epsilon,
                         #'rejected_%d' % self.ID: self.rejected_partner_list
                         })
            except PermissionError:
                self.output_data_to_file(self.outcome_list)

    def output_data_to_file(self, outcomes):
        """ Outputs the data collected each turn on multiple agent variables to a .csv file"""

        for m in self.current_partner_list:
            if self.per_partner_strategies[m] == self.strategy:
                self.similar_partners += 1


        # List to add:
        # Number of Partners
        numbPartners = len(self.current_partner_list)

        # Who Each Partner Is
        partnerList = self.current_partner_list

        # My Average and Median Total Utility - TODO: IS THIS TOTAL UTILITY OR PER ROUND?
        utils = []
        # print("my partners are", self.current_partner_list, "and my per partner utilities are", self.per_partner_utility)
        for i in self.current_partner_list:
            utils.append(self.per_partner_utility[i])
        # print("My per partner utility is: ", utils)
        if len(utils) == 0:
            utils = [0]
        avUtility = sum(utils)/len(utils)
        self.utilityRatio = sum(utils)/len(utils)
        medianUtility = statistics.median(utils)

        # My Average and Median Round Score Per Partner
        pays = []
        for i in self.current_partner_list:
            end = len(self.per_partner_payoffs[i]) - 1
            item = self.per_partner_payoffs[i]
            pays.append(item[end])
        if len(pays) == 0:
            pays = [0]
        # print("My per partner payoff is: ", pays)
        avPayoff = sum(pays)/len(pays)
        self.avPayoff = avPayoff
        self.payoffRatio = sum(pays)/len(pays)
        medianPayoff = statistics.median(pays)

        # My Total Utility
        totalUtility = self.score

        # ============ Centrality Measures ===============
        # The Normalised Actor Degree Centrality is actually the connectedness measure below
        """ The Index of Group Degree Centralisation is the normalised (I think?) version of a measure that  measures
        the extent to which actors in the network differ from one another in their degree centralities. """


        # How Connected I am (out of max partners ratio)
        # print("my id is, ", self.ID, "and I have ", len(self.current_partner_list), "partners")
        self.actorDegreeCentrality = len(self.current_partner_list)
        self.normalizedActorDegreeCentrality = len(self.current_partner_list) / (self.model.number_of_agents - 1)
        # print("my id is", self.ID, "and my centralities are", self.actorDegreeCentrality, self.normalizedActorDegreeCentrality, "my partners are", self.current_partner_list)
        self.model.groupDegreeCentralities[self.ID] = self.actorDegreeCentrality
        IGDC = self.model.group_degree_centralization

        """ When changing over to random network agents, there was some kind of background bug in the data 
        output where each round's number of c/number of d was correct for the round AFTER (some misalignment
        in the csv) - this below is a temporary fix for that tso save time. """
        dCount = 0
        cCount = 0
        for n in self.current_partner_list:
            move = self.itermove_result[n]
            if move == 'D':
                dCount += 1
            elif move == 'C':
                cCount += 1
        # Number of C
        numbC = cCount

        # Number of D
        numbD = dCount

        # Number of Mutual C
        numbMutC = self.mutual_c_outcome

        # Mood
        mood = self.mood

        # Coops / N Partners
        if numbPartners == 0:
            cooperationRatio = 0
        else:
            cooperationRatio = cCount / numbPartners

        # Number of Similar Partners
        similarPartners = self.similar_partners

        # Average Scores of my Partners? Because we can't track who partners are directly
        # Average Connectedness of my Partners? Because we can't track who partners are directly
        # TODO: PR AND PAYOFF RATIO DOESN'T SEEM TO BE WORKING



        partnerURs = []
        partnerPRs = []
        partnerCONs = []

        if numbPartners == 0:
            partnerURs = [0]
            partnerPRs = [0]
            partnerCONs = [0]
        # TODO: Get the other partner's self.utilityRatio values?
        for i in self.current_partner_list:
            partnerURs.append(self.pp_UR[i])
            partnerPRs.append(self.pp_PR[i])
            partnerCONs.append(self.pp_CON[i])

        averagePartnerUtilityRatio = statistics.mean(partnerURs)
        averagePartnerPayoffRatio = statistics.mean(partnerPRs)
        averagePartnerConnectedness = statistics.mean(partnerCONs)
        self.average_ppUR = averagePartnerUtilityRatio
        self.average_ppPR = averagePartnerPayoffRatio
        self.average_ppCON = averagePartnerConnectedness

        # Model Connectedness
        graphConnectedness = self.model.graph_connectedness

        prob_list = []
        util_list = []
        move_list = []
        average_list = []

        for i in self.indivAvPayoff:
            average_list.append(self.indivAvPayoff[i])

        for i in self.ppD_partner:
            prob_list.append(self.ppD_partner[i])

        for i in self.per_partner_utility:
            util_list.append(self.per_partner_utility[i])

        for i in self.itermove_result:  # This encoding is per move type, allows graphing trends in move selection
            if self.itermove_result[i] == 'C':
                move_list.append(1)
            elif self.itermove_result[i] == 'D':
                move_list.append(2)

        strategy_code = 'None'

        if self.strategy == 'RANDOM':
            strategy_code = 0
        elif self.strategy == 'ANGEL':
            strategy_code = 1
        elif self.strategy == 'DEVIL':
            strategy_code = 2
        elif self.strategy == 'EV':
            strategy_code = 3
        elif self.strategy == 'VEV':
            strategy_code = 4
        elif self.strategy == 'TFT':
            strategy_code = 5
        elif self.strategy == 'VPP':
            strategy_code = 6
        elif self.strategy == 'WSLS':
            strategy_code = 7
        elif self.strategy == "LEARN":
            strategy_code = 8
        elif self.strategy == "MOODYLEARN":
            strategy_code = 9

        """ The above will error catch for when agents don't have those values, and will still let us print 
            to csv. **** WOULD ALSO LIKE TO DO THIS FOR MOVE PER PARTNER """

        #TODO: FIX DATA OUTPUTTING PLEASE?
        if self.strategy == "MOODYLEARN":
            try:
                self.attempts_taken += 1
                with open('{}.csv'.format(self.filename), 'a', newline='') as csvfile:
                    fieldnames = ['stepcount_%d' % self.ID,
                                  'strategy_%d' % self.ID,
                                  'strat_code_%d' % self.ID,
                                  'move_%d' % self.ID,
                                  'utility_%d' % self.ID,
                                  'common_move_%d' % self.ID,
                                  'number_coop_%d' % self.ID,
                                  'number_defect_%d' % self.ID,
                                  'mutualC_%d' % self.ID,
                                  'outcomes_%d' % self.ID,
                                  'n_partners_%d' % self.ID,
                                  'partner_list_%d' % self.ID,
                                  'av_utility_%d' % self.ID,
                                  'median_utility_%d' % self.ID,
                                  'av_payoff_%d' % self.ID,
                                  'median_payoff_%d' % self.ID,
                                  'mood_%d' % self.ID,
                                  'coop_ratio_%d' % self.ID,
                                  'similarPartners_%d' % self.ID,
                                  'avPartnerUR_%d' % self.ID,
                                  'avPartnerPR_%d' % self.ID,
                                  'avPartnerConnected_%d' % self.ID,
                                  'graphConnectedness_%d' % self.ID,
                                  'degree_centrality_%d' % self.ID,
                                  'normalized_centrality_%d' % self.ID,
                                  'groupDegreeCent_%d' % self.ID,
                                  'globav_%d' % self.ID,
                                  'sensitivity_%d' % self.ID,
                                  'epsilon_%d' % self.ID,
                                  #'rejected_%d' % self.ID
                                  'blackboard_%d' % self.ID,
                                  ]

                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                    # moves = []
                    # for i in self.move:
                    #     moves.append(self.move[i])

                    if self.stepCount == 1:
                        writer.writeheader()

                    writer.writerow(
                        {'stepcount_%d' % self.ID: self.stepCount,
                                  'strategy_%d' % self.ID: self.strategy,
                                  'strat_code_%d' % self.ID: strategy_code,
                                  'move_%d' % self.ID: self.itermove_result,
                                  'utility_%d' % self.ID: totalUtility,
                                  'common_move_%d' % self.ID: self.common_move,
                                  'number_coop_%d' % self.ID: numbC,
                                  'number_defect_%d' % self.ID: numbD,
                                  'mutualC_%d' % self.ID: numbMutC,
                                  'outcomes_%d' % self.ID: outcomes,
                                  'n_partners_%d' % self.ID: numbPartners,
                                  'partner_list_%d' % self.ID: partnerList,
                                  'av_utility_%d' % self.ID: avUtility,
                                  'median_utility_%d' % self.ID: medianUtility,
                                  'av_payoff_%d' % self.ID: avPayoff,
                                  'median_payoff_%d' % self.ID: medianPayoff,
                                  'mood_%d' % self.ID: mood,
                                  'coop_ratio_%d' % self.ID: cooperationRatio,
                                  'similarPartners_%d' % self.ID: similarPartners,
                                  'avPartnerUR_%d' % self.ID: averagePartnerUtilityRatio,
                                  'avPartnerPR_%d' % self.ID: averagePartnerPayoffRatio,
                                  'avPartnerConnected_%d' % self.ID: averagePartnerConnectedness,
                                  'graphConnectedness_%d' % self.ID: graphConnectedness,
                                  'degree_centrality_%d' % self.ID: self.actorDegreeCentrality,
                                  'normalized_centrality_%d' % self.ID: self.normalizedActorDegreeCentrality,
                                  'groupDegreeCent_%d' % self.ID: IGDC,
                                  'globav_%d' % self.ID: self.globalAvPayoff,
                                  'sensitivity_%d' % self.ID: self.sensitivity_mod,
                                  'epsilon_%d' % self.ID: self.epsilon,
                                  #'rejected_%d' % self.ID: self.rejected_partner_list
                                  'blackboard_%d' % self.ID: self.betrayals,
                                   })
            except PermissionError:
                self.output_data_to_file(self.outcome_list)

        else:
            try:
                with open('{}.csv'.format(self.filename), 'a', newline='') as csvfile:
                    fieldnames = ['stepcount_%d' % self.ID,
                                  'strategy_%d' % self.ID,
                                  'strat_code_%d' % self.ID,
                                  'move_%d' % self.ID,
                                  'utility_%d' % self.ID,
                                  'common_move_%d' % self.ID,
                                  'number_coop_%d' % self.ID,
                                  'number_defect_%d' % self.ID,
                                  'mutualC_%d' % self.ID,
                                  'outcomes_%d' % self.ID,
                                  'n_partners_%d' % self.ID,
                                  'partner_list_%d' % self.ID,
                                  'av_utility_%d' % self.ID,
                                  'median_utility_%d' % self.ID,
                                  'av_payoff_%d' % self.ID,
                                  'median_payoff_%d' % self.ID,
                                  'mood_%d' % self.ID,
                                  'coop_ratio_%d' % self.ID,
                                  'similarPartners_%d' % self.ID,
                                  'avPartnerUR_%d' % self.ID,
                                  'avPartnerPR_%d' % self.ID,
                                  'avPartnerConnected_%d' % self.ID,
                                  'graphConnectedness_%d' % self.ID,
                                  'degree_centrality_%d' % self.ID,
                                  'normalized_centrality_%d' % self.ID,
                                  'groupDegreeCent_%d' % self.ID,
                                  'globav_%d' % self.ID,
                                  'sensitivity_%d' % self.ID,
                                  'epsilon_%d' % self.ID,
                                  #'rejected_%d' % self.ID
                                  'blackboard_%d' % self.ID,
                                  ]

                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                    # moves = []
                    # for i in self.move:
                    #     moves.append(self.move[i])

                    if self.stepCount == 1:
                        writer.writeheader()

                    writer.writerow(
                        {'stepcount_%d' % self.ID: self.stepCount,
                         'strategy_%d' % self.ID: self.strategy,
                         'strat_code_%d' % self.ID: strategy_code,
                         'move_%d' % self.ID: self.itermove_result,
                         'utility_%d' % self.ID: totalUtility,
                         'common_move_%d' % self.ID: self.common_move,
                         'number_coop_%d' % self.ID: numbC,
                         'number_defect_%d' % self.ID: numbD,
                         'mutualC_%d' % self.ID: numbMutC,
                         'outcomes_%d' % self.ID: outcomes,
                         'n_partners_%d' % self.ID: numbPartners,
                         'partner_list_%d' % self.ID: partnerList,
                         'av_utility_%d' % self.ID: avUtility,
                         'median_utility_%d' % self.ID: medianUtility,
                         'av_payoff_%d' % self.ID: avPayoff,
                         'median_payoff_%d' % self.ID: medianPayoff,
                         'mood_%d' % self.ID: mood,
                         'coop_ratio_%d' % self.ID: cooperationRatio,
                         'similarPartners_%d' % self.ID: similarPartners,
                         'avPartnerUR_%d' % self.ID: averagePartnerUtilityRatio,
                         'avPartnerPR_%d' % self.ID: averagePartnerPayoffRatio,
                         'avPartnerConnected_%d' % self.ID: averagePartnerConnectedness,
                         'graphConnectedness_%d' % self.ID: graphConnectedness,
                         'degree_centrality_%d' % self.ID: self.actorDegreeCentrality,
                         'normalized_centrality_%d' % self.ID: self.normalizedActorDegreeCentrality,
                         'groupDegreeCent_%d' % self.ID: IGDC,
                         'globav_%d' % self.ID: self.globalAvPayoff,
                         'sensitivity_%d' % self.ID: self.sensitivity_mod,
                         'epsilon_%d' % self.ID: self.epsilon,
                         #'rejected_%d' % self.ID: self.rejected_partner_list
                         'blackboard_%d' % self.ID: self.betrayals,
                         })
            except PermissionError:
                self.output_data_to_file(self.outcome_list)

    # def output_data_to_file(self, outcomes):
    #     """ Outputs the data collected each turn on multiple agent variables to a .csv file"""
    #     for m in self.per_partner_strategies:
    #         if self.per_partner_strategies[m] == self.strategy:
    #             self.similar_partners += 1
    #
    #     prob_list = []
    #     util_list = []
    #     move_list = []
    #     average_list = []
    #
    #     for i in self.indivAvPayoff:
    #         average_list.append(self.indivAvPayoff[i])
    #
    #     # print(average_list)
    #
    #     avpay_partner_1 = 'None'
    #     avpay_partner_2 = 'None'
    #     avpay_partner_3 = 'None'
    #     avpay_partner_4 = 'None'
    #
    #     if len(average_list) == 0:
    #         avpay_partner_1 = 'None'
    #         avpay_partner_2 = 'None'
    #         avpay_partner_3 = 'None'
    #         avpay_partner_4 = 'None'
    #     elif len(average_list) == 1:
    #         avpay_partner_1 = average_list[0]
    #         avpay_partner_2 = 'None'
    #         avpay_partner_3 = 'None'
    #         avpay_partner_4 = 'None'
    #     elif len(average_list) == 2:
    #         avpay_partner_1 = average_list[0]
    #         avpay_partner_2 = average_list[1]
    #         avpay_partner_3 = 'None'
    #         avpay_partner_4 = 'None'
    #     elif len(average_list) == 3:
    #         avpay_partner_1 = average_list[0]
    #         avpay_partner_2 = average_list[1]
    #         avpay_partner_3 = average_list[2]
    #         avpay_partner_4 = 'None'
    #     elif len(average_list) == 4:
    #         avpay_partner_1 = average_list[0]
    #         avpay_partner_2 = average_list[1]
    #         avpay_partner_3 = average_list[2]
    #         avpay_partner_4 = average_list[3]
    #
    #     for i in self.ppD_partner:
    #         prob_list.append(self.ppD_partner[i])
    #
    #     ppd_partner_1 = 'None'
    #     ppd_partner_2 = 'None'
    #     ppd_partner_3 = 'None'
    #     ppd_partner_4 = 'None'
    #
    #     if len(prob_list) == 0:
    #         ppd_partner_1 = 'None'
    #         ppd_partner_2 = 'None'
    #         ppd_partner_3 = 'None'
    #         ppd_partner_4 = 'None'
    #     elif len(prob_list) == 1:
    #         ppd_partner_1 = prob_list[0]
    #         ppd_partner_2 = 'None'
    #         ppd_partner_3 = 'None'
    #         ppd_partner_4 = 'None'
    #     elif len(prob_list) == 2:
    #         ppd_partner_1 = prob_list[0]
    #         ppd_partner_2 = prob_list[1]
    #         ppd_partner_3 = 'None'
    #         ppd_partner_4 = 'None'
    #     elif len(prob_list) == 3:
    #         ppd_partner_1 = prob_list[0]
    #         ppd_partner_2 = prob_list[1]
    #         ppd_partner_3 = prob_list[2]
    #         ppd_partner_4 = 'None'
    #     elif len(prob_list) == 4:
    #         ppd_partner_1 = prob_list[0]
    #         ppd_partner_2 = prob_list[1]
    #         ppd_partner_3 = prob_list[2]
    #         ppd_partner_4 = prob_list[3]
    #
    #     for i in self.per_partner_utility:
    #         util_list.append(self.per_partner_utility[i])
    #
    #     utility_partner_1 = 'None'
    #     utility_partner_2 = 'None'
    #     utility_partner_3 = 'None'
    #     utility_partner_4 = 'None'
    #
    #     if len(util_list) == 0:
    #         utility_partner_1 = 'None'
    #         utility_partner_2 = 'None'
    #         utility_partner_3 = 'None'
    #         utility_partner_4 = 'None'
    #     elif len(util_list) == 1:
    #         utility_partner_1 = util_list[0]
    #         utility_partner_2 = 'None'
    #         utility_partner_3 = 'None'
    #         utility_partner_4 = 'None'
    #     elif len(util_list) == 2:
    #         utility_partner_1 = util_list[0]
    #         utility_partner_2 = util_list[1]
    #         utility_partner_3 = 'None'
    #         utility_partner_4 = 'None'
    #     elif len(util_list) == 3:
    #         utility_partner_1 = util_list[0]
    #         utility_partner_2 = util_list[1]
    #         utility_partner_3 = util_list[2]
    #         utility_partner_4 = 'None'
    #     elif len(util_list) == 4:
    #         utility_partner_1 = util_list[0]
    #         utility_partner_2 = util_list[1]
    #         utility_partner_3 = util_list[2]
    #         utility_partner_4 = util_list[3]
    #
    #     for i in self.itermove_result:  # This encoding is per move type, allows graphing trends in move selection
    #         if self.itermove_result[i] == 'C':
    #             move_list.append(1)
    #             print()
    #         elif self.itermove_result[i] == 'D':
    #             move_list.append(2)
    #
    #     move_partner_1 = 'None'
    #     move_partner_2 = 'None'
    #     move_partner_3 = 'None'
    #     move_partner_4 = 'None'
    #
    #     if len(move_list) == 0:
    #         move_partner_1 = 'None'
    #         move_partner_2 = 'None'
    #         move_partner_3 = 'None'
    #         move_partner_4 = 'None'
    #     elif len(move_list) == 1:
    #         move_partner_1 = move_list[0]
    #         move_partner_2 = 'None'
    #         move_partner_3 = 'None'
    #         move_partner_4 = 'None'
    #     elif len(move_list) == 2:
    #         move_partner_1 = move_list[0]
    #         move_partner_2 = move_list[1]
    #         move_partner_3 = 'None'
    #         move_partner_4 = 'None'
    #     elif len(move_list) == 3:
    #         move_partner_1 = move_list[0]
    #         move_partner_2 = move_list[1]
    #         move_partner_3 = move_list[2]
    #         move_partner_4 = 'None'
    #     elif len(move_list) == 4:
    #         move_partner_1 = move_list[0]
    #         move_partner_2 = move_list[1]
    #         move_partner_3 = move_list[2]
    #         move_partner_4 = move_list[3]
    #
    #     strategy_code = 'None'
    #
    #     if self.strategy == 'RANDOM':
    #         strategy_code = 0
    #     elif self.strategy == 'ANGEL':
    #         strategy_code = 1
    #     elif self.strategy == 'DEVIL':
    #         strategy_code = 2
    #     elif self.strategy == 'EV':
    #         strategy_code = 3
    #     elif self.strategy == 'VEV':
    #         strategy_code = 4
    #     elif self.strategy == 'TFT':
    #         strategy_code = 5
    #     elif self.strategy == 'VPP':
    #         strategy_code = 6
    #     elif self.strategy == 'WSLS':
    #         strategy_code = 7
    #     elif self.strategy == "LEARN":
    #         strategy_code = 8
    #     elif self.strategy == "MOODYLEARN":
    #         strategy_code = 9
    #
    #     """ The above will error catch for when agents don't have those values, and will still let us print
    #         to csv. **** WOULD ALSO LIKE TO DO THIS FOR MOVE PER PARTNER """
    #
    #     if self.strategy == "MOODYLEARN":
    #         try:
    #             self.attempts_taken += 1
    #             with open('{}.csv'.format(self.filename), 'a', newline='') as csvfile:
    #                 fieldnames = ['stepcount_%d' % self.ID, 'strategy_%d' % self.ID, 'strat code_%d' % self.ID,
    #                               'move_%d' % self.ID,
    #                               'probabilities_%d' % self.ID, 'utility_%d' % self.ID, 'common_move_%d' % self.ID,
    #                               'number_coop_%d' % self.ID, 'number_defect_%d' % self.ID,
    #                               'outcomes_%d' % self.ID,
    #                               #'p1_%d' % self.ID, 'p2_%d' % self.ID, 'p3_%d' % self.ID, 'p4_%d' % self.ID,
    #                               'u1_%d' % self.ID,
    #                               'u2_%d' % self.ID,
    #                               'u3_%d' % self.ID,
    #                               'u4_%d' % self.ID,
    #                               'm1_%d' % self.ID, 'm2_%d' % self.ID, 'm3_%d' % self.ID, 'm4_%d' % self.ID,
    #                               'uv_%d' % self.ID,
    #                               'ps_%d' % self.ID, 'nc_%d' % self.ID, 'mutC_%d' % self.ID, 'simP_%d' % self.ID,
    #                               'avp1_%d' % self.ID, 'avp2_%d' % self.ID, 'avp3_%d' % self.ID, 'avp4_%d' % self.ID,
    #                               'globav_%d' % self.ID, 'epsilon_%d' % self.ID, 'alpha_%d' % self.ID, 'mood_%d' % self.ID,
    #                               'sensitivity_%d' % self.ID]
    #
    #                 writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #
    #                 # moves = []
    #                 # for i in self.move:
    #                 #     moves.append(self.move[i])
    #
    #                 if self.stepCount == 1:
    #                     writer.writeheader()
    #
    #                 writer.writerow(
    #                     {'stepcount_%d' % self.ID: self.stepCount,
    #                      'strategy_%d' % self.ID: self.strategy,
    #                      'strat code_%d' % self.ID: strategy_code,
    #                      'move_%d' % self.ID: self.itermove_result,
    #                      'probabilities_%d' % self.ID: self.ppD_partner,
    #                      'utility_%d' % self.ID: self.score,
    #                      'common_move_%d' % self.ID: self.common_move,
    #                      'number_coop_%d' % self.ID: self.number_of_c,
    #                      'number_defect_%d' % self.ID: self.number_of_d,
    #                      'outcomes_%d' % self.ID: outcomes,
    #                      'u1_%d' % self.ID: utility_partner_1,
    #                      'u2_%d' % self.ID: utility_partner_2,
    #                      'u3_%d' % self.ID: utility_partner_3,
    #                      'u4_%d' % self.ID: utility_partner_4,
    #                      'm1_%d' % self.ID: move_partner_1,
    #                      'm2_%d' % self.ID: move_partner_2,
    #                      'm3_%d' % self.ID: move_partner_3,
    #                      'm4_%d' % self.ID: move_partner_4,
    #                      'uv_%d' % self.ID: self.update_value,
    #                      'ps_%d' % self.ID: self.partner_states,
    #                      'nc_%d' % self.ID: self.number_of_c,
    #                      'mutC_%d' % self.ID: self.mutual_c_outcome,
    #                      'simP_%d' % self.ID: self.similar_partners,
    #                      'avp1_%d' % self.ID: avpay_partner_1,
    #                      'avp2_%d' % self.ID: avpay_partner_2,
    #                      'avp3_%d' % self.ID: avpay_partner_3,
    #                      'avp4_%d' % self.ID: avpay_partner_4,
    #                      'globav_%d' % self.ID: self.globalAvPayoff,
    #                      'epsilon_%d' % self.ID: self.moody_epsilon,
    #                      'alpha_%d' % self.ID: self.moody_alpha,
    #                      'mood_%d' % self.ID: self.mood,
    #                      'sensitivity_%d' % self.ID: self.sensitivity_mod})
    #         except PermissionError:
    #             self.output_data_to_file(self.outcome_list)
    #
    #     else:
    #         try:
    #             with open('{}.csv'.format(self.filename), 'a', newline='') as csvfile:
    #                 fieldnames = ['stepcount_%d' % self.ID, 'strategy_%d' % self.ID, 'strat code_%d' % self.ID,
    #                               'move_%d' % self.ID, 'probabilities_%d' % self.ID,
    #                               'utility_%d' % self.ID, 'common_move_%d' % self.ID, 'number_coop_%d' % self.ID,
    #                               'number_defect_%d' % self.ID,
    #                               'outcomes_%d' % self.ID, 'u1_%d' % self.ID, 'u2_%d' % self.ID, 'u3_%d' % self.ID,
    #                               'u4_%d' % self.ID, 'm1_%d' % self.ID, 'm2_%d' % self.ID, 'm3_%d' % self.ID,
    #                               'm4_%d' % self.ID, 'uv_%d' % self.ID,
    #                               'wm_%d' % self.ID, 'nc_%d' % self.ID, 'mutC_%d' % self.ID, 'simP_%d' % self.ID,
    #                               'avp1_%d' % self.ID, 'avp2_%d' % self.ID, 'avp3_%d' % self.ID, 'avp4_%d' % self.ID,
    #                               'globav_%d' % self.ID, 'epsilon_%d' % self.ID, 'alpha_%d' % self.ID, 'mood_%d' % self.ID,
    #                               'sensitivity_%d' % self.ID]
    #
    #                 writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #
    #                 # moves = []
    #                 # for i in self.move:
    #                 #     moves.append(self.move[i])
    #
    #                 if self.stepCount == 1:
    #                     writer.writeheader()
    #
    #                 writer.writerow(
    #                     {'stepcount_%d' % self.ID: self.stepCount,
    #                      'strategy_%d' % self.ID: self.strategy,
    #                      'strat code_%d' % self.ID: strategy_code,
    #                      'move_%d' % self.ID: self.itermove_result,
    #                      'probabilities_%d' % self.ID: self.ppD_partner,
    #                      'utility_%d' % self.ID: self.score,
    #                      'common_move_%d' % self.ID: self.common_move,
    #                      'number_coop_%d' % self.ID: self.number_of_c,
    #                      'number_defect_%d' % self.ID: self.number_of_d,
    #                      'outcomes_%d' % self.ID: outcomes,
    #                      'u1_%d' % self.ID: utility_partner_1,
    #                      'u2_%d' % self.ID: utility_partner_2,
    #                      'u3_%d' % self.ID: utility_partner_3,
    #                      'u4_%d' % self.ID: utility_partner_4,
    #                      'm1_%d' % self.ID: move_partner_1,
    #                      'm2_%d' % self.ID: move_partner_2,
    #                      'm3_%d' % self.ID: move_partner_3,
    #                      'm4_%d' % self.ID: move_partner_4,
    #                      'uv_%d' % self.ID: self.update_value,
    #                      'wm_%d' % self.ID: self.working_memory,
    #                      'nc_%d' % self.ID: self.number_of_c,
    #                      'mutC_%d' % self.ID: self.mutual_c_outcome,
    #                      'simP_%d' % self.ID: self.similar_partners,
    #                      'avp1_%d' % self.ID: avpay_partner_1,
    #                      'avp2_%d' % self.ID: avpay_partner_2,
    #                      'avp3_%d' % self.ID: avpay_partner_3,
    #                      'avp4_%d' % self.ID: avpay_partner_4,
    #                      'globav_%d' % self.ID: self.globalAvPayoff,
    #                      'epsilon_%d' % self.ID: self.epsilon,
    #                      'alpha_%d' % self.ID: self.alpha,
    #                      'mood_%d' % self.ID: self.mood,
    #                      'sensitivity_%d' % self.ID: self.sensitivity_mod})
    #         except PermissionError:
    #             self.output_data_to_file(self.outcome_list)

    def reset_values(self):
        """ Resets relevant global variables to default values. """
        self.number_of_d = 0
        self.number_of_c = 0
        self.update_value = self.init_uv
        self.mutual_c_outcome = 0
        self.similar_partners = 0

    # TODO: Move all the kNN stuff to a separate script that we just reference

    def knn_decision(self, partner_ids, partner_utils, partner_selfcoops, partner_oppcoops, partner_mutcoops, ppds):

        """ ppds needs to be self.model.agent_ppds"""
        updated_ppds = []
        old_ppds = self.model.agent_ppds[self.ID]
        # print("The old ppds are", old_ppds)

        updated_ppds = old_ppds
        training_data = copy.deepcopy(self.model.training_data)
        # print(len(training_data))
        decision_data = copy.deepcopy(self.model.training_data)

        for i in partner_ids:
            # training_data_list = training_data
            partner_index = partner_ids.index(i)
            game_utility = partner_utils[i]
            game_selfcoops = partner_selfcoops[i]
            game_oppcoops = partner_oppcoops[i]
            game_mutcoops = partner_mutcoops[i]
            game_ppd = ppds[i]

            """ The bit above might not work; because when we get ppds from the model it's a 4-long list,
                and some agents only use the first 2 to 3 items, we need to update the ppds in the list by 
                their indices to let them be used against the same agent next game"""

            class_list, classification = self.knn_analysis(game_utility, game_selfcoops, game_oppcoops, game_mutcoops,
                                                           game_ppd, training_data,
                                                           self.model.k)
            priority = "U"

            # print("My ID:", self.ID,
            #       "Partner ID:", i,
            #       "k Classifications:", class_list,
            #       "Decided Class:", classification)
            self.knn_error_statement(classification, i)
            # print("kNN was", self.knn_error_statement(classification, i))

            # so far, we should have a knn classification of what the ith partner is, which we then feed in to
            new_ppd = self.ppd_select(decision_data, classification, priority)

            updated_ppds[partner_index] = new_ppd

        # print("The Old ppds were:", old_ppds)
        # print("The new ppds are", updated_ppds)
        self.model.agent_ppds[self.ID] = updated_ppds
        return

    def knn_error_statement(self, classification, opp_id):

        strat = 0

        if self.per_partner_strategies[opp_id] == 'VPP':
            strat = 1
        elif self.per_partner_strategies[opp_id] == 'ANGEL':
            strat = 2
        elif self.per_partner_strategies[opp_id] == 'DEVIL':
            strat = 3
        elif self.per_partner_strategies[opp_id] == 'TFT':
            strat = 4
        elif self.per_partner_strategies[opp_id] == 'WSLS':
            strat = 5
        elif self.per_partner_strategies[opp_id] == 'iWSLS':
            strat = 6

        if classification != strat:
            return "Wrong"
        else:
            self.model.kNN_accuracy += 1
            return "Right"

    def BinaryPPDSearch(self, list, value, n_times, indx):
        """ This should return a list of indexes to search the data with """

        copy_list = copy.deepcopy(list)
        # print("bsearchinput", len(list), value, n_times, indx)
        # index_list = []
        data_list = []
        n_times = int(n_times)

        for i in range(0, n_times):     # would alternatively prefer this to be a while loop
            index = self.BSearch(copy_list, value, indx)  # get the index of the ppd item
            # print("I found an index, it's", index)
            # print(len(copy_list))
            if index != -1:
                # index_list.append(index)  # index list is garbage because of popping
                # print("The item is", copy_list[index])
                data_list.append(copy_list[index])
                # print("I appended it and the list now has", len(data_list), "items")
                # copy_list[index] = [0, 0, 0.0, 0]
                copy_list.pop(index)
                # print("I removed it")
            else:
                copy_list.pop(index)

        """ So, could we improve this by running the Search function indefinitely until it can no longer 
            find the value we're looking for? """
        # print("agent", self.ID, "index:",indx,"bsearch output", data_list)
        return data_list

    def BSearch(self, lys, val, indx_value):
        first = 0
        last = len(lys) - 1
        index = -1
        while (first <= last) and (index == -1):
            mid = (first + last) // 2
            if lys[mid][indx_value] == val:
                index = mid
            else:
                if val < lys[mid][indx_value]:
                    last = mid - 1
                else:
                    first = mid + 1
        return index

    def knn_analysis(self, utility, selfcoops, oppcoops, mutcoops, ppd, training_data, k):
        """ Takes an input, checks it against training data, and returns a partner classification """
        classification = 1  # CLASSES START AT 1 INSTEAD OF 0 BECAUSE IM A FOOL
        # print("Initialising knn analysis")
        current_data = [utility, selfcoops, oppcoops, mutcoops, ppd]
        current_data2 = [utility, selfcoops, ppd]
        # print(current_data)
        # print(len(training_data), type(training_data))

        relevant_data = []
        r_data_indices = []
        r_data_distances_to_goal = []

        # for i in training_data:
        #     """ We'll just use standard linear search for now, and maybe implement something faster later"""
        #     # get the third index of i
        #     if i[2] == ppd:
        #         relevant_data.append(i)
        #         r_data_indices.append(training_data.index(i))

        # For Binary Search, we need to know how many times to search the list - I think with the 113,400 data
        # it's 12,600 data points per ppd we collected
        # print("gonna do binary search with ppd of", ppd)
        training_size = len(training_data)
        relevant_data = self.BinaryPPDSearch(training_data, ppd, (training_size/3), 4)

        # print("rd", relevant_data)
        # can't get the indices from this, nor replace properly - but we don't need indices for a later search
        # because the search for optimal values will be a separate search

        for i in relevant_data:
            """ We take each item and calculate the Euclidean distance to the data we already have"""
            slice = i[:5]   #  =========== THIS MIGHT NEED TO BE 5 =======
            slice2 = [i[0], i[1], i[5]]
            distance_to_target = dst.cosine(current_data, slice)
            # print("data:", i, "distance:", distance_to_target)
            r_data_distances_to_goal.append(distance_to_target)

        # print("rel data", relevant_data)
        # print("distances", r_data_distances_to_goal)

        """ Now we have a list of distances to our current dataset, need to select k closest in terms of utility 
        and cooperations. Then access the relevant data and find the classification values ( i[3]) for them. """

        # sorted_distances = {k: v for k, v in sorted(r_data_distances_to_goal.items(), key=lambda item: item[1])}
        # # this may or may not work, it's a method taken from elsewhere...

        #
        ascending_data = sorted(zip(relevant_data, r_data_distances_to_goal), key=lambda t: t[1])[0:]
        # print("ass data", ascending_data)
        # print("ascend", ascending_data)
        # print(len(ascending_data))
        categories = []

        for i in range(0, k):
            # print("ass data2", ascending_data)
            temp = ascending_data[i]
            categories.append(temp[0][5])

        """Then, we find the most common category offered and return it. """
        # print("The k closest categories were:", categories)
        try:
            classification = statistics.mode(categories)
        except statistics.StatisticsError:
            classification = categories[0]
            # classification = statistics.mode(categories[0])

            # tryk = copy.deepcopy(k)
            # tryk = int((tryk/2)-1)

            # for i in range(0, tryk):
            #     # print("ass data2", ascending_data)
            #     temp = ascending_data[i]
            #     categories.append(temp[0][3])

            # classification = statistics.mode(categories)

            # if k < 3:
            #     classification = statistics.mode(categories[0])
            # if k > 3:
            #     classification = statistics.mode(categories[0:5])
        # print("The most common classification from the k neighbours is:", classification)
        # TODO - in future might want to weight each neighbour by closeness
        return categories, classification

    def ppd_select(self, list, classification, optimisation_choice):
        """ Takes a class of partner, given by the kNN algorithm, and returns a starting ppD to
        use in future games for the same partner based on which variable (or combo) we want to optimise """
        # don't need to make a deep copy of this list, because we're not altering it
        # print("pdsel", list, classification, optimisation_choice)
        # relevant_data = []

        relevant_data = self.BinaryPPDSearch(list,
                                             classification,
                                             (len(list)), 5)  # need to decide how many times to run this for
        npRev = np.array(relevant_data)
        # if len(npRev) == 0:
            # print("I'm agent", self.ID, "and ", len(list), classification, optimisation_choice, len(relevant_data))
            # print("npRev empty")

        col = 0
        if optimisation_choice == 'MC':  # optimise for MY cooperation
            col = 1
        elif optimisation_choice == 'U':  # optimise to maximise utility
            col = 0
        elif optimisation_choice == 'OC':
            col = 2
        elif optimisation_choice == 'MutC':
            col = 3

        npRev = npRev[np.argsort(npRev[:, col])]
        npRev = np.ndarray.tolist(npRev)
        # so, now we have a list sorted by the column we prefer, we can select the ppD associated with the highest score
        # print("checking npRev", npRev)

        highest = npRev[len(npRev)-1]
        new_ppd = highest[4]
        # print("I'm agent", self.ID, "My partner's class is", classification, "and the ppd I should use for them is", new_ppd)

        return new_ppd

    def find_average_move(self):
        """ Counts up how many of each behaviour type was performed that round and returns which was
            the most commonly chosen move (or Eq if there was a tie). """
        move_list = []
        for n in self.itermove_result:
            move_list.append(self.itermove_result[n])

        move_counter = {}
        for move in move_list:
            if move in move_counter:
                move_counter[move] += 1
            else:
                move_counter[move] = 1
        # print("Move counter:", move_counter)

        if move_counter.get('C') and move_counter.get('D') is not None:

            if move_counter['C'] == move_counter['D']:
                self.common_move = 'Eq'
                # print("Moves", self.move, "Counters: ", move_counter)
                # print("My most chosen move is:", self.common_move, 'because my counters are:', move_counter['C'],
                # move_counter['D'])

            else:
                commonest_move = sorted(move_counter, key=move_counter.get, reverse=True)
                self.common_move = commonest_move[
                                   :1]  # This isn't perfect as it doesn't display ties -----------------------
                # print("My most chosen move is:", self.common_move)
        else:
            commonest_move = sorted(move_counter, key=move_counter.get, reverse=True)
            self.common_move = commonest_move[:1]

    def outputQtable(self, init):
        # if I am the chosen one
        qvalues = []
        if self.ID == self.model.chosenOne:
            # get my qtable
            for i in self.qtable:
                pair = self.qtable[i]
                for j in pair:
                    qvalues.append(j)

            # for each numerical value in it, append it to a new list
            # for each item in that list, open up the csv and print it to it
        if init:
            for i in qvalues:
                with open('{} qinit_agent37.csv'.format(self.model.filename), 'a', newline='') as csvfile:
                    fieldnames = ['q']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    # writer.writeheader()
                    writer.writerow({'q': i})
        else:
            for i in qvalues:
                with open('{} qend_agent37.csv'.format(self.model.filename), 'a', newline='') as csvfile:
                    fieldnames = ['q']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    # writer.writeheader()
                    writer.writerow({'q': i})

    def outputData(self, fake):
        # print("It's round", self.stepCount, "and I exported data whilst fake was ", fake)
        if not fake:
            self.output_data_to_model()
            if self.model.collect_data:
                self.output_data_to_file(self.outcome_list)
                #if self.attempts_taken > 1:
                    # print("Agent", self.ID, "on step", self.stepCount, "took", self.attempts_taken, "attempt/s to export.")
                self.attempts_taken = 0
        else:
            self.output_data_to_model()
            if self.model.collect_data:
                self.fake_data_to_file(self.outcome_list)
                # if self.attempts_taken > 1:
                # print("Agent", self.ID, "on step", self.stepCount, "took", self.attempts_taken, "attempt/s to export.")
                self.attempts_taken = 0

    def avScore(self):
        pays = []
        for i in self.current_partner_list:
            if i in self.per_partner_payoffs:
                end = len(self.per_partner_payoffs[i]) - 1
                item = self.per_partner_payoffs[i]
                pays.append(item[end])
            else:
                pass
        if len(pays) == 0:
            pays = [0]
        # print("My per partner payoff is: ", pays)
        avPayoff = sum(pays) / len(pays)
        return avPayoff


    def set_starting_oldstates(self, strategy, learning_from, size):
        if strategy is not 'MOODYLEARN':
            if learning_from == "me":
                zeroes = []
                for j in range(size):
                    zeroes.append(0)
                return zeroes
            elif learning_from == "them":
                zeroes = []
                for j in range(size):
                    zeroes.append(0)
                return zeroes
            elif learning_from == "us":
                zeroes = []
                for j in range(size):
                    zeroes.append((0, 0))
                return zeroes
        else:
            if learning_from == "me":
                zeroes = []
                for j in range(size):
                    zeroes.append(0)
                return zeroes
            elif learning_from == "them":
                zeroes = []
                for j in range(size):
                    zeroes.append(0)
                return zeroes
            elif learning_from == "us":
                zeroes = []
                for j in range(size):
                    zeroes.append((0, 0))
                return zeroes

    def averageScoreComparison(self, oppID, moodyStrat, current_partners):
        """ Returns my average score, compared with a specific partner? """
        scores = {}
        payoffs = {}
        recent_payoffs = {}
        averages = {}

        x, y = self.pos
        neighbouring_agents = current_partners
        neighbouring_cells = []

        # neighbouring_cells = [(x, y + 1), (x + 1, y), (x, y - 1), (x - 1, y)]  # N, E, S, W
        for partner in neighbouring_agents:
            neighbouring_cells.append(self.model.agent_positions[partner])

        # First, get the neighbours
        for i in neighbouring_cells:
            bound_checker = self.model.grid.out_of_bounds(i)
            if not bound_checker:
                this_cell = self.model.grid.get_cell_list_contents([i])
                # print("This cell", this_cell)
                if self.stepCount == 2:
                    self.n_partners += 1

                if len(this_cell) > 0:
                    partner = [obj for obj in this_cell
                               if isinstance(obj, PDAgent)][0]

                    partner_ID = partner.ID
                    if partner.indivAvPayoff.get(self.ID) is None:  # if you try and get their average payoff against me and it isn't there
                        if moodyStrat:
                            averages[partner_ID] = self.moody_pp_oppPayoff[partner_ID]  # instead use the payoff for that turn
                        else:
                            averages[partner_ID] = self.pp_oppPayoff[partner_ID]
                    else:
                        averages[partner_ID] = partner.indivAvPayoff[self.ID]

            my_average = self.indivAvPayoff[oppID]  # BEWARE, THIS IS A ===FULL AVERAGE==== NOT AN AVERAGE OVER X PERIODS, averaged over per_partner_payoffs (full history)
            #TODO: does this need to give partner's score against me, or score as a whole? Because if ============================================================================================
            #TODO: it's the latter, you could use pp_utility from the opponent
        if moodyStrat:
            return my_average, averages[oppID], self.moody_pp_oppPayoff[oppID]
        else:
            return my_average, averages[oppID], self.pp_oppPayoff[partner_ID]

    def step(self):

        if not self.model.resetTurn:
            self.compare_score()
            self.reset_values()
            self.current_partner_list = copy.deepcopy(self.model.updated_graphD[self.ID])

            """  So a step for our agents, right now, is to calculate the utility of each option and then pick? """
            # TODO: the below will set the starting behaviour dict for everyone to have options for every possible partner they will ever have.
            # This will become a problem if agents leave or join the network, or if the number of agents in the network changes,
            # but for now, this is an easier solution

            if self.stepCount == 1:
                self.current_partner_list = copy.deepcopy(self.model.updated_graphD[self.ID])
                self.partner_IDs = copy.deepcopy(self.current_partner_list)
                self.set_defaults(self.partner_IDs)

                self.get_IDs(self.current_partner_list)
                for i in self.current_partner_list:
                    for j in self.all_possible_partners:
                        if j != i:
                            self.potential_partner_list.append(j)
                for i in self.partner_IDs:
                    self.oldstates[i] = self.set_starting_oldstates(self.strategy, self.model.learnFrom, self.delta)
                    if self.statemode == 'stateless':
                        md = 1
                    elif self.statemode == 'agentstate':
                        md = 2
                    elif self.statemode == 'moodstate':
                        md = 3
                    self.moody_oldstates[i] = self.set_starting_oldstates(self.strategy, self.model.moody_learnFrom, md)

                if self.strategy is None or 0 or []:
                    self.strategy = self.pick_strategy()
                    # self.next_move = self.pick_move(self.strategy, self.payoffs, 0)
                    self.previous_moves.append(self.move)
                    self.set_defaults(self.partner_IDs)
                    # print("My ppDs are:", self.ppD_partner)
                    # print("my name is agent ", self.ID, "my strategy is ", self.strategy)
                    if self.strategy == 'LEARN':
                        # Initialise the q tables and states on the first turn
                        self.qtable = sarsa.init_qtable(copy.deepcopy(self.model.memory_states), 2, True)
                        self.states = copy.deepcopy(self.model.memory_states)
                    if self.strategy == 'MOODYLEARN':
                        # Initialise the q tables and states on the first turn
                        self.moody_qtable = sarsa_moody.init_qtable(copy.deepcopy(self.model.moody_memory_states), 2, True)
                        self.state_working_memory = sarsa_moody.init_statememory(copy.deepcopy(self.model.moody_memory_states), 2, self.moody_delta)
                        # print('init qtable len:', len(self.moody_qtable))
                        self.moody_states = copy.deepcopy(self.model.moody_memory_states)

                    new_moves = self.iter_pick_move(self.strategy, self.payoffs, self.current_partner_list, [0])
                    for n in self.current_partner_list:
                        self.itermove_result[n] = new_moves[n]
                    # self.itermove_result =
                    self.find_average_move()
                    if self.stepCount > 1:
                        self.avPayoff = self.avScore()

                    if self.model.schedule_type != "Simultaneous":
                        self.advance()

                    # self.stepCount += 1
                else:
                    self.current_partner_list = copy.deepcopy(self.model.updated_graphD[self.ID])
                    self.partner_IDs = copy.deepcopy(self.current_partner_list)
                    self.set_defaults(self.partner_IDs)
                    for i in self.current_partner_list:
                        for j in self.all_possible_partners:
                            if j != i:
                                self.potential_partner_list.append(j)
                    # print("My ppDs are:", self.ppD_partner)

                    if self.strategy == 'LEARN':
                        # Initialise the q tables and states on the first turn
                        self.qtable = sarsa.init_qtable(copy.deepcopy(self.model.memory_states), 2, True)
                        self.states = copy.deepcopy(self.model.memory_states)
                    if self.strategy == 'MOODYLEARN':
                        # Initialise the q tables and states on the first turn
                        self.qtable = sarsa_moody.init_qtable(copy.deepcopy(self.model.moody_memory_states), 2, True)
                        # print('init qtable len:', len(self.moody_qtable))
                        self.states = copy.deepcopy(self.model.moody_memory_states)

                    new_moves = self.iter_pick_move(self.strategy, self.payoffs, self.current_partner_list, [0])
                    for n in self.current_partner_list:
                        self.itermove_result[n] = new_moves[n]
                    # self.itermove_result =
                    self.previous_moves.append(self.move)
                    self.find_average_move()
                    if self.stepCount > 1:
                        self.avPayoff = self.avScore()

                    if self.model.schedule_type != "Simultaneous":
                        self.advance()

            else:
                if self.strategy is None or 0 or []:
                    self.strategy = self.pick_strategy()
                    # self.next_move = self.pick_move(self.strategy, self.payoffs, 0)
                    self.previous_moves.append(self.move)

                    if self.strategy == 'LEARN':
                        #if self.pp_aprime exists, itermove_result = copy.deepcopy(self.pp_aprime)
                        #clear next_action?
                        if self.pp_aprime:
                            primes = copy.deepcopy(self.pp_aprime)
                            for n in self.current_partner_list:
                                self.itermove_result[n] = primes[n]
                            # self.itermove_result = copy.deepcopy(self.pp_aprime)
                            # TO/DO: Do I then need to wipe the aprime dict?
                    elif self.strategy == 'MOODYLEARN':
                        # if self.pp_aprime exists, itermove_result = copy.deepcopy(self.pp_aprime)
                        # clear next_action?
                        if self.moody_pp_aprime:
                            primes = copy.deepcopy(self.moody_pp_aprime)
                            for n in self.current_partner_list:
                                self.itermove_result[n] = primes[n]
                            # self.itermove_result = copy.deepcopy(self.moody_pp_aprime)
                    else:
                        moves = self.iter_pick_move(self.strategy, self.payoffs, self.current_partner_list, [0])
                        for n in self.current_partner_list:
                            self.itermove_result[n] = moves[n]
                        # self.itermove_result =

                    self.find_average_move()
                    if self.stepCount > 1:
                        self.avPayoff = self.avScore()

                    if self.model.schedule_type != "Simultaneous":
                        self.advance()

                    # self.stepCount += 1
                else:

                    if self.strategy == 'LEARN':
                        if self.pp_aprime:
                            primes = copy.deepcopy(self.pp_aprime)
                            for n in self.current_partner_list:
                                self.itermove_result[n] = primes[n]
                            # TO/DO: Do I then need to wipe the aprime dict?
                    elif self.strategy == 'MOODYLEARN':
                        if self.moody_pp_aprime:
                            primes = copy.deepcopy(self.moody_pp_aprime)
                            for n in self.current_partner_list:
                                self.itermove_result[n] = primes[n]
                    else:
                        moves = self.iter_pick_move(self.strategy, self.payoffs, self.current_partner_list, [0])
                        for n in self.current_partner_list:
                            self.itermove_result[n] = moves[n]

                    self.previous_moves.append(self.move)  # Does this need to be here? Why is it nowhere else?
                    self.find_average_move()
                    if self.stepCount > 1:
                        self.avPayoff = self.avScore()

                    if self.model.schedule_type != "Simultaneous":
                        self.advance()

            if self.stepCount == (self.model.rounds - 1):
                self.last_round = True

            # self.stepCount += 1

            if self.printing:
                for n in range(1):
                    print("----------------------------------------------------------")
        else:
            # We need to generate a number of partners that are allowed to switch and w/ whom they form a new connection
            #print("IT'S A RESET TURN! and my selection strat is,", self.partnerSelectionStrat)
            """ If it's a reset round where we just change partners, all we want to do is update our partner list.
                The method by which I do this is randomly, for this section. """
            self.current_partner_list = copy.deepcopy(self.model.updated_graphD[self.ID])
            new_partners = []
            removed_partners = []

            for i in self.current_partner_list:
                if self.itermove_result.get(i) is None:
                    new_partners.append(i)

            for n in new_partners:
                "I WILL GIVE THEM A DEFAULT"
                self.itermove_result[n] = self.model.startingBehav

            self.get_IDs(self.current_partner_list)
            # Set defaults for any new partners, remove any old partners from the lists
            if len(new_partners) > 0:
                for i in new_partners:
                    for j in self.all_possible_partners:
                        if j != i:
                            self.potential_partner_list.append(j)
                for i in new_partners:
                    self.oldstates[i] = self.set_starting_oldstates(self.strategy, self.model.learnFrom, self.delta)
                    if self.statemode == 'stateless':
                        md = 1
                    elif self.statemode == 'agentstate':
                        md = 2
                    elif self.statemode == 'moodstate':
                        md = 3
                    self.moody_oldstates[i] = self.set_starting_oldstates(self.strategy, self.model.moody_learnFrom, md)

                self.set_defaults(new_partners)
                # Do an iter_pick_move ?
                # We need to pick default moves for new partners...
                # pickmove won't work though because we don't have any prior history with them
                # I have set it so that if the ID appears in new_partners, they will just do they default behaviour?
                moves = self.iter_pick_move(self.strategy, self.payoffs,
                                                            self.current_partner_list, new_partners)
                for n in self.current_partner_list:
                    self.itermove_result[n] = moves[n]  # We now try and find our partners from here

                self.find_average_move()
                if self.stepCount > 1:
                    self.avPayoff = self.avScore()

            for i in self.current_partner_list:
                if i not in self.partner_IDs:
                    self.partner_IDs.remove(j)
                    removed_partners.append(j)
                    # print("partner I removed was", i)

            if self.stepCount == (self.model.rounds - 1):
                self.last_round = True
            self.find_average_move()
            if self.stepCount > 1:
                self.avPayoff = self.avScore()
            if self.model.schedule_type != "Simultaneous":
                self.advance()

            # elif self.partnerSelectionStrat == "SCORE":  # ===========================================================
            #
            #
            # elif self.partnerSelectionStrat == "REP":  # ===========================================================
            #     print("IT'S A RESET TURN! and my selection strat is,", self.partnerSelectionStrat)
            #     """ If it's a reset round where we just change partners, all we want to do is update our partner list.
            #         The method by which I do this is [TO BE FINALISED]"""
            #
            #     if self.stepCount == (self.model.rounds - 1):
            #         self.last_round = True
            #     self.find_average_move()
            #     if self.model.schedule_type != "Simultaneous":
            #         self.advance()
            #
            # else:
            #     print("IT'S A RESET TURN! and my selection strat is,", self.partnerSelectionStrat)
            #     if self.stepCount == (self.model.rounds - 1):
            #         self.last_round = True
            #     self.find_average_move()
            #     if self.model.schedule_type != "Simultaneous":
            #         self.advance()


    def advance(self):
        """ If we have partners, execute normal strategy. If we do not, find a partner.
            Also, if it's a reset round - do none of these things, and just """
        if not self.model.resetTurn:
            if len(self.current_partner_list) > 0:
                if self.strategy == 'LEARN':
                    self.check_partner(self.current_partner_list)  # We took action a, what s prime did we end up in?
                    # ----- WORKING MEMORY IS NOW S-PRIME -----
                    round_payoffs = self.increment_score(self.payoffs)  # Accept the reward for that s prime

                    # update the sprimes (the states we have found ourselves in)
                    # for i in self.working_memory:
                    #     state = self.working_memory[i]
                    #     obs = self.partner_latest_move[i]
                    #     self.pp_sprime[i] = sarsa.output_sprime(state, obs)

                    # get aprimes (next actions to do)
                    primemoves = self.iter_pick_nextmove(self.strategy, self.payoffs, self.working_memory, self.current_partner_list)
                    for n in self.current_partner_list:
                        self.pp_aprime[n] = primemoves[n]

                    # update the Q for the CURRENT sprime

                    for i in self.partner_IDs:
                        s = self.oldstates[i]           # the state I used to be in
                        # print("(line 2674)my id is", self.ID, "my partners are", self.partner_IDs, "and my moves were", self.itermove_result)
                        a = self.itermove_result[i]     # the action I took
                        sprime = self.working_memory[i] # the state I found myself in
                        reward = self.pp_payoff[i]      # the reward I observed
                        aprime = self.pp_aprime[i]      # the action I will take next

                        # print('ostates=', self.oldstates)
                        # print('sstates=', self.itermove_result)
                        # print('sprimes=', self.working_memory)
                        if self.delta == 1:
                            if self.model.memoryPaired:
                                s = s[0]
                        oldQValues = self.qtable[tuple(s)]  # THIS MIGHT BREAK BECAUSE OF TUPLES

                        if a == 'C':  # This still works because it's keyed off the itermove_result and not part of the state
                            idx = 0
                        elif a == 'D':
                            idx = 1

                        if self.delta == 1:
                            if self.model.memoryPaired:
                                sprime = sprime[0]
                        newQValues = self.qtable[tuple(sprime)]  # THIS ISN'T RIGHT IS IT?
                        if aprime == 'C':
                            idxprime = 0
                        elif aprime == 'D':
                            idxprime = 1

                        Qsa = oldQValues[idx]
                        Qsaprime = newQValues[idxprime]

                        # update the Q value for the old state and old action

                        newQsa = sarsa.update_q(reward, self.gamma, self.alpha, Qsa, Qsaprime)
                        # print('My old Q for this partner was:', Qsa, 'and my new Q is:', newQsa)
                        # then put newQ in the Qtable[s] at index idx
                        change = self.qtable[tuple(s)]
                        change[idx] = newQsa
                        self.qtable[tuple(s)] = change

                        if self.model.moody_opponents:
                            myAv, oppAv, oppScore = self.averageScoreComparison(i, False, self.current_partner_list)
                            # TODO: ARE THE SCORES BELOW SCORES AGAINST EACH PARTNER, OR ARE THEY TOTAL SCORES?
                            # if self.ID == 9:
                            #     print("It's turn ", self.stepCount)
                            #     print("My mood going into this was ", self.mood)
                            #     print("My values were ", reward, myAv, oppScore, oppAv)
                            self.mood, self.sensitivity_mod = sarsa_moody.update_mood_new(self.mood, reward, myAv, oppScore, oppAv, self.sensitive, self.sensitivity_mod)
                            # if self.ID == 9:
                            #     print("My mood coming out of it was ", self.mood)


                    # for i in self.working_memory:
                    #     state = self.working_memory[i]
                    #     my_move = self.itermove_result[i]
                    #     reward = self.pp_payoff[i]  # the unmanipulated value
                    #
                    #     # for the states I'm in, update the relevant q value
                    #     # access the qtable value we already have
                    #     oldQValues = self.qtable[state]   # THIS MIGHT BREAK BECAUSE OF TUPLES
                    #     if my_move == 'C':
                    #         idx = 0
                    #     elif my_move == 'D':
                    #         idx = 1
                    #
                    #     next_state = self.pp_sprime[i]
                    #     nextQValues = self.qtable[next_state]
                    #     nextQValue = nextQValues[idx]  # this is wrong, just want
                    #
                    #     oldQValue = oldQValues[idx]
                    #     new_value = sarsa.update_q(reward, self.gamma, self.alpha, oldQValue)

                    # update epsilon
                    self.epsilon = sarsa.decay_value(self.model.epsilon, self.epsilon, self.model.rounds, True, self.model.epsilon_floor)
                    self.alpha = sarsa.decay_value(self.model.alpha, self.alpha, self.model.rounds, True, self.model.alpha_floor)

                    # update s to be sprime
                    for i in self.partner_IDs:
                        self.oldstates[i] = self.working_memory[i]

                    if self.model.export_q:
                        if self.stepCount == 1:
                            self.outputQtable(True)
                        elif self.stepCount == self.model.rounds - 1:
                            self.outputQtable(False)

                    self.outputData(False)

                    # =============== UPDATE YOUR PARTNERS AND WHO U WANT AS PARTNERS =========
                    self.current_partner_list = copy.deepcopy(self.model.updated_graphD[self.ID])
                    if self.model.checkTurn:      # todo: this needs to  be repeated on all the other checkturns
                        # =============== CHECK IF YOU ARE ELIGIBLE FOR RESTRUCTURING =============
                        connections = []
                        for i in self.model.agentsToRestructure:
                            if i[0] == self.ID:
                                # if we have a connection to evaluate,
                                connections.append(i)
                            else:
                                pass


                        # =============== CHECK IF RESTRUCTURE CONNECTION EXISTS AND WHAT TO DO W/ IT ==========
                        break_check = []
                        gain_check = []
                        # then, for each of these, we check if that connection exists, and what to do with it
                        if len(connections) > 0:
                            for i in connections:
                                if i[1] not in self.current_partner_list:
                                    gain_check.append(i[1])
                                else:
                                    if i[1] in self.current_partner_list:
                                        break_check.append(i[1])

                        # TODO: this only happens once, when actually it needs to happen for each connection in the restructuring list.
                        # TODO: partnerDecision needs to be a decision per partner, returning single removRequest over and over, which are each removed in that moment from the model

                        if self.model.complexRestructuring:
                            if len(break_check) > 0:
                                for partnerID in break_check:
                                    toBreak = True
                                    #print("1my partners are", self.current_partner_list)
                                    #print("1my wm is:", self.working_memory)
                                    #print("1my pp_oppP is:", self.pp_oppPayoff[partnerID])
                                    request = rnf.partnerDecision(toBreak, self.partnerSelectionStrat, partnerID,
                                                                  self.ID, self.rejected_partner_list,
                                                                  self.working_memory[partnerID],
                                                                  self.indivAvPayoff[partnerID],  # using this instead of working memory as sometimes wm breaks
                                                                  self.oppAvPayoff[partnerID],
                                                                  #self.model.CC,
                                                                  2,
                                                                  #self.model.reputationBlackboard[self.ID],
                                                                  self.model.averageBetrayals,  # this is the target reputation
                                                                  self.mood,
                                                                  50,
                                                                  #self.model.reputationBlackboard[partnerID],
                                                                  self.check_item(partnerID, "rep"),
                                                                  self.normalizedActorDegreeCentrality)
                                    # check if the request is valid (aka, not Null) - if it is, send it to the model
                                    #   if it isn't, ignore it
                                    #print("1request outcome was ", request)
                                    if request[1] != None:
                                        self.model.graph_removals.append(request)
                                        self.rejected_partner_list.append(partnerID)
                                        #print("My strat is", self.strategy, "and I just rejected", partnerID)
                                        self.current_partner_list.remove(partnerID)

                            if len(self.current_partner_list) < self.model.maximumPartners:
                                if len(gain_check) > 0:
                                    for partnerID in gain_check:
                                        toBreak = False
                                        # then do the partner decision line
                                        #print("2my partners are", self.current_partner_list)
                                        #print("2my wm is:", self.working_memory)
                                        #print("2my pp_oppP is:", 0)
                                        request = rnf.partnerDecision(toBreak, self.partnerSelectionStrat, partnerID,
                                                                      self.ID, self.rejected_partner_list,
                                                                      [0, 0, 0, 0, 0, 0, 0],
                                                                      0,
                                                                      0,
                                                                      #self.model.CC,
                                                                      2,
                                                                      #self.model.reputationBlackboard[self.ID],
                                                                      self.model.averageBetrayals,
                                                                      # this is the target reputation
                                                                      self.mood,
                                                                      50,
                                                                      #self.model.reputationBlackboard[partnerID],
                                                                      self.check_item(partnerID, "rep"),
                                                                      self.normalizedActorDegreeCentrality)
                                        # check if the request is valid (aka, not Null) - if it is, send it to the model
                                        #   if it isn't, ignore it
                                        #print("2request outcome was ", request)
                                        if request[1] != None:
                                            self.model.graph_additions.append(request)
                            # else:
                            #   todo: some kind of pruning behaviour

                        else:
                            removRequest, addRequest, removals, additions = rnf.basicPartnerDecision(self.current_partner_list, self.rejected_partner_list,
                                                                                                 self.potential_partner_list, self.all_possible_partners,
                                                                                                 0.4, self.ID)
                            if removRequest:
                                if removRequest[1] != None:
                                    self.model.graph_removals.append(removRequest)
                                    self.current_partner_list.remove(removals)
                                    self.rejected_partner_list.append(removals)
                                    #print("My strat is", self.strategy, "and I just rejected", partnerID)
                            if addRequest:
                                if addRequest[1] != None:
                                    self.model.graph_additions.append(addRequest)

                    # if additions not in self.current_partner_list:
                    #     self.current_partner_list.append(additions)
                    # if removals not in self.rejected_partner_list:
                    #     self.rejected_partner_list.append(removals)
                    # self.partner_IDs = copy.deepcopy(self.current_partner_list)

                    self.stepCount += 1

                    if round_payoffs is not None:
                        if self.printing:
                            print("I am agent", self.ID, ", and I have earned", round_payoffs, "this round")
                        self.score += round_payoffs
                        # print("My total overall score is:", self.score)
                        return

                elif self.strategy == 'MOODYLEARN':
                    self.check_partner(self.current_partner_list)  # We took action a, what s prime did we end up in?
                    """This should hopefully update the state for pick_nextmove"""
                    # ----- WORKING MEMORY IS NOW S-PRIME -----
                    round_payoffs = self.increment_score(self.payoffs)  # Accept the reward for that s prime  #TODO: MIGHT NEED A MOODY INCREMENT_SCORE

                    # update the sprimes (the states we have found ourselves in)
                    # for i in self.working_memory:
                    #     state = self.working_memory[i]
                    #     obs = self.partner_latest_move[i]
                    #     self.pp_sprime[i] = sarsa.output_sprime(state, obs)

                    # print("my id is", self.ID)
                    # print("step", self.stepCount)
                    # get aprimes (next actions to do)
                    moodyPrimemoves = self.iter_pick_nextmove(self.strategy, self.payoffs, self.partner_states, self.current_partner_list)
                    for n in self.current_partner_list:
                        self.moody_pp_aprime[n] = moodyPrimemoves[n]

                    # update the Q for the CURRENT sprime

                    for i in self.partner_IDs:
                        s = self.moody_oldstates[i]           # the state I used to be in
                        # print("(line 810) my id is", self.ID, "my partners are", self.partner_IDs, "and my moves were", self.itermove_result)
                        a = self.itermove_result[i]           # the action I took
                        """ This part below is different. Our sprime is now the state we observe from our opponent, not just our 
                            payoff memory. """
                        sprime = sarsa_moody.observe_state(self.partner_latest_move[i], i, self.partner_moods[i],
                                                           self.statemode)       # the state I found myself in
                        reward = self.moody_pp_payoff[i]      # the reward I observed
                        # print(self.ID, " is my ID and my aprimes are", self.moody_pp_aprime, "my partners are", self.partner_IDs)
                        aprime = self.moody_pp_aprime[i]      # the action I will take next

                        # print('ostates=', self.oldstates)
                        # print('sstates=', self.itermove_result)
                        # print('sprimes=', self.working_memory)
                        if self.moody_delta == 1:
                            if self.model.moody_memoryPaired:
                                s = s[0]
                        oldQValues = self.moody_qtable[tuple(s)]  # THIS MIGHT BREAK BECAUSE OF TUPLES


                        # Get the Q value we want to change, based on our state and our action
                        # qToChange = self.moody_qtable[tuple(s)]

                        # if a == 'C':
                        #     idx = 0
                        # elif a == 'D':
                        #     idx = 1
                        #
                        # qToChange = qToChange[idx]

                        # get the updated Qs according to the function provided

                        stateMem = self.state_working_memory[tuple(sprime)]
                        if aprime == 'C':
                            stateActionMem = stateMem[0]
                        else:
                            stateActionMem = stateMem[1]
                        # So now we have the list (memory) of the payoffs for the (sprime, aprime) that we want to do next
                        # We want to send that to be used to estimate future rewards
                        # TODO: Important warning! The memories are initialised at 0, so averaging over them will produce 0
                        # as an estimated future reward (of course, unless we have explored there before


                        # updatedQone, updatedQtwo = sarsa_moody.update_q(self.stepCount, a, s, self.moody_qtable, reward, self.working_memory[i], self.mood)
                        updatedQone, updatedQtwo = sarsa_moody.update_q(self.stepCount, a, s, self.moody_qtable, reward,
                                                                        stateActionMem, self.mood, self.moody_alpha, self.moody_gamma)

                        # if self.moody_delta == 1:
                        #     if self.model.moody_memoryPaired:
                        #         sprime = sprime[0]  #TODO: WILL WE NEED 4 SPRIMES? - OR WAIT, ========= DO WE DO THIS WHOLE SECTION PER PARTNER?=====
                        # newQValues = self.moody_qtable[tuple(sprime)]  # THIS ISN'T RIGHT IS IT?  #TODO: SEE ONE TODO ABOVE -> THIS IS GONNA CHANGE FROM MOODY_QTABLE TO ID_QTABLE, POSSIBLY
                        # if aprime == 'C':
                        #     idxprime = 0
                        # elif aprime == 'D':
                        #     idxprime = 1
                        #
                        # Qsa = oldQValues[idx]
                        # Qsaprime = newQValues[idxprime]
                        #
                        # # update the Q value for the old state and old action
                        #
                        # newQsa = sarsa_moody.update_q(reward, self.moody_gamma, self.moody_alpha, Qsa, Qsaprime)
                        # print('My old Q for this partner was:', Qsa, 'and my new Q is:', newQsa)
                        # then put newQ in the Qtable[s] at index idx
                        # change = self.moody_qtable[tuple(s)]
                        # change[idx] = newQsa
                        self.moody_qtable[tuple(s)][0] = updatedQone
                        self.moody_qtable[tuple(s)][1] = updatedQtwo

                        """Update mood here at the end of each interaction WITHIN a ROUND. This means that initial interactions
                            in each round will influence subsequent interactions"""
                        myAv, oppAv, oppScore = self.averageScoreComparison(i, True, self.current_partner_list)
                        #TODO: ARE THE SCORES BELOW SCORES AGAINST EACH PARTNER, OR ARE THEY TOTAL SCORES?
                        # self.mood = sarsa_moody.update_mood(self.mood, self.score, myAv, oppScore, oppAv)

                        # if self.ID == 9:
                        #     print("It's turn ", self.stepCount)
                        #     print("My mood going into this was ", self.mood)
                        #     print("My values were ", reward, myAv, oppScore, oppAv)
                        self.mood, self.sensitivity_mod = sarsa_moody.update_mood_new(self.mood, reward, myAv, oppScore, oppAv, self.sensitive, self.sensitivity_mod)
                        # if self.ID == 9:
                            # print("My mood coming out of it was ", self.mood)


                    # for i in self.working_memory:
                    #     state = self.working_memory[i]
                    #     my_move = self.itermove_result[i]
                    #     reward = self.pp_payoff[i]  # the unmanipulated value
                    #
                    #     # for the states I'm in, update the relevant q value
                    #     # access the qtable value we already have
                    #     oldQValues = self.qtable[state]   # THIS MIGHT BREAK BECAUSE OF TUPLES
                    #     if my_move == 'C':
                    #         idx = 0
                    #     elif my_move == 'D':
                    #         idx = 1
                    #
                    #     next_state = self.pp_sprime[i]
                    #     nextQValues = self.qtable[next_state]
                    #     nextQValue = nextQValues[idx]  # this is wrong, just want
                    #
                    #     oldQValue = oldQValues[idx]
                    #     new_value = sarsa.update_q(reward, self.gamma, self.alpha, oldQValue)

                    # update epsilon
                    # self.moody_epsilon = sarsa_moody.decay_value(self.model.moody_epsilon, self.moody_epsilon, self.model.rounds, True, self.model.moody_epsilon_floor)
                    # self.moody_alpha = sarsa_moody.decay_value(self.model.moody_alpha, self.moody_alpha, self.model.rounds, True, self.model.moody_alpha_floor)
                    #TODO: I don't believe they decay any values?

                    # update s to be sprime
                    for i in self.partner_IDs:
                        # self.moody_oldstates[i] = self.working_memory[i]
                        self.moody_oldstates[i] = sarsa_moody.observe_state(self.partner_latest_move[i], i, self.partner_moods[i],
                                                                            self.statemode)

                        # Update how we feel
                        # TODO: update mood earlier, after each Q value update??
                        # myAv, oppAv, oppScore = self.averageScoreComparison(i)
                        # self.mood = sarsa_moody.update_mood_new(self.mood, self.score, myAv, oppScore, oppAv, False, 0)

                    if self.model.moody_export_q:
                        if self.stepCount == 1:
                            self.outputQtable(True)
                        elif self.stepCount == self.model.rounds - 1:
                            self.outputQtable(False)

                    self.outputData(False)
                    # =============== UPDATE YOUR PARTNERS AND WHO U WANT AS PARTNERS =========
                    # print("garf", self.model.updated_graphD)
                    self.current_partner_list = copy.deepcopy(self.model.updated_graphD[self.ID])
                    if self.model.checkTurn:      # todo: this needs to  be repeated on all the other checkturns
                        # =============== CHECK IF YOU ARE ELIGIBLE FOR RESTRUCTURING =============
                        connections = []
                        for i in self.model.agentsToRestructure:
                            if i[0] == self.ID:
                                # if we have a connection to evaluate,
                                connections.append(i)
                            else:
                                pass


                        # =============== CHECK IF RESTRUCTURE CONNECTION EXISTS AND WHAT TO DO W/ IT ==========
                        break_check = []
                        gain_check = []
                        # then, for each of these, we check if that connection exists, and what to do with it
                        if len(connections) > 0:
                            for i in connections:
                                if i[1] not in self.current_partner_list:
                                    gain_check.append(i[1])
                                else:
                                    if i[1] in self.current_partner_list:
                                        break_check.append(i[1])

                        # TODO: this only happens once, when actually it needs to happen for each connection in the restructuring list.
                        # TODO: partnerDecision needs to be a decision per partner, returning single removRequest over and over, which are each removed in that moment from the model

                        if self.model.complexRestructuring:
                            if len(break_check) > 0:
                                for partnerID in break_check:
                                    toBreak = True
                                    #print("3my partners are", self.current_partner_list)
                                    #print("3my wm is:", self.working_memory)
                                    #print("3my pp_oppP is:", self.moody_pp_oppPayoff[partnerID])
                                    request = rnf.partnerDecision(toBreak, self.partnerSelectionStrat, partnerID,
                                                                  self.ID, self.rejected_partner_list,
                                                                  self.working_memory[partnerID],
                                                                  self.indivAvPayoff[partnerID],  # using this instead of working memory as sometimes wm breaks
                                                                  self.oppAvPayoff[partnerID],
                                                                  #self.model.CC,
                                                                  2,
                                                                  #self.model.reputationBlackboard[self.ID],
                                                                  self.model.averageBetrayals,
                                                                  # this is the target reputation
                                                                  self.mood,
                                                                  50,
                                                                  #self.model.reputationBlackboard[partnerID],
                                                                  self.check_item(partnerID, "rep"),
                                                                  self.normalizedActorDegreeCentrality)
                                    # check if the request is valid (aka, not Null) - if it is, send it to the model
                                    #   if it isn't, ignore it
                                    # print("3request outcome was ", request)
                                    if request[1] != None:
                                        self.model.graph_removals.append(request)
                                        self.rejected_partner_list.append(partnerID)
                                        #print("My strat is", self.strategy, "and I just rejected", partnerID)
                                        self.current_partner_list.remove(partnerID)

                            if len(self.current_partner_list) < self.model.maximumPartners:
                                if len(gain_check) > 0:
                                    for partnerID in gain_check:
                                        toBreak = False
                                        # then do the partner decision line
                                        #print("4my partners are", self.current_partner_list)
                                        #print("4my wm is:", self.working_memory)
                                        #print("4my pp_oppP is:", 0)
                                        request = rnf.partnerDecision(toBreak, self.partnerSelectionStrat, partnerID,
                                                                      self.ID, self.rejected_partner_list,
                                                                      [0,0,0,0,0,0,0],
                                                                      0,
                                                                      0,
                                                                      #self.model.CC,
                                                                      2,
                                                                      #self.model.reputationBlackboard[self.ID],
                                                                      self.model.averageBetrayals,
                                                                      # this is the target reputation
                                                                      self.mood,
                                                                      50,
                                                                      #self.model.reputationBlackboard[partnerID],
                                                                      self.check_item(partnerID, "rep"),
                                                                      self.normalizedActorDegreeCentrality)
                                        # check if the request is valid (aka, not Null) - if it is, send it to the model
                                        #   if it isn't, ignore it
                                        #print("4request outcome was ", request)
                                        if request[1] != None:
                                            self.model.graph_additions.append(request)
                            # else:
                            #   todo: some kind of pruning behaviour

                        else:
                            removRequest, addRequest, removals, additions = rnf.basicPartnerDecision(self.current_partner_list, self.rejected_partner_list,
                                                                                                 self.potential_partner_list, self.all_possible_partners,
                                                                                                 0.4, self.ID)
                            if removRequest:
                                if removRequest[1] != None:
                                    self.model.graph_removals.append(removRequest)
                                    self.current_partner_list.remove(removals)
                                    self.rejected_partner_list.append(removals)
                                    #print("My strat is", self.strategy, "and I just rejected", partnerID)
                            if addRequest:
                                if addRequest[1] != None:
                                    self.model.graph_additions.append(addRequest)

                    # if additions not in self.current_partner_list:
                    #     self.current_partner_list.append(additions)
                    # if removals not in self.rejected_partner_list:
                    #     self.rejected_partner_list.append(removals)
                    # self.partner_IDs = copy.deepcopy(self.current_partner_list)

                    self.stepCount += 1

                    if round_payoffs is not None:
                        if self.printing:
                            print("I am agent", self.ID, ", and I have earned", round_payoffs, "this round")
                        self.score += round_payoffs
                        # print("My total overall score is:", self.score)
                        return

                else:

                    # self.move = self.next_move
                    self.check_partner(self.current_partner_list)  # Update Knowledge
                    round_payoffs = self.increment_score(self.payoffs)
                    if self.model.moody_opponents:
                        for i in self.partner_IDs:
                            myAv, oppAv, oppScore = self.averageScoreComparison(i, False, self.current_partner_list)
                            # TODO: ARE THE SCORES BELOW SCORES AGAINST EACH PARTNER, OR ARE THEY TOTAL SCORES?
                            self.mood, self.sensitivity_mod = sarsa_moody.update_mood_new(self.mood, self.score, myAv, oppScore, oppAv, self.sensitive, self.sensitivity_mod)

                    if self.last_round:
                        if self.strategy == 'VPP':
                            if self.model.kNN_training:
                                self.training_data = self.export_training_data()

                    if self.stepCount == self.model.rounds - 1:
                        if self.strategy == 'VPP':
                            if not self.model.kNN_training:
                                self.knn_decision(self.partner_IDs, self.per_partner_utility, self.per_partner_mcoops,
                                                  self.per_partner_tcoops, self.per_partner_mutc, self.default_ppds)
                                # Only use this if we are not training as presumably we will have no training data because we will be training
                    """ Because model outputting is below, we can add update values to the list before it *may get reset """
                    # self.compare_score()

                    self.outputData(False)
                    # =============== UPDATE YOUR PARTNERS AND WHO U WANT AS PARTNERS =========
                    self.current_partner_list = copy.deepcopy(self.model.updated_graphD[self.ID])
                    if self.model.checkTurn:      # todo: this needs to  be repeated on all the other checkturns
                        # =============== CHECK IF YOU ARE ELIGIBLE FOR RESTRUCTURING =============
                        connections = []
                        for i in self.model.agentsToRestructure:
                            if i[0] == self.ID:
                                # if we have a connection to evaluate,
                                connections.append(i)
                            else:
                                pass


                        # =============== CHECK IF RESTRUCTURE CONNECTION EXISTS AND WHAT TO DO W/ IT ==========
                        break_check = []
                        gain_check = []
                        # then, for each of these, we check if that connection exists, and what to do with it
                        if len(connections) > 0:
                            for i in connections:
                                if i[1] not in self.current_partner_list:
                                    gain_check.append(i[1])
                                else:
                                    if i[1] in self.current_partner_list:
                                        break_check.append(i[1])

                        # TODO: this only happens once, when actually it needs to happen for each connection in the restructuring list.
                        # TODO: partnerDecision needs to be a decision per partner, returning single removRequest over and over, which are each removed in that moment from the model

                        if self.model.complexRestructuring:
                            if len(break_check) > 0:
                                for partnerID in break_check:
                                    toBreak = True
                                    #print("5my partners are", self.current_partner_list)
                                    #print("5my wm is:", self.working_memory)
                                    request = rnf.partnerDecision(toBreak, self.partnerSelectionStrat, partnerID,
                                                                  self.ID, self.rejected_partner_list,
                                                                  self.working_memory[partnerID],
                                                                  self.indivAvPayoff[partnerID],
                                                                  self.oppAvPayoff[partnerID],
                                                                  #self.model.CC,
                                                                  2,
                                                                  self.model.averageBetrayals,
                                                                  # this is the target reputation
                                                                  #self.model.reputationBlackboard[self.ID],
                                                                  self.mood,
                                                                  50,
                                                                  #self.model.reputationBlackboard[partnerID],
                                                                  self.check_item(partnerID, "rep"),
                                                                  self.normalizedActorDegreeCentrality)
                                    # check if the request is valid (aka, not Null) - if it is, send it to the model
                                    #   if it isn't, ignore it
                                    #print("5request outcome was ", request)
                                    if request[1] != None:
                                        self.model.graph_removals.append(request)
                                        self.rejected_partner_list.append(partnerID)
                                        #print("My strat is", self.strategy, "and I just rejected", partnerID)
                                        self.current_partner_list.remove(partnerID)

                            if len(self.current_partner_list) < self.model.maximumPartners:
                                if len(gain_check) > 0:
                                    for partnerID in gain_check:
                                        toBreak = False
                                        # then do the partner decision line
                                        #print("6my partners are", self.current_partner_list)
                                        #print("6my wm is:", self.working_memory)
                                        request = rnf.partnerDecision(toBreak, self.partnerSelectionStrat, partnerID,
                                                                      self.ID, self.rejected_partner_list,
                                                                      [0, 0, 0, 0, 0, 0, 0],
                                                                      0,
                                                                      0,
                                                                      #self.model.CC,
                                                                      2,
                                                                      self.model.averageBetrayals,
                                                                      # this is the target reputation
                                                                      #self.model.reputationBlackboard[self.ID],
                                                                      self.mood,
                                                                      50,
                                                                      #self.model.reputationBlackboard[partnerID],
                                                                      self.check_item(partnerID, "rep"),
                                                                      self.normalizedActorDegreeCentrality)
                                        # check if the request is valid (aka, not Null) - if it is, send it to the model
                                        #   if it isn't, ignore it
                                        #print("6request outcome was ", request)
                                        if request[1] != None:
                                            self.model.graph_additions.append(request)
                            # else:
                            #   todo: some kind of pruning behaviour

                        else:
                            removRequest, addRequest, removals, additions = rnf.basicPartnerDecision(self.current_partner_list, self.rejected_partner_list,
                                                                                                 self.potential_partner_list, self.all_possible_partners,
                                                                                                 0.4, self.ID)
                            if removRequest:
                                if removRequest[1] != None:
                                    self.model.graph_removals.append(removRequest)
                                    self.current_partner_list.remove(removals)
                                    self.rejected_partner_list.append(removals)
                                    #print("My strat is", self.strategy, "and I just rejected", partnerID)
                            if addRequest:
                                if addRequest[1] != None:
                                    self.model.graph_additions.append(addRequest)

                    # if additions not in self.current_partner_list:
                    #     self.current_partner_list.append(additions)
                    # if removals not in self.rejected_partner_list:
                    #     self.rejected_partner_list.append(removals)
                    # self.partner_IDs = copy.deepcopy(self.current_partner_list)

                    self.stepCount += 1

                    if round_payoffs is not None:
                        if self.printing:
                            print("I am agent", self.ID, ", and I have earned", round_payoffs, "this round")
                        self.score += round_payoffs
                        # print("My total overall score is:", self.score)

                        return
            else:
                # Find a new partner?
                self.outputData(False)
                self.current_partner_list = copy.deepcopy(self.model.updated_graphD[self.ID])
                if self.model.checkTurn:  # todo: this needs to  be repeated on all the other checkturns
                    # =============== CHECK IF YOU ARE ELIGIBLE FOR RESTRUCTURING =============
                    connections = []
                    for i in self.model.agentsToRestructure:
                        if i[0] == self.ID:
                            # if we have a connection to evaluate,
                            connections.append(i)
                        else:
                            pass

                    # =============== CHECK IF RESTRUCTURE CONNECTION EXISTS AND WHAT TO DO W/ IT ==========
                    break_check = []
                    gain_check = []
                    # then, for each of these, we check if that connection exists, and what to do with it
                    if len(connections) > 0:
                        for i in connections:
                            if i[1] not in self.current_partner_list:
                                gain_check.append(i[1])
                            else:
                                if i[1] in self.current_partner_list:
                                    break_check.append(i[1])

                    # TODO: this only happens once, when actually it needs to happen for each connection in the restructuring list.
                    # TODO: partnerDecision needs to be a decision per partner, returning single removRequest over and over, which are each removed in that moment from the model

                    if self.model.complexRestructuring:
                        if len(break_check) > 0:
                            for partnerID in break_check:
                                toBreak = True
                                # print("7my partners are", self.current_partner_list)
                                # print("7my wm is:", self.working_memory)
                                request = rnf.partnerDecision(toBreak, self.partnerSelectionStrat, partnerID,
                                                              self.ID, self.rejected_partner_list,
                                                              self.working_memory[partnerID],
                                                              statistics.mean(self.working_memory[partnerID]),
                                                              statistics.mean(self.pp_oppPayoff[partnerID]),
                                                              #self.model.CC,
                                                              2,
                                                              #self.model.reputationBlackboard[self.ID],
                                                              self.model.averageBetrayals,
                                                              # this is the target reputation
                                                              self.mood,
                                                              50,
                                                              #self.model.reputationBlackboard[partnerID],
                                                              self.check_item(partnerID, "rep"),
                                                              self.normalizedActorDegreeCentrality)
                                # check if the request is valid (aka, not Null) - if it is, send it to the model
                                #   if it isn't, ignore it
                                # print("7request outcome was ", request)
                                if request[1] != None:
                                    self.model.graph_removals.append(request)
                                    self.rejected_partner_list.append(partnerID)
                                    #print("My strat is", self.strategy, "and I just rejected", partnerID)
                                    self.current_partner_list.remove(partnerID)

                        if len(self.current_partner_list) < self.model.maximumPartners:
                            if len(gain_check) > 0:
                                for partnerID in gain_check:
                                    toBreak = False
                                    # then do the partner decision line
                                    # print("8my partners are", self.current_partner_list)
                                    # print("8my wm is:", self.working_memory)
                                    request = rnf.partnerDecision(toBreak, self.partnerSelectionStrat, partnerID,
                                                                  self.ID, self.rejected_partner_list,
                                                                  [0, 0, 0, 0, 0, 0, 0],
                                                                  0,
                                                                  0,
                                                                  #self.model.CC,
                                                                  2,
                                                                  #self.model.reputationBlackboard[self.ID],
                                                                  self.model.averageBetrayals,
                                                                  # this is the target reputation
                                                                  self.mood,
                                                                  50,
                                                                  #self.model.reputationBlackboard[partnerID],
                                                                  self.check_item(partnerID, "rep"),
                                                                  self.normalizedActorDegreeCentrality)
                                    # check if the request is valid (aka, not Null) - if it is, send it to the model
                                    #   if it isn't, ignore it
                                    # print("8request outcome was ", request)
                                    if request[1] != None:
                                        self.model.graph_additions.append(request)
                        # else:
                        #   todo: some kind of pruning behaviour

                    else:
                        removRequest, addRequest, removals, additions = rnf.basicPartnerDecision(
                            self.current_partner_list, self.rejected_partner_list,
                            self.potential_partner_list, self.all_possible_partners,
                            0.4, self.ID)
                        if removRequest:
                            if removRequest[1] != None:
                                self.model.graph_removals.append(removRequest)
                                self.current_partner_list.remove(removals)
                                self.rejected_partner_list.append(removals)
                                #print("My strat is", self.strategy, "and I just rejected", partnerID)
                        if addRequest:
                            if addRequest[1] != None:
                                self.model.graph_additions.append(addRequest)

                # if additions not in self.current_partner_list:
                #     self.current_partner_list.append(additions)
                # if removals not in self.rejected_partner_list:
                #     self.rejected_partner_list.append(removals)
                # self.partner_IDs = copy.deepcopy(self.current_partner_list)
                self.stepCount += 1
                return
        else:
            if self.model.forgivenessTurn:
                self.rejected_partner_list = []
                self.indivAvPayoff = {}
                self.betrayals = 0
            self.similar_partners = 0   # TODO: DOES THIS NEED TO BE INDENTED?
            if self.partnerSelectionStrat == "DEFAULT":  # ===========================================================
                #print("This was a reset round, so all I did was update my partners, my selection strat was,", self.partnerSelectionStrat)
                self.outputData(True)  # TODO: need to find out where to put this so that it doesn't break
                # TODO: Or, make it output a null/zeroes for this turn because it's a changeover round/can we duplicate the last round's outputs
                self.current_partner_list = copy.deepcopy(self.model.updated_graphD[self.ID])
                self.partner_IDs = copy.deepcopy(self.current_partner_list)
                self.stepCount += 1
                # self.check_partner(self.current_partner_list)
                return
            elif self.partnerSelectionStrat == "REP":  # ===========================================================
                #print("This was a reset round, so all I did was update my partners, my selection strat was,", self.partnerSelectionStrat)
                self.outputData(True)
                self.current_partner_list = copy.deepcopy(self.model.updated_graphD[self.ID])
                self.partner_IDs = copy.deepcopy(self.current_partner_list)
                self.stepCount += 1   # tekkers


                # self.check_partner(self.current_partner_list)
                return
            elif self.partnerSelectionStrat == "SCORE":  # ===========================================================
                #print("This was a reset round, so all I did was update my partners, my selection strat was,", self.partnerSelectionStrat)
                self.outputData(True)
                self.current_partner_list = copy.deepcopy(self.model.updated_graphD[self.ID])
                self.partner_IDs = copy.deepcopy(self.current_partner_list)
                self.stepCount += 1
                # self.check_partner(self.current_partner_list)
                return
            else:
                return


        #TODO: This may not work, as agents will not have some starting moves against new partners (aa primes and whatnot) but hopefully it does?

