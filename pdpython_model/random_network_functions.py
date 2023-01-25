import random
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def init_partner_list(height, width, id):
    """ Set ID to 0 to generate the full list of all agent IDs."""
    pardners = []
    maxm = (height*width)+1
    ids = list(range(1, maxm))
    for i in ids:
        if not i == id:
            pardners.append(i)
    return pardners


def init_pot_partner_reps(potential_partner_list, rep_type):
    return


def initRandomGraph(nodes, probability, ids):
    """ Produces a Random Erdos Renyi Graph, based on a probability of connecting to each next node, and a dict for
    each agent of who their starting partners should be. """
    N = nodes
    P = probability
    g = nx.Graph()

    edgesDict = {}

    for i in ids:
        edgesDict[i] = 0

    # Adding nodes
    g.add_nodes_from(range(1, N + 1))

    # Add edges to the graph randomly.
    for i in g.nodes():
        for j in g.nodes():
            if (i < j):

                # Take random number R.
                R = random.random()

                # Check if R<P add the edge to the graph else ignore.
                if (R < P):
                    g.add_edge(i, j)
        pos = nx.circular_layout(g)

    # for k in ids:
    #     # Get each's nodes edges into one big nested list
    #     bigList[k] = g.edges(k)

    for k in ids:
        current_edges = list(g.edges(k))
        exported_edges = []
        length = range(len(current_edges))
        for i in length:
            current_edge = current_edges[i]
            exported_edges.append(current_edge[1])
        edgesDict[k] = exported_edges

    # Export the Edge List of the graph, and Export the version Agents can understand
    return edgesDict, g


def maxEdgesPossible(nodes, ids):
    """ Produces a Random Erdos Renyi Graph, based on a probability of connecting to each next node, and a dict for
    each agent of who their starting partners should be. """
    N = nodes
    P = 1
    g = nx.Graph()

    edgesDict = {}

    for i in ids:
        edgesDict[i] = 0

    # Adding nodes
    g.add_nodes_from(range(1, N + 1))

    # Add edges to the graph randomly.
    for i in g.nodes():
        for j in g.nodes():
            if (i < j):

                # Take random number R.
                R = random.random()

                # Check if R<P add the edge to the graph else ignore.
                if (R < P):
                    g.add_edge(i, j)
        pos = nx.circular_layout(g)

    # for k in ids:
    #     # Get each's nodes edges into one big nested list
    #     bigList[k] = g.edges(k)

    for k in ids:
        current_edges = list(g.edges(k))
        exported_edges = []
        length = range(len(current_edges))
        for i in length:
            current_edge = current_edges[i]
            exported_edges.append(current_edge[1])
        edgesDict[k] = exported_edges

    # Export the Edge List of the graph, and Export the version Agents can understand
    return g.number_of_edges()


def convertGraph_to_dictEdges(graph, ids):
    """ Output the graph instead as an ID-keyed dict that contains vectors of each agent's initial partners. """
    edgesDict = {}

    for i in ids:
        edgesDict[i] = 0

    g = graph

    for k in ids:
        current_edges = list(g.edges(k))
        exported_edges = []
        length = range(len(current_edges))
        for i in length:
            current_edge = current_edges[i]
            exported_edges.append(current_edge[1])
        edgesDict[k] = exported_edges

    return edgesDict


def convertDict_to_graph(dict, ids):
    """ Take in a dict, construct a graph from the edge lists for each node."""
    G = nx.Graph()
    G.add_nodes_from(ids)
    edge_counter = 0
    for i in ids:
        node = i
        nodesEdges = dict[i]
        for j in nodesEdges:
            edge_counter +=1
            pair = (i, j)
            if not G.has_edge(*pair):
                G.add_edge(*pair)

    # print(edge_counter/2)
    # print(G.number_of_edges())
    return G


def update_graph(old_edges, additions, removals, sorting, ids):
    """ Import the previous graph, get its data in dict form, get the current edge list and compare changes at regular intervals."""
    changeable = convertDict_to_graph(old_edges, ids)   # Create a graph from the old dict the agents were reading from
    for pair in additions:     # Additions should be a list of tuples that wish to have edge connections
        if not changeable.has_edge(*pair):
            changeable.add_edge(*pair)

    for pair in removals:
        if changeable.has_edge(*pair):
            changeable.remove_edge(*pair)

    # Covert the new graph to something the agents can read
    changesDict = convertGraph_to_dictEdges(changeable, ids)
    if sorting:
        for i in ids:               # Sort the lists for each agent - this is for readability for us when we export it, but I don't know if it will affect agent behaviour, so am making it optional for now
            list = changesDict[i]
            sortedList = sorted(list)
            changesDict[i] = sortedList
            # print("the changes were", changesDict)
        # Then output both the graph translation and the graph itself
        return changesDict, changeable
    else:
        # Then output both the graph translation and the graph itself
        # print("the changes were", changesDict)
        print(type(changeable))
        print("Density:", nx.density(changeable), " Nodes:", nx.number_of_nodes(changeable), " Edges:", nx.number_of_edges(changeable))
        return changesDict, changeable


def basicPartnerDecision(current_partners, rejected_partners, untested_partners,
                         possible_partners, new_partnerProb, my_ID):
    """ Takes in a list of current partners, drops one and requests a new one from untested list. """
    removal_request = []
    addition_request = []

    removal = 0
    addition = 0

    # if current_partners:  # This has to be the case otherwise it will start appending None
    # TODO: Currently the agent-side code for this actively accepts None, so we'll see if that works
    if len(current_partners) > 0:
        removal = random.choice(current_partners)
        removal_request.append(my_ID)
        removal_request.append(removal)

    addition = chooseNewPartner(untested_partners, rejected_partners, possible_partners, new_partnerProb, [0,0])
    addition_request.append(my_ID)
    addition_request.append(addition)

    return tuple(removal_request), tuple(addition_request), removal, addition

def scoreCheck(my_av, their_av, threshold):
    breakCheck = False
    # Currently this prioritises getting a score against someone of 3 or higher
    if my_av < threshold:
        breakCheck = True
    return breakCheck

def partnerDecision(breakCheck, selectionStrategy, partnerID, myID, rejectedPartners, partnerHistory,
                    scoreAgainst, partnerScore, scoreThreshold, targetRep, mood, moodThreshold,
                    partnerRep, myConnectedness):
    # May want things like partner mood, current_mood,
    """ Initially this function will , but later will be designed to make decisions based on information and rules as to
        which new partners they desire. TODO: Do highly connected agents have a lower chance to connect? """
    request = []
    #removal = 0
    #addition = 0
    request.append(myID)
    request.append(None)

    if selectionStrategy == "DEFAULT":
        # the default is to make these decisions at random, so we will gain and lose them randomly
        if breakCheck:
            r = random.random()
            if r > 0.5:
                # request.append(myID)
                request[1] = partnerID
                #removal = partnerID
        else:
            if partnerID not in rejectedPartners:
                r = random.random()
                if r > 0.5:
                    # request.append(myID)
                    request[1] = partnerID
                    #addition = partnerID

    elif selectionStrategy == "SCORE":
        if breakCheck:
            if scoreCheck(scoreAgainst, partnerScore, scoreThreshold):
                # request.append(myID)
                request[1] = partnerID
        else:  # if we are gaining them as a partner
            if partnerID not in rejectedPartners:
                # request.append(myID)
                request[1] = partnerID

    elif selectionStrategy == "REP":
        if breakCheck:
            if partnerRep >= targetRep:
                request[1] = partnerID
        else:
            if partnerID not in rejectedPartners:
                if partnerRep < targetRep:
                    request[1] = partnerID
    else:
        # this is just the default strategy, as a backup in case something goes wrong
        if breakCheck:
            r = random.random()
            if r > 0.5:
                # request.append(myID)
                request[1] = partnerID
                # removal = partnerID
        else:
            if partnerID not in rejectedPartners:
                r = random.random()
                if r > 0.5:
                    # request.append(myID)
                    request[1] = partnerID
                    # addition = partnerID

    return tuple(request)  # , removal, addition


def chooseNewPartner(untested_partners, tested_partners, allPossPartners, probability, weights):
    """ For now, either select a partner with a random chance, or select from either untested or previously tested partners
     with a weighting system - presumably lower weights for previously rejected partners. """
    choice = None
    if probability:
        R = random.random()
        if R > probability:
            choice = random.choice(allPossPartners)
    else:
        partner_type = random.choices([untested_partners, tested_partners], weights)
        if partner_type == untested_partners:
            choice = random.choice(untested_partners)
        else:
            choice = random.choice(tested_partners)
    return choice


def getPossiblePartners(ids, current_partners):
    possible = []
    for i in ids:
        if i not in current_partners:
            possible.append(i)
    return possible

# The above should then be checked against each agent's personal memory of 'trustworthiness metrics' for these
# possible partners, if they have interacted before. The should have lists of rejected partners (for the other functions
# listed above) for use as well (added at the same time as they add to the dictionary)


def analysis(D, G):
    analy = ("Density:", nx.density(G), " Nodes:", nx.number_of_nodes(G), " Edges:", nx.number_of_edges(G),
             "nodes", nx.nodes(G), "info", nx.info(G))
    return(analy)


def allPossibleCombinations(n_agents):
    pair_list = []
    all_agents = [i for i in range(n_agents+1)]
    all_agents = all_agents[1:n_agents]  #  remove the initial zero? I don't think there's any 0 agents
    for i in all_agents:
        for j in all_agents:
            if i != j:
                pair = [i, j]
                pair_list.append(pair)
    return pair_list

def generateRewireList(n_agents, percentage_rewiring):
    # we should be okay just returning a list, as we only need to know the ID numbers of who is switching
    allCombinations = allPossibleCombinations(n_agents)
    total_count = len(allCombinations)
    to_select = round(total_count*percentage_rewiring)
    rewire_list = random.sample(allCombinations, to_select)
    rewire_list.sort()  # sort them into
    return rewire_list



