def chunker(seq, size):
    # Remember to use type conversion on this function call to get the result you want, i.e. tuple(chunker(x, y))
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def get_memory_states(behaviours, statemode, n_agents):
    """ Get a list of all possible states given n behaviour options and
        r spaces in the agent's memory - Size: Number of Objects in That Memory  """
    permutations = []
    mood_values = ["LOW", "NEUTRAL", "HIGH"] #THIS MIGHT NEED A GENERATOR ALL ON ITS OWN
    agents = list(range(1, n_agents+1))

    if statemode == "stateless":
        """Where agents only know about what they observed on that previous round"""
        # permutations.append([0, ])
        for i1 in behaviours:
            permutations.append([i1])

    elif statemode == "agentstate":
        """Where agents the observed behaviour and which agent they are interacting with"""
        permutations.append([0, 0])
        for i1 in behaviours:
            for i2 in agents:
                permutations.append([i1, i2])

    elif statemode == "moodstate":
        """Where agents know the observed behaviour, which agent they are interacting with,
           and the mood level (low, neutral, high) of their opponent"""
        permutations.append([0, 0, 0])
        for i1 in behaviours:
            for i2 in agents:
                for i3 in mood_values:
                    permutations.append([i1, i2, i3])

    return permutations

