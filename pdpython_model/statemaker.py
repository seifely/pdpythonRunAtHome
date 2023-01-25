def chunker(seq, size):
    # Remember to use type conversion on this function call to get the result you want, i.e. tuple(chunker(x, y))
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def get_memory_states(behaviours, size, paired):
    """ Get a list of all possible states given n behaviour options and
        r spaces in the agent's memory - Size: Number of Objects in That Memory  """
    options = behaviours
    permutations = []
    if not paired:

        if size == 1:
            for i1 in options:
                permutations.append([i1])

        elif size == 2:
            for i1 in options:
                for i2 in options:
                    permutations.append([i1, i2])

        elif size == 3:
            for i1 in options:
                for i2 in options:
                    for i3 in options:
                        permutations.append([i1, i2, i3])

        elif size == 4:
            for i1 in options:
                for i2 in options:
                    for i3 in options:
                        for i4 in options:
                            permutations.append([i1, i2, i3, i4])

        elif size == 5:
            for i1 in options:
                for i2 in options:
                    for i3 in options:
                        for i4 in options:
                            for i5 in options:
                                permutations.append([i1, i2, i3, i4, i5])

        elif size == 6:
            for i1 in options:
                for i2 in options:
                    for i3 in options:
                        for i4 in options:
                            for i5 in options:
                                for i6 in options:
                                    permutations.append([i1, i2, i3, i4, i5, i6])

        elif size >= 7:
            for i1 in options:
                for i2 in options:
                    for i3 in options:
                        for i4 in options:
                            for i5 in options:
                                for i6 in options:
                                    for i7 in options:
                                        permutations.append([i1, i2, i3, i4, i5, i6, i7])

        # to generate the < step 7 states
        permutations.append([0, 0, 0, 0, 0, 0, 0])
        permutations.append([0, 0, 0, 0, 0, 0])
        permutations.append([0, 0, 0, 0, 0])
        permutations.append([0, 0, 0, 0])
        permutations.append([0, 0, 0])
        permutations.append([0, 0])
        permutations.append([0,])

        initial_state1 = [0, 0, 0, 0, 0, 0]
        initial_state2 = [0, 0, 0, 0, 0]
        initial_state3 = [0, 0, 0, 0]
        initial_state4 = [0, 0, 0]
        initial_state5 = [0, 0]
        initial_state6 = [0,]

        for ii1 in options:
            new = initial_state1 + [ii1]
            permutations.append(new)
        for ii2 in options:
            for iii2 in options:
                new = initial_state2 + [ii2] + [iii2]
                permutations.append(new)
        for ii3 in options:
            for iii3 in options:
                for iiii3 in options:
                    new = initial_state3 + [ii3] + [iii3] + [iiii3]
                    permutations.append(new)
        for ii4 in options:
            for iii4 in options:
                for iiii4 in options:
                    for iiiii4 in options:
                        new = initial_state4 + [ii4] + [iii4] + [iiii4] + [iiiii4]
                        permutations.append(new)
        for ii5 in options:
            for iii5 in options:
                for iiii5 in options:
                    for iiiii5 in options:
                        for iiiiii5 in options:
                            new = initial_state5 + [ii5] + [iii5] + [iiii5] + [iiiii5] + [iiiiii5]
                            permutations.append(new)
        for ii6 in options:
            for iii6 in options:
                for iiii6 in options:
                    for iiiii6 in options:
                        for iiiiii6 in options:
                            for iiiiiii6 in options:
                                new = initial_state6 + [ii6] + [iii6] + [iiii6] + [iiiii6] + [iiiiii6] + [iiiiiii6]
                                permutations.append(new)

    elif paired:
        # if size < 4:
        size = size * 2
        # Items in outcomes are doubled, as the states we need have pairs in them and can never have single values
        # i.e. (['C', 'C'], ['C', 'C'], ['C', 'C'], ['D']) is not allowed

        # next, we generate the same lists as previous, but we then chunk the outcomes into pairs
        # TODO: Are the states as they get recorded by agents 'my move, their move'?
        # TODO: Output of size == 2 paired states = (['C', 'D'],)   --> will this extra comma be a problem?
        # TODO: If so, you can access it by doing like, a = statelist[object index],   b = a[0]

        if size == 2:
            for i1 in options:
                for i2 in options:
                    permutations.append((i1, i2))

        if size == 4:
            for i1 in options:
                for i2 in options:
                    permutations.append((i1, i2))

            for i1 in options:
                for i2 in options:
                    for i3 in options:
                        for i4 in options:
                            permutations.append(((i1, i2), (i3, i4)))

        if size == 6:
            for i1 in options:
                for i2 in options:
                    permutations.append((i1, i2))

            for i1 in options:
                for i2 in options:
                    for i3 in options:
                        for i4 in options:
                            permutations.append(((i1, i2), (i3, i4)))

            for i1 in options:
                for i2 in options:
                    for i3 in options:
                        for i4 in options:
                            for i5 in options:
                                for i6 in options:
                                    permutations.append(((i1, i2), (i3, i4), (i5, i6)))

        if size >= 8:
            for i1 in options:
                for i2 in options:
                    permutations.append((i1, i2))

            for i1 in options:
                for i2 in options:
                    for i3 in options:
                        for i4 in options:
                            permutations.append(((i1, i2), (i3, i4)))

            for i1 in options:
                for i2 in options:
                    for i3 in options:
                        for i4 in options:
                            for i5 in options:
                                for i6 in options:
                                    permutations.append(((i1, i2), (i3, i4), (i5, i6)))

            for i1 in options:
                for i2 in options:
                    for i3 in options:
                        for i4 in options:
                            for i5 in options:
                                for i6 in options:
                                    for i7 in options:
                                        for i8 in options:
                                            permutations.append(((i1, i2), (i3, i4), (i5, i6), (i7, i8)))


        permutations.append([(0, 0), (0, 0), (0, 0), (0, 0)])
        permutations.append([(0,0),])
        permutations.append([0,0])
        permutations.append([(0,0), (0,0)])
        permutations.append([(0,0), (0,0), (0,0)])
        # print(permutations)
        # for i in permutations:
        #     indx = permutations.index(i)
        #     permutations[indx] = tuple(chunker(i, 2))

        # then we need to add in initial states
        initial_state1 = [(0,0), (0,0), (0,0)]
        initial_state2 = [(0,0), (0,0)]
        initial_state3 = [(0,0)]

        permu = []
        for i1 in options:
            for i2 in options:
                permu.append((i1, i2))

        for ii1 in permu:
            new = initial_state1 + [ii1]
            permutations.append(tuple(new))

        for ii2 in permu:
            for iii2 in permu:
                new = initial_state2 + [ii2] + [iii2]
                permutations.append(tuple(new))

        for ii3 in permu:
            for iii3 in permu:
                for iiii3 in permu:
                    new = initial_state3 + [ii3] + [iii3] + [iiii3]
                    permutations.append(tuple(new))

    return permutations

