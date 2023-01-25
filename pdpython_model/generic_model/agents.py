from mesa import Agent

class Gagent(Agent):
    def __init__(self, pos, model, stepcount=0, score=0):
        super().__init__(pos, model)

        self.pos = pos
        self.stepCount = stepcount
        self.score = score

    def some_function(self):
        a = 2 + 2
        return a

    def step(self):
        print(self.some_function())