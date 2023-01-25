from mesa import Model
from mesa.space import MultiGrid
from mesa.time import BaseScheduler, RandomActivation, SimultaneousActivation
from pdpython_model.generic_model.agents import Gagent

from mesa.datacollection import DataCollector

class GenericModel(Model):

    schedule_types = {"Sequential": BaseScheduler,
                     "Random": RandomActivation,
                     "Simultaneous": SimultaneousActivation}

    def __init__(self, height=5, width=5,
                 number_of_agents=1,
                 schedule_type="Random",):

        # Model Parameters
        self.height = height
        self.width = width
        self.number_of_agents = number_of_agents
        self.step_count = 0
        self.schedule_type = schedule_type

        # Model Functions
        self.schedule = self.schedule_types[self.schedule_type](self)
        self.grid = MultiGrid(self.height, self.width, torus=True)

        self.make_agents()
        self.running = True

    def make_agents(self):
        for i in range(self.number_of_agents):
            x, y = self.grid.find_empty()
            gagent = Gagent((x, y), self, True)
            self.grid.place_agent(gagent, (x, y))
            self.schedule.add(gagent)
            print("agent added")

    def step(self):
        self.schedule.step()
        self.step_count += 1

    def run_model(self, rounds=1):
        for i in range(rounds):
            self.step()

