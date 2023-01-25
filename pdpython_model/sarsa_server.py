from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import CanvasGrid, ChartModule, TextElement
from mesa.visualization.UserParam import UserSettableParameter

from agents import PDAgent
from sarsa_model import PDModel

# Green
C_COLOR = "#007f7f"
# Red
D_COLOR = "#ce0e2d"
# Blue
MID_COLOR = "#ffc0cb"

# OR, COLORS FOR SCORE VISUALISATION:

HIGHEST_COL = "#083E33"

TOP_COL = "#0B5345"

HIGH_MID = "#117A65"

LOW_MID = "#16A085"

LOWEST_COL = "#8ACFC2"

REST_COL = "#E8F8F5"

score_vis = False

def gen_Model_Portrayal(agent):
    if agent is None:
        return

    portrayal = {}

    if not score_vis:

        if type(agent) is PDAgent:
            portrayal = {"Shape": "circle",
                         "scale": 1,
                         # "Color": "white",
                         "Filled": "true",
                         "Layer": 1,
                         "r": 0.75,
                         "text": [agent.strategy, agent.score],
                         #"text_color": "black",
                         }

            color = MID_COLOR
            text_color = "black"

            # set agent color based on cooperation level
            if agent.common_move == ["C"]:
                color = C_COLOR
            if agent.common_move == ["D"]:
                color = D_COLOR
            if agent.common_move == ["Eq"]:
                color = MID_COLOR

            portrayal["Color"] = color
            portrayal["text_color"] = text_color

    elif score_vis:

        if type(agent) is PDAgent:
            portrayal = {"Shape": "circle",
                         "scale": 1,
                         # "Color": "white",
                         "Filled": "true",
                         "Layer": 1,
                         "r": 0.75,
                         "text": [agent.strategy, agent.score],
                         #"text_color": "black",
                         }

            color = REST_COL
            text_color = "black"

            # set agent color based on proportional score
            if agent.proportional_score == 100:
                color = HIGHEST_COL
                text_color = "white"
            if agent.proportional_score <= 97 < 100:
                color = HIGHEST_COL
                text_color = "black"
            if agent.proportional_score <= 94 < 97:
                color = HIGH_MID
                text_color = "black"
            if agent.proportional_score <= 92 < 94:
                color = LOW_MID
                text_color = "black"
            if agent.proportional_score <= 90 < 92:
                color = LOWEST_COL
                text_color = "black"
            if agent.proportional_score <= 0 < 90:
                color = REST_COL
                text_color = "black"

            portrayal["Color"] = color
            portrayal["text_color"] = text_color


        # if agent.strategy is None:
        #     portrayal = {"Shape": "circle",
        #                  "scale": 1,
        #                  "Color": "white",
        #                  "Filled": "true",
        #                  "Layer": 1,
        #                  "r": 0.75,
        #                  "text": [agent.common_move, agent.score],
        #                  "text_color": "black",
        #                  "scale": 1
        #                  }
        # elif agent.strategy == "ANGEL":
        #     portrayal = {"Shape": "circle",
        #                  "scale": 1,
        #                  "Color": "#ffe700",
        #                  "Filled": "true",
        #                  "Layer": 1,
        #                  "r": 0.75,
        #                  "text": [agent.common_move, agent.score],
        #                  "text_color": "#f08619",
        #                  "scale": 1
        #                  }
        # elif agent.strategy == "DEVIL":
        #     portrayal = {"Shape": "circle",
        #                  "scale": 1,
        #                  "Color": "#d52719",
        #                  "Filled": "true",
        #                  "Layer": 1,
        #                  "r": 0.75,
        #                  "text": [agent.common_move, agent.score],
        #                  "text_color": "#3e0400",
        #                  "scale": 1
        #                  }
        # elif agent.strategy == "EV":
        #     portrayal = {"Shape": "circle",
        #                  "scale": 1,
        #                  "Color": "#84f2cf",
        #                  "Filled": "true",
        #                  "Layer": 1,
        #                  "r": 0.75,
        #                  "text": [agent.common_move, agent.score],
        #                  "text_color": "#09806b",
        #                  "scale": 1
        #                  }
        # elif agent.strategy == "RANDOM":
        #     portrayal = {"Shape": "circle",
        #                  "scale": 1,
        #                  "Color": "grey",
        #                  "Filled": "true",
        #                  "Layer": 1,
        #                  "r": 0.75,
        #                  "text": [agent.common_move, agent.score],
        #                  "text_color": "white",
        #                  "scale": 1
        #                  }
        # elif agent.strategy == "VEV":
        #     portrayal = {"Shape": "circle",
        #                  "scale": 1,
        #                  "Color": "#008080",
        #                  "Filled": "true",
        #                  "Layer": 1,
        #                  "r": 0.75,
        #                  "text": [agent.common_move, agent.score],
        #                  "text_color": "#84f2cf",
        #                  "scale": 1
        #                  }
        # elif agent.strategy == "TFT":
        #     portrayal = {"Shape": "circle",
        #                  "scale": 1,
        #                  "Color": "#ffd0ef",
        #                  "Filled": "true",
        #                  "Layer": 1,
        #                  "r": 0.75,
        #                  "text": [agent.common_move, agent.score],
        #                  "text_color": "#f57694",
        #                  "scale": 1
        #                  }
        # elif agent.strategy == "WSLS":
        #     portrayal = {"Shape": "circle",
        #                  "scale": 1,
        #                  "Color": "#add8e6",
        #                  "Filled": "true",
        #                  "Layer": 1,
        #                  "r": 0.75,
        #                  "text": [agent.common_move, agent.score],
        #                  "text_color": "blue",
        #                  "scale": 1
        #                  }
        # elif agent.strategy == "VPP":
        #     portrayal = {"Shape": "circle",
        #                  "scale": 1,
        #                  "Color": "#003333",
        #                  "Filled": "true",
        #                  "Layer": 1,
        #                  "r": 0.75,
        #                  "text": [agent.common_move, agent.score],
        #                  "text_color": "#99cccc",
        #                  "scale": 1
        #                  }

    return portrayal


class StepCountDisplay(TextElement):

    def render(self, model):
        return "Step Count: " + str(model.step_count), "  --- K: " + str(model.coop_index)


canvas_element = CanvasGrid(gen_Model_Portrayal, 8, 8, 500, 500)
step_element = StepCountDisplay()
# chart_element = ChartModule([{"Label": "Walkers", "Color": "#AA0000"},
#                              {"Label": "Closed Boxes", "Color": "#666666"}])

model_params = {#"number_of_agents": UserSettableParameter('slider', 'Number of Agents', 25, 2, 64, 1),
                "number_of_agents": UserSettableParameter('choice', 'Number of Agents', value=16,
                                          choices=[2, 4, 9, 16, 25, 36, 49, 64]),
                "sarsa_oppo": UserSettableParameter('choice', 'Opponent Type', value='TFT',
                                          choices=['MOODYLEARN', 'LEARN', 'TFT', 'VPP', 'ANGEL', 'DEVIL', 'RANDOM', 'WSLS', 'MIXED']),
                #"startingBehav": UserSettableParameter('choice', 'First Round Move', value='C',
                 #                         choices=['C', 'D']),
                #"moody_statemode": UserSettableParameter('choice', 'State Information', value='stateless',
                #                          choices=['stateless', 'agentstate', 'moodstate']),
                "rounds": UserSettableParameter('slider', 'Number of Rounds', 5000,1,100000,10),
                "collect_data": UserSettableParameter('checkbox', 'Collect Data', False),
                #"init_ppD": UserSettableParameter('slider', 'Initial Probability VPP Agents Defect', 0.50,0.01,1,0.01),
                # "agent_printing": UserSettableParameter('checkbox', 'Agent Printouts', False),
                "CC": UserSettableParameter('number', 'Payoff for CC (Default: 3)', value=3),
                "CD": UserSettableParameter('number', 'Payoff for CD (Default: 0)', value=0),
                "DC": UserSettableParameter('number', 'Payoff for DC (Default: 5)', value=5),
                "DD": UserSettableParameter('number', 'Payoff for DD (Default: 2)', value=2),
                "epsilon": UserSettableParameter('number', 'Starting Epsilon (Default: 0.9)', value=0.9),
                "sarsa_distro": UserSettableParameter('slider', '% of SARSA Agents (0 = Checkerboard)', 0,0,1,0.1),

                "memoryPaired": UserSettableParameter('checkbox', 'Use memory pairs? (Sets Below Option to "Them" and Memory Limit to 4)', False),
                "msize": UserSettableParameter('choice', 'Memory Size', value=1,
                                          choices=[1, 2, 3, 4, 5, 6, 7]),
                "learnFrom": UserSettableParameter('choice', 'Who do we learn from?', value='me',
                                          choices=['me', 'them', 'us']),


                # "simplified_payoffs": UserSettableParameter('checkbox', 'Simplified Payoffs', False),
                # "b": UserSettableParameter('number', 'Simplified Payoffs: Benefit of Co-op', value=4),
                # "c": UserSettableParameter('number', 'Simplified Payoffs: Cost of Co-op', value=1),

                }

# chart_element = ChartModule([{"Label": "Cooperations", "Color": C_COLOR},
#                              {"Label": "Defections", "Color": D_COLOR}])
# TODO: Kind of want to add in mutual cooperations tracking, but that's extraneous right now

chart_element = ChartModule([{"Label": "Percentage Cooperations", "Color": C_COLOR},
                             {"Label": "Percentage Defections", "Color": D_COLOR},
                             {"Label": "Average Mood", "Color": "#ffc0cb"}])

server = ModularServer(PDModel, [canvas_element, step_element, chart_element], "Prisoner's Dilemma Simulation", model_params)
server.port = 8521
