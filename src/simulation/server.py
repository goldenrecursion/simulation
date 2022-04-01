from tokenize import triple_quoted
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.UserParam import UserSettableParameter

from simulation.agents import Triple, Agent
from simulation.model import AgentModel


total_token_metric_chart = ChartModule(
    [
        {"Label": "Total Tokens Earned", "Color": "#AAA000"},
        {"Label": "Total Staked Tokens", "Color": "#00AA00"},
    ]
)

average_token_metric_chart = ChartModule(
    [
        {"Label": "Average Tokens Earned", "Color": "#AA0000"},
        {"Label": "Average Staked Tokens", "Color": "#666666"},
    ]
)

triples_status_chart = ChartModule(
    [
        {"Label": "Total Accepted Triples", "Color": "#AA0000"},
        {"Label": "Total Rejected Triples", "Color": "#AAA000"},
        {"Label": "Total Pending Triples", "Color": "#666666"},
    ]
)

sim_configs = {
    # Percentage of triples that have a grounds truth validated (it's a canon triple)
    "gt_validation_ratio": UserSettableParameter(
        "slider", "Grounds Truth Validation Ratio", 0.8, 0.1, 1.0, 0.01
    ),
    # Value of triple
    "triple_validation_reward": UserSettableParameter(
        "slider", "Triple Validation Reward", 1, 0.01, 10, 0.01
    ), 
    "triple_validation_slash": UserSettableParameter(
        "slider", "Triple Validation Slash", 1, 0.01, 10, 0.01
    ),  
    "triple_submission_reward": UserSettableParameter(
        "slider", "Triple Submission Slash", 5, 0.01, 10, 0.01
    ),  
    "triple_submission_slash": UserSettableParameter(
        "slider", "Triple Submission Slash", 1, 0.01, 10, 0.01
    ),  


    # Agent State variables
    # Number of agents, these agents will be submitting and validating triples
    "number_of_agents": UserSettableParameter(
        "slider", "Number of Agents", 10, 1, 10000 
    ),  
    # Initial stake
    "initial_token_amount": UserSettableParameter(
        "slider", "Initial Token Amount", 100, 1, 1000 
    ),  
    # Initial reputation
    "initial_reputation": UserSettableParameter(
        "slider", "Initial Repuation", 0, 0, 10000 
    ),  
    # Validation rate per second
    "validation_rate": UserSettableParameter(
        "slider", "Validation Rate (Seconds)", 60, 1, 10000 
    ),  
    # Submission rate per second
    "submission_rate": UserSettableParameter(
        "slider", "Submission Rate (Seconds)", 180, 1, 10000 
    ),  

    # Sim Model variables
    #"time_frame": UserSettableParameter( 
    #    "slider", "Time Frame (Seconds)", 60*60*8, 1, 100000
    #),  
}

server = ModularServer(
    AgentModel,
    [
        total_token_metric_chart,
        average_token_metric_chart,
        triples_status_chart,
    ],
    "Golden Web3 Agent Simulation",
    sim_configs,
)

server.port = 8521