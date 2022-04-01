import time
from queue import Queue
import random
from typing import List
import copy

from tqdm import tqdm
import pandas as pd
import numpy as np
from random import normalvariate
import matplotlib.pyplot as plt

from mesa import Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector

from simulation.agents import Agent

# Data Collection
def compute_average_tokens_earned(model):
    agents_token_earned = [agent.tokens_earned for agent in model.schedule.agents]
    N = model.num_agents
    return sum(agents_token_earned)/N

def compute_average_staked_tokens(model):
    agents_staked_tokens = [agent.staked_tokens for agent in model.schedule.agents]
    N = model.num_agents
    return sum(agents_staked_tokens)/N

def compute_total_tokens_earned(model):
    agents_token_earned = [agent.tokens_earned for agent in model.schedule.agents]
    return sum(agents_token_earned)

def compute_total_staked_tokens(model):
    agents_staked_tokens = [agent.staked_tokens for agent in model.schedule.agents]
    return sum(agents_staked_tokens)

def compute_total_accepted_triples(model):
    total_accepted = [len(agent.submitted_triples["accepted"]) for agent in model.schedule.agents]
    return sum(total_accepted)

def compute_total_rejected_triples(model):
    total_rejected = [len(agent.submitted_triples["rejected"]) for agent in model.schedule.agents]
    return sum(total_rejected)

def compute_total_pending_triples(model):
    total_pending = [len(agent.submitted_triples["pending"]) for agent in model.schedule.agents]
    return sum(total_pending)



# Create Identical Agents for simplicity(will change them up later)
class AgentModel(Model):
    
    def __init__(
        self,
        number_of_agents,
        initial_token_amount,
        validation_rate,
        submission_rate,
        triple_submission_reward,
        triple_submission_slash,
        triple_validation_reward,
        triple_validation_slash,
        gt_validation_ratio,
        **kwargs,
    ):
        self.num_agents = number_of_agents
        self.schedule = RandomActivation(self)
        
        # Sim variables
        self.staked_tokens = initial_token_amount
        self.validation_rate = validation_rate
        self.submission_rate = submission_rate
        self.triple_submission_reward = triple_submission_reward
        self.triple_submission_slash = triple_submission_slash
        self.triple_validation_reward = triple_validation_reward
        self.triple_validation_slash = triple_validation_slash
        self.gt_validation_ratio = gt_validation_ratio
        
        # Create and add agents to scheduler
        for i in range(self.num_agents):
            a = Agent(
                agent_id=i,
                model=self,
                staked_tokens=self.staked_tokens, 
                validation_rate=self.validation_rate+i,
                submission_rate=self.submission_rate+i,
                triple_submission_reward=self.triple_submission_reward,
                triple_submission_slash=self.triple_submission_slash,
                triple_validation_reward=self.triple_validation_reward,
                triple_validation_slash=self.triple_validation_slash,
                gt_validation_ratio=self.gt_validation_ratio,
            )
            self.schedule.add(a)
        
        # Data collector
        model_reporters = {
            "Average Tokens Earned": compute_average_tokens_earned,
            "Total Tokens Earned": compute_total_tokens_earned,
            "Average Staked Tokens": compute_average_staked_tokens,
            "Total Staked Tokens": compute_total_staked_tokens,
            "Total Accepted Triples": compute_total_accepted_triples,
            "Total Rejected Triples": compute_total_rejected_triples,
            "Total Pending Triples": compute_total_pending_triples,
            }
        agent_reporters = {
            "Tokens Earned": "tokens_earned",
            "Staked Tokens": "staked_tokens",
            }
        self.datacollector = DataCollector(
            model_reporters=model_reporters,
            agent_reporters=agent_reporters,
        )

        # For batch run and visual server
        self.running = True

    
    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()