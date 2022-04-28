import time
from queue import Queue
import random
from typing import List
import copy

from tqdm import tqdm
import pandas as pd
import numpy as np
from random import normalvariate, gauss
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
        triple_space_size,
        **kwargs,
    ):
        self.num_agents = number_of_agents
        self.schedule = RandomActivation(self)
        
        # Sim variables
        self.initial_token_amount = initial_token_amount
        self.staked_tokens = 0
        self.validation_rate = validation_rate
        self.submission_rate = submission_rate
        self.triple_submission_reward = triple_submission_reward
        self.triple_submission_slash = triple_submission_slash
        self.triple_validation_reward = triple_validation_reward
        self.triple_validation_slash = triple_validation_slash
        self.gt_validation_ratio = gt_validation_ratio
        self.triple_space_size = triple_space_size
        self.stalled_count = 0  # count how many agents did nothing at each step
        sigma = 0.2

        # Create and add agents to scheduler
        for i in range(self.num_agents):
            validation_ratio = gauss(self.gt_validation_ratio, sigma)
            if validation_ratio >= 1:
                validation_ratio = 1
            if validation_ratio <= 0:
                validation_ratio = 0
            a = Agent(
                agent_id=i,
                model=self,
                initial_token_amount=self.initial_token_amount,
                validation_rate=self.validation_rate,
                submission_rate=self.submission_rate,
                triple_submission_reward=self.triple_submission_reward,
                triple_submission_slash=self.triple_submission_slash,
                triple_validation_reward=self.triple_validation_reward,
                triple_validation_slash=self.triple_validation_slash,
                gt_validation_ratio=validation_ratio,  #self.gt_validation_ratio,
                triple_space_size=self.triple_space_size,
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

    def get_all_submitted_triple_ids(self):
        agents = self.schedule.agents
        ids = []
        for agent in agents:
            for k in agent.submitted_triples.keys():
                new_ids = [triple._id for triple in agent.submitted_triples.get(k)]
                ids.extend(new_ids)
        return set(ids)

    def get_all_triples(self):
        agents = self.schedule.agents
        triples = [list(agent.submitted_triples.values()) for agent in agents]
        triples = [item for sublist in triples for item in sublist]
        triples = [item for sublist in triples for item in sublist]
        return triples

    def analyze_triples(self):
        triples = self.get_all_triples()
        triples_df = pd.DataFrame([(t._id, t.gt_validated, t.status.name, len(t.accepts), len(t.rejects)) for t in triples], columns=['id', 'gt_validated', 'status', 'accepts', 'rejects'])
        triples_df['validations'] = triples_df['accepts'] + triples_df['rejects']
        return triples_df

    def step(self):
        if self.stalled_count == self.num_agents:
            print("stalled")
            return
        else:
            self.stalled_count = 0
        self.datacollector.collect(self)
        self.schedule.step()