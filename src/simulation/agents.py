import time
from queue import Queue
from random import randint, choice
from typing import List
import copy

from tqdm import tqdm
import pandas as pd
import numpy as np
from random import normalvariate
import matplotlib.pyplot as plt

from mesa import Agent as MesaAgent
from mesa import Model
from mesa.time import RandomActivation

from enum import Enum
class TripleStatus(Enum):
    PENDING = 1
    ACCEPTED = 2
    REJECTED = 3

class Triple:
    
    def __init__(self,
                 source_entity: int,
                 submitter,
                 gt_validated: bool = False,
                 triple_submission_reward: float = 5,
                 triple_submission_slash: float = 1,
                 triple_validation_reward: float = 1,
                 triple_validation_slash: float = 1,
                ):
        self.source_entity = source_entity
        self.submitter = submitter
        
        self.status = TripleStatus.PENDING
        
        self.accepts = []
        self.rejects = []
        
        self.gt_validated = gt_validated

        self.triple_submission_reward = triple_submission_reward
        self.triple_submission_slash = triple_submission_slash
        self.triple_validation_reward = triple_validation_reward
        self.triple_validation_slash = triple_validation_slash
                
    def __repr__(self):
        return f"""Triple ID: {self.source_entity}
        submitted by: {self.submitter.agent_id}
        num_accepts: {len(self.accepts)}
        num_rejects: {len(self.rejects)}
        """
    
    def accepted(self):
        total_votes = len(self.accepts) + len(self.rejects)
        accept_ratio = len(self.accepts)/(total_votes+1)
        p_thresh = .6  # accept_ratio threshold to validate
        q_thresh = 5  # minimum votes to validate

        return total_votes >= q_thresh and accept_ratio > p_thresh
    
    def rejected(self):
        total_votes = len(self.accepts) + len(self.rejects)
        accept_ratio = len(self.accepts)/(total_votes+1)
        p_thresh = .1  # accept_ratio threshold to validate
        q_thresh = 5 # minimum votes to validate

        return total_votes >= q_thresh and accept_ratio < p_thresh
    
    def add_reject(self, agent):
        assert self.status == TripleStatus.PENDING
        self.rejects.append(agent)
        if self.rejected():
            self.status = TripleStatus.REJECTED
            # Move in agent's submitted pool
            self.submitter.submitted_triples["pending"].remove(self)
            self.submitter.submitted_triples["rejected"].append(self)
            for agent in self.rejects:
                agent.tokens_earned += self.triple_validation_reward
                agent.rewarded_validations.append(self)
            for agent in self.accepts:
                agent.staked_tokens -= self.triple_validation_slash
                agent.slashed_validations.append(self)
            self.submitter.staked_tokens -= self.triple_submission_slash
        
    def add_accept(self, agent):
        assert self.status == TripleStatus.PENDING
        self.accepts.append(agent)
        if self.accepted():
            # Move in agent's submitted pool
            self.submitter.submitted_triples["pending"].remove(self)
            self.submitter.submitted_triples["accepted"].append(self)
            self.status = TripleStatus.ACCEPTED
            for agent in self.accepts:
                agent.tokens_earned += self.triple_validation_reward
                agent.rewarded_validations.append(self)
            for agent in self.rejects:
                agent.staked_tokens -= self.triple_validation_slash
                agent.slashed_validations.append(self)
            self.submitter.tokens_earned += self.triple_submission_reward
        

class Agent(MesaAgent):
    
    def __init__(self,
                 agent_id: int,
                 model,
                 staked_tokens = 100,
                 submission_rate: int = 90,
                 validation_rate: int = 60,
                 triple_submission_reward: float = 5,
                 triple_submission_slash: float = 1,
                 triple_validation_reward: float = 1,
                 triple_validation_slash: float = 1,
                 gt_validation_ratio: float = .9,
                ):
        super().__init__(agent_id, model)
        self.agent_id = agent_id 
        
        self.staked_tokens = staked_tokens
        self.tokens_earned = 0
        
        self.reputation = 0 # normalized between -1 and 1, 0 is neutral
        
        self.submission_rate = submission_rate
        self.validation_rate = validation_rate
                
        self.submitted_triples = {
            "pending": [],
            "accepted": [],
            "rejected": [],
        }
        
        # Might get memory intensive
        self.rewarded_validations = []
        self.slashed_validations = []
        
        self.triple_submission_reward = triple_submission_reward
        self.triple_submission_slash = triple_submission_slash
        self.triple_validation_reward = triple_validation_reward
        self.triple_validation_slash = triple_validation_slash
        self.gt_validation_ratio = gt_validation_ratio
        
    def __repr__(self):
        return f"""Agent: {self.agent_id}
        Has {self.staked_tokens} tokens staked.
        Has {self.tokens_earned} tokens earned.
        Reputation: {self.reputation}
        Pending Triples: {len(self.submitted_triples["pending"])}
        Accepted Triples: {len(self.submitted_triples["accepted"])}
        Rejected Triples: {len(self.submitted_triples["rejected"])}
        Rewarded Validations: {len(self.rewarded_validations)}
        Slashed Validations: {len(self.slashed_validations)}
        """
    
    def __hash__(self):
        return hash(self.agent_id)

    def submit_entity(self):
        for i in range(0, 5):
            self.submit_triple(1)
        
    def submit_triple(self, entity = 1):
        # create a triple in the graph submitted by this agent and adds to their submitted_triples
        self.submitted_triples["pending"].append(Triple(
            entity,
            self,
            gt_validated = bool(np.random.choice([0,1], p =[1-self.gt_validation_ratio, self.gt_validation_ratio])),
            triple_submission_reward = self.triple_submission_reward,
            triple_submission_slash = self.triple_submission_slash,
            triple_validation_reward = self.triple_validation_reward,
            triple_validation_slash = self.triple_validation_slash,
        ))
        
    def submitted_triples_available_to_validate(self):
        # returns: array of oldest X triples from submitted_triples queue not yet accepted or rejected where X is determined by remaining staked_tokens
        return [t for t in self.submitted_triples["pending"]][:self.staked_tokens]
    
    # Mesa Step
    def step(self):
        # Get current step count (Seconds)
        t_step = self.model.schedule.steps
        # Get agents pool
        agents = self.model.schedule.agents
        
        # Submit triple
        if t_step%self.submission_rate==0:
            self.submit_triple(randint(0,1000))

        # Validate triple
        if t_step%self.validation_rate==0:
            # Can't validate when out of staked tokens
            if self.staked_tokens <= 0:
                return

            # Agent begins validation by retrieving triple from "queue"
            triples = np.concatenate([t for t in [a.submitted_triples_available_to_validate() for a in agents]]).flat
            # Filter triples agent already voted on
            triples = list(filter(lambda x: (self not in x.accepts) and (self not in x.rejects), triples))
            # pick triple from triples
            triple = choice(triples) if triples else None

            if not triple:
                return

            # Agent will validate triple here
            if triple.gt_validated:
                agent_will_validate = bool(np.random.choice([0,1], p =[1-self.gt_validation_ratio, self.gt_validation_ratio]))  # Adding some randomness based on grounds truth validation ratio
                if agent_will_validate:
                    triple.add_accept(self)
                else:
                    triple.add_reject(self)
            else:
                agent_will_not_validate = bool(np.random.choice([0,1], p =[1-self.gt_validation_ratio, self.gt_validation_ratio]))  # Adding some randomness based on grounds truth validation ratio
                if agent_will_not_validate:
                    triple.add_reject(self)
                else:
                    triple.add_accept(self)

            # Show Agent State
            #agents_staked_overtime[t] = pd.DataFrame([agent.__dict__ for agent in agents_state]).staked_tokens
            #agents_earned_overtime[t] = pd.DataFrame([agent.__dict__ for agent in agents_state]).tokens_earned