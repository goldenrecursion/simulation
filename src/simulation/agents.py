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
                 _id: int,
                 submitter,
                 gt_validated: bool = False,
                 triple_submission_reward: float = 5,
                 triple_submission_slash: float = 1,
                 triple_validation_reward: float = 1,
                 triple_validation_slash: float = 1,
                ):
        self._id = _id
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
        return f"""Triple ID: {self._id}
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
        q_thresh = 5  # minimum votes to validate

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
                agent.tokens_earned += 2 * self.triple_validation_reward
                agent.wallet += 2 * self.triple_validation_reward
                agent.rewarded_validations.append(self)
            for agent in self.accepts:
                agent.staked_tokens -= self.triple_validation_slash
                agent.wallet -= self.triple_validation_slash
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
                agent.tokens_earned += 2 * self.triple_validation_reward
                agent.wallet += 2 * self.triple_validation_reward
                agent.rewarded_validations.append(self)
            for agent in self.rejects:
                agent.staked_tokens -= self.triple_validation_slash
                agent.wallet -= self.triple_validation_slash
                agent.slashed_validations.append(self)
            self.submitter.tokens_earned += self.triple_submission_reward


class Agent(MesaAgent):

    def __init__(self,
                 agent_id: int,
                 model,
                 initial_token_amount=100,
                 submission_rate: int = 90,
                 validation_rate: int = 60,
                 triple_submission_reward: float = 5,
                 triple_submission_slash: float = 1,
                 triple_validation_reward: float = 1,
                 triple_validation_slash: float = 1,
                 gt_validation_ratio: float = .9,
                 triple_space_size: int = 1000,
                ):
        super().__init__(agent_id, model)
        self.agent_id = agent_id

        self.staked_tokens = 0
        self.wallet = initial_token_amount
        self.tokens_earned = 0

        self.reputation = 0  # normalized between -1 and 1, 0 is neutral

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

        self.submitted_validations_count = 0
        self.starved_count = 0
        self.triple_submission_reward = triple_submission_reward
        self.triple_submission_slash = triple_submission_slash
        self.triple_validation_reward = triple_validation_reward
        self.triple_validation_slash = triple_validation_slash
        self.gt_validation_ratio = gt_validation_ratio

    def __repr__(self):
        return f"""Agent: {self.agent_id}
        Has {self.wallet} tokens in possession.
        Has {self.tokens_earned} tokens earned.
        Has {self.staked_tokens} tokens staked.
        Reputation: {self.reputation}
        Pending Triples: {len(self.submitted_triples["pending"])}
        Accepted Triples: {len(self.submitted_triples["accepted"])}
        Rejected Triples: {len(self.submitted_triples["rejected"])}
        Rewarded Validations: {len(self.rewarded_validations)}
        Slashed Validations: {len(self.slashed_validations)}
        Submitted Validations: {self.submitted_validations_count}
        """

    def __hash__(self):
        return hash(self.agent_id)

    def submit_entity(self):
        if self.wallet <= 0:
            return
        for i in range(0, 5):
            self.submit_triple(1)

    def submit_triple(self, _id=1):
        self.staked_tokens += self.triple_validation_slash
        self.wallet -= self.triple_validation_slash
        # create a triple in the graph submitted by this agent and adds to their submitted_triples
        self.submitted_triples["pending"].append(Triple(
            _id,
            self,
            gt_validated=bool(np.random.choice([0, 1], p=[1 - self.gt_validation_ratio, self.gt_validation_ratio])),
            triple_submission_reward=self.triple_submission_reward,
            triple_submission_slash=self.triple_submission_slash,
            triple_validation_reward=self.triple_validation_reward,
            triple_validation_slash=self.triple_validation_slash,
        ))

    def submitted_triples_available_to_validate(self):
        # returns: array of oldest X triples from submitted_triples queue not yet accepted or rejected where X is determined by remaining staked_tokens
        return [t for t in self.submitted_triples["pending"]][:int(self.staked_tokens//self.triple_validation_slash)]

    # Mesa Step
    def step(self):
        # Get current step count (Seconds)
        t_step = self.model.schedule.steps
        # Get agents pool
        agents = self.model.schedule.agents
        # print(self)
        # can't do anything if out of tokens
        if self.wallet <= 0:
            self.starved_count += 1
            print('agent starving!', t_step)
            return

        submit_p = self.validation_rate / (self.submission_rate + self.validation_rate)
        will_submit = bool(np.random.choice([0, 1], p=[1 - submit_p, submit_p]))
        # Submit triple
        if will_submit:
            submitted_ids = self.model.get_all_submitted_triple_ids()
            all_ids = set(range(self.model.triple_space_size))
            available_ids = all_ids.difference(submitted_ids)
            if len(available_ids) > 0:
                chosen = choice(list(available_ids))
                # print(self.agent_id, t_step, chosen)
                self.submit_triple(chosen)
                # self.submit_triple(randint(0, self.model.triple_space_size))
                # you can't do validation now
                return

        # Validate triple
        if not(will_submit):
            # Agent begins validation by retrieving triple from "queue", excluding self
            triples = np.concatenate([t for t in [a.submitted_triples_available_to_validate() for a in agents if a!=self]]).flat
            # Filter triples agent already voted on
            triples = list(filter(lambda x: (self not in x.accepts) and (self not in x.rejects), triples))
            # pick triple from triples
            triple = choice(triples) if triples else None

            if not triple:
                return

            # Agent will validate triple here
            if triple.gt_validated:
                agent_will_validate = bool(np.random.choice([0, 1], p=[1 - self.gt_validation_ratio, self.gt_validation_ratio]))  # Adding some randomness based on grounds truth validation ratio
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

            self.staked_tokens += 1
            self.submitted_validations_count += 1
            # Show Agent State
            #agents_staked_overtime[t] = pd.DataFrame([agent.__dict__ for agent in agents_state]).staked_tokens
            #agents_earned_overtime[t] = pd.DataFrame([agent.__dict__ for agent in agents_state]).tokens_earned