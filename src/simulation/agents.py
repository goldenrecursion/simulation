import time
import datetime
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
        self.creation_date = datetime.datetime.now()

    def __repr__(self):
        return f"""Triple ID: {self._id}
        submitted by: {self.submitter.agent_id}
        num_accepts: {len(self.accepts)}
        num_rejects: {len(self.rejects)}
        """

    def score(self):
        value = 0
        value += sum([item.gt_validation_ratio for item in self.accepts])
        value -= sum([item.gt_validation_ratio for item in self.rejects])
        return value

    def accepted(self):
        return self.score() >= 3.5
        total_votes = len(self.accepts) + len(self.rejects)
        accept_ratio = len(self.accepts) / (total_votes + 1)
        p_thresh = .6  # accept_ratio threshold to validate
        q_thresh = 5  # minimum votes to validate

        return total_votes >= q_thresh and accept_ratio > p_thresh

    def rejected(self):
        return self.score() <= -3.5
        total_votes = len(self.accepts) + len(self.rejects)
        accept_ratio = len(self.accepts) / (total_votes + 1)
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
                # agent.staked_tokens -= self.triple_validation_slash
                # agent.wallet -= self.triple_validation_slash
                agent.slashed_validations.append(self)
            # self.submitter.staked_tokens -= self.triple_submission_slash

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
                # agent.staked_tokens -= self.triple_validation_slash
                # agent.wallet -= self.triple_validation_slash
                agent.slashed_validations.append(self)
            self.submitter.tokens_earned += self.triple_submission_reward


class AgentProfile():

    def __init__(self,
                 initial_token_amount=0,
                 submission_rate=0,
                 validation_rate=0,
                 triple_submission_reward: float = 5,
                 triple_submission_slash: float = 1,
                 triple_validation_reward: float = 1,
                 triple_validation_slash: float = 1,
                 gt_validation_ratio: float = .9,
                 is_colluder: bool = False,
                 ):
        self.initial_token_amount = initial_token_amount
        self.submission_rate = submission_rate
        self.validation_rate = validation_rate
        self.triple_submission_reward = triple_submission_reward
        self.triple_submission_slash = triple_submission_slash
        self.triple_submission_reward = triple_submission_reward
        self.triple_submission_slash = triple_submission_slash
        self.triple_validation_reward = triple_validation_reward   
        self.triple_validation_slash = triple_validation_slash
        self.gt_validation_ratio = gt_validation_ratio 
        self.is_colluder = is_colluder

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
                 is_colluder: bool = False,
                ):

        super().__init__(agent_id, model)
        self.agent_id = agent_id

        self.staked_tokens = initial_token_amount
        self.wallet = 0
        self.tokens_earned = 0

        self.reputation = gt_validation_ratio  # normalized between -1 and 1, 0 is neutral

        self.submission_rate = submission_rate
        self.validation_rate = validation_rate

        self.is_colluder = is_colluder

        self.submitted_triples = {
            "pending": [],
            "accepted": [],
            "rejected": [],
        }

        # Might get memory intensive
        self.rewarded_validations = []
        self.slashed_validations = []

        self.submitted_validations_count = 0
        self.triple_submission_reward = triple_submission_reward
        self.triple_submission_slash = triple_submission_slash
        self.triple_validation_reward = triple_validation_reward
        self.triple_validation_slash = triple_validation_slash
        self.gt_validation_ratio = gt_validation_ratio


    @classmethod
    def from_agent_profile(cls,
                 agent_id: int,
                 model: Model,
                 agent_profile: AgentProfile,
                 triple_space_size: int = 1000,
                ):
        return cls(
                agent_id=agent_id,
                model=model,
                initial_token_amount=agent_profile.initial_token_amount,
                submission_rate=agent_profile.submission_rate,
                validation_rate=agent_profile.validation_rate,
                triple_submission_reward=agent_profile.triple_submission_reward,
                triple_submission_slash=agent_profile.triple_submission_slash,
                triple_validation_reward=agent_profile.triple_validation_reward,
                triple_validation_slash=agent_profile.triple_validation_slash,
                gt_validation_ratio=agent_profile.gt_validation_ratio,
                triple_space_size=triple_space_size,
                is_colluder=agent_profile.is_colluder,
                )

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
        if self.staked_tokens <= 0:
            return
        for i in range(0, 5):
            self.submit_triple(1)

    def calculate_fee(self):
        return self.triple_validation_slash
        triples = self.submitted_triples
        triples = [triples.get(k) for k in triples.keys() if k!='pending']
        triples = [item for sublist in triples for item in sublist]
        submission_data = [(t.creation_date.timestamp(), 1 if t.status==TripleStatus.ACCEPTED else -1) for t in triples]

        validated_triples = self.validated_triples()
        validated_triples = [t for t in validated_triples if t.status in (TripleStatus.ACCEPTED, TripleStatus.REJECTED)]
        validation_data = [(t.creation_date.timestamp(), 1 if ((t.status==TripleStatus.ACCEPTED and self in t.accepts) or (t.status==TripleStatus.REJECTED and self in t.rejects)) else -1) for t in validated_triples]

        data = submission_data
        data.extend(validation_data)

        if len(data)==0:
            return self.triple_validation_slash

        data = pd.DataFrame(data)
        data.sort_values(0, inplace=True)
        data.insert(0, 't', range(0, 0 + len(data)))
        data.columns = ['t', 'ts', 'q']
        data['w'] = data['t'].ewm(span=20, adjust=False).mean()
        data['score'] = data['w'] * data['q']
        return self.triple_validation_slash * max([0.01, 1 - data['score'].sum() / data['w'].sum()])


    def submit_triple(self, _id=1):
        self.staked_tokens -= self.calculate_fee()
        # self.wallet -= self.triple_validation_slash
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
        return [t for t in self.model.get_pending_triples() if (self not in t.accepts and t not in t.rejects and t.submitter!=self)]
        # returns: array of oldest X triples from submitted_triples queue not yet accepted or rejected where X is determined by remaining staked_tokens
        # return [t for t in self.submitted_triples["pending"]][:int(self.staked_tokens//self.triple_validation_slash)]

    def submitted_triples_count(self):
        triples = self.submitted_triples
        return sum([len(triples.get(k)) for k in triples.keys()])

    def validated_triples(self):
        triples = self.model.get_all_triples()
        triples = [t for t in triples if (self in t.accepts or self in t.rejects)]
        return triples

    def validated_triples_pending(self):
        triples = self.validated_triples()
        triples = [t for t in triples if t.status==TripleStatus.PENDING]
        return triples

    # Mesa Step
    def step(self):
        # Get current step count (Seconds)
        t_step = self.model.schedule.steps
        # Get agents pool
        agents = self.model.schedule.agents
        # print(self)
        # can't do anything if out of tokens
        if self.staked_tokens <= 0:
            self.model.stalled_count += 1
            # print('agent starving!', t_step)
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
            else:
                will_submit = False

        # Validate triple
        if not(will_submit):
            # Agent begins validation by retrieving triple from "queue", excluding self
            # triples = np.concatenate([t for t in [a.submitted_triples_available_to_validate() for a in agents if a!=self]]).flat
            # Filter triples agent already voted on
            # triples = list(filter(lambda x: (self not in x.accepts) and (self not in x.rejects), triples))

            # faster
            triples = self.submitted_triples_available_to_validate()
            # triples = [item for sublist in
            #            [t for t in [a.submitted_triples_available_to_validate() for a in agents if a!=self]]
            #            for item in sublist
            #            if (self not in item.accepts and self not in item.rejects)]
            # pick triple from triples
            if self.is_colluder:
                colluder_ids = [k for k in self.model.colluder_dict.keys() if self.model.colluder_dict.get(k)]
                colluder_triples = [t for t in triples if (t.submitter.agent_id % 10 ==0)]
                if len(colluder_triples) > 0:
                    triples = colluder_triples
            triple = choice(triples) if triples else None

            if not triple:
                self.model.stalled_count += 1
                return

            # Agent will validate triple here
            if triple.gt_validated:
                if self.is_colluder:
                    agent_will_validate = True
                else:
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

            self.staked_tokens -= self.calculate_fee()
            self.submitted_validations_count += 1
            # Show Agent State
            #agents_staked_overtime[t] = pd.DataFrame([agent.__dict__ for agent in agents_state]).staked_tokens
            #agents_earned_overtime[t] = pd.DataFrame([agent.__dict__ for agent in agents_state]).tokens_earned