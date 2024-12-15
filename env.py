import gym
import torch
import numpy as np
import random
import re

class GridEnv(gym.Env):
    def __init__(self,args):
        super(GridEnv, self).__init__()
        self.args = args

        self.grid = self.create_network_grid(args)
        self.grid = self.populate_trains(args)
        
    def create_network_grid(self,args):
        self.num_time_steps = args["max_time"] // args["time_step"] + 1
        self.num_nodes = len(args["stations"]) + len(args["sections"])
        state_tensor = torch.zeros(self.num_nodes,self.num_time_steps)

        self.stations = []
        for s in self.args['stations']:
            self.stations.append(list(s.keys())[0])

        self.sections = []
        for s in self.args['sections']:
            self.sections.append(list(s.keys())[0])
        state_tensor[:, 0] = torch.tensor([float(re.sub(r'[^0-9.]', '', item)) for pair in zip(self.stations, self.sections) for item in pair] + [float(re.sub(r'[^0-9.]', '', self.stations[-1]))] )            
        
        return state_tensor

    def populate_trains(self,args):
        state_tensor = self.grid

        train_init_states = []
        for t in args["train_configuration"]:
            train_init_states.append((float(t['name'][1:]),float(t['origin'][1:]),t['starting_time']))

        for nm, ss, st in train_init_states:
            row = torch.where(state_tensor[:,0] == ss)[0]
            col = st//args['time_step'] +1
            state_tensor[row,col] = nm
        
        return state_tensor

 
    def reset(self):
        
        return self.grid

    def render(self,):
        print("Current State:", self.grid)

    
    def step(self, action):
        pass


if __name__ == "__main__":
   
    args = {
        "stations": [
            {"S1": {"capacity": 3}},
            {"S2": {"capacity": 3}},
            {"S3": {"capacity": 3}},
            {"S4": {"capacity": 3}},
            {"S5": {"capacity": 3}},
            {"S6": {"capacity": 3}},
            {"S7": {"capacity": 3}},
            {"S8": {"capacity": 3}},
            {"S9": {"capacity": 3}},
            {"S10": {"capacity": 5}}
        ],

        "sections": [
            {'s12': {"start": "S1", "end": "S2", "length": 10000}},
            {'s23': {"start": "S2", "end": "S3", "length": 20000}},
            {'s34': {"start": "S3", "end": "S4", "length": 10000}},
            {'s45': {"start": "S4", "end": "S5", "length": 20000}},
            {'s56': {"start": "S5", "end": "S6", "length": 20000}},
            {'s67': {"start": "S6", "end": "S7", "length": 10000}},
            {'s78': {"start": "S7", "end": "S8", "length": 20000}},
            {'s89': {"start": "S8", "end": "S9", "length": 20000}},
            {'s910': {"start": "S9", "end": "S10", "length": 20000}}
        ],

    "train_configuration" : [
        {"name": "t1", "origin": "S1", "destination": "S10", "starting_time": 0, "priority": 3, "speed": 1500},
        {"name": "t2", "origin": "S10", "destination": "S1", "starting_time": 40, "priority": 2, "speed": 1000},
        {"name": "t3", "origin": "S1", "destination": "S10", "starting_time": 80, "priority": 1, "speed": 500},
        {"name": "t4", "origin": "S10", "destination": "S1", "starting_time": 120, "priority": 3, "speed": 1500},
        {"name": "t5", "origin": "S1", "destination": "S10", "starting_time": 160, "priority": 3, "speed": 1500},
        {"name": "t6", "origin": "S10", "destination": "S1", "starting_time": 200, "priority": 3, "speed": 1500},
        {"name": "t7", "origin": "S1", "destination": "S10", "starting_time": 240, "priority": 3, "speed": 1500},
        {"name": "t8", "origin": "S10", "destination": "S1", "starting_time": 280, "priority": 3, "speed": 1500}
    ],
    "max_time":1000,
    "time_step": 10

        }



    env = GridEnv(args)
    env.reset()
    env.render()
