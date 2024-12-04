import gym
import torch
import numpy as np
import random

class GraphEnv(gym.Env):
    def __init__(self,args):
        super(GraphEnv, self).__init__()
        self.args = args
        self.graph = {
            0: [1],        # Node 0 connects to Node 1
            1: [0, 2, 3, 4],  # Node 1 connects to Nodes 0, 2, 3, and 4
            2: [1, 5],        # Node 2 connects to Node 1 and 5
            3: [1, 5],        # Node 3 connects to Node 1 and 5
            4: [1, 6],        # Node 4 connects to Node 1 and 6
            5: [2, 3, 7],        # Node 5 connects to Node 2, 3 and 7
            6: [4, 7],        # Node 6 connects to Node 4 and 7
            7: [5, 6]          # Node 7 connects to Nodes 5 and 6
        }

        self.nodes = list(self.graph.keys())


        self.edges = []
        for src in self.graph:
            for dst in self.graph[src]:
                self.edges.append((src, dst))

        self.edges_tensor = torch.tensor(self.edges, dtype=torch.long)
        self.action_space = torch.tensor(self.nodes, dtype=torch.long)
        self.observation_space = torch.zeros(len(self.nodes), dtype=torch.long)

        self.state = torch.zeros(len(self.nodes), dtype=torch.long)

    def reset(self):
        self.state = torch.zeros(len(self.nodes), dtype=torch.long)
        return self.state

    def render(self,):
        print("Current State:", self.state)

    
    def step(self, action):
        pass

if __name__ == "__main__":
    NUM_TRAINS = 3
    NUM_STATIONS = 7
    trains = [ _ for _ in range(NUM_TRAINS)]
    trains_start = random.sample(range(NUM_STATIONS + 1), NUM_TRAINS)
    trains_end = []
    
    for start in trains_start:
        possible_ends = [i for i in range(NUM_STATIONS + 1) if i != start]
        end = random.choice(possible_ends)
        trains_end.append(end)
    
    train_direction = []
    for idx in trains:
        if trains_start[idx] > trains_end[idx]:
            train_direction.append(-1.)
        else:
            train_direction.append(1.)


    args = {
        'NUM_TRAINS' : NUM_TRAINS,
        'NUM_STATIONS':NUM_STATIONS,
        'trains':trains,
        'trains_start':trains_start,
        'trains_end':trains_end,
        'train_direction': train_direction
    }
    env = GraphEnv(args)
    env.reset()
    env.render()
