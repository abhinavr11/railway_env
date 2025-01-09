
import gym
import torch
import numpy as np
import random
import re
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict


class GridEnv(gym.Env):
    def __init__(self,args):
        super(GridEnv, self).__init__()
        self.args = args
        self.grid = self.create_network_grid(args)
        self.populate_trains(args)
        self.train_state = {}
       
       
    def create_network_grid(self,args):
        self.num_time_steps = args["max_time"] // args["time_step"] + 1
        self.num_nodes = len(self.create_station_section_array(args))
        state_tensor = torch.zeros(self.num_nodes,self.num_time_steps)
        return state_tensor

    def populate_trains(self,args):
        state_tensor = self.grid

        train_init_states = []
        for t in args["train_configuration"]:
            train_init_states.append((t['name'],t['origin'],t['starting_time'],t['destination'],t['priority']))

        train_id = 0
        for nm, ss, st, ds, pr in train_init_states:
            row = self.find_indices(self.GRID_ROW_INFO,ss)[0]
            destination = self.find_indices(self.GRID_ROW_INFO,ds)[0]
            col = st//args['time_step']
            state_tensor[row,col] = 1
            self.train_state[train_id] = (row, destination, pr)
            train_id += 1
       
        self.grid = state_tensor
        
        return state_tensor

    def step(self, action, start, connecting_edge, end, t_start, train_id, destination, neighbors):
        # Implement Logic of Transition
        tau_d = 2 # Block section from t_end+1 to t_end+tau_d and t_start-tau_d to t_start-1
        tau_arr = 1 # Block all other tracks of the destination station from t_end-tau_arr to t_end+tau_arr
        tau_pre = 2 # Block the destination track from t_end-tau_pre to t_end-1.  
        tau_min = 3 # Minimum Dwelling Time    
        list_tracks_end = [3, 4, 6]  
        section_length = 1000
        speed = 100                                                                                                                                                                                                                                                                                    
        action = 'move'
        state = self.grid
        P = -100
        alpha = 1
        beta = 4000
        DP = 200 # DP = prioirty*Halt
        R_done = 10
        R_halt = -0.01
        R_move = 0
        done = False

        if action == 'move':
            time_section = section_length/speed
            t_end = t_start + time_section
           
            if (self.grid[connecting_edge, t_start:t_end+1] != 0).any()== True:
                reward = P # Give negative reward
                return state, action, self.grid, reward, done

           
            self.grid[start, t_start] = 1 # Starting track
            self.grid[connecting_edge, t_start:t_end+1] = 1 # Populate section
           
            self.grid[connecting_edge, t_start-tau_d:t_start] = 255 # Block section from t_start-tau_d to t_start-1
            self.grid[connecting_edge, t_end+1:t_end+tau_d+1] = 255 # Block section from t_end+1 to t_end+tau_d

           
            for track in list_tracks_end:
                self.grid[track, t_end-tau_arr:t_end+tau_arr+1] = 255 # Block all tracks of the destination station from t_end-tau_arr to t_end+tau_arr

            self.grid[end, t_end-tau_pre:t_end] = 200 # Block the destination track from t_end-tau_pre to t_end-1.      
            self.grid[end,t_end+1] = 255
            self.grid[end, t_end] = 1 # Ending track
            self.train_state[train_id][0] = end
            reward = R_move * self.train_state[train_id][2]

            if all(value[0] == value[1] for value in self.train_state.values()): ### If all trains reach their respective destinations do this.
                reward = alpha*(beta-DP)
                return state, action, self.grid, reward, done

            if self.train_state[train_id][0] == self.train_state[train_id][1] :
                reward = R_done * self.train_state[train_id][2]
                return state, action, self.grid, reward, done

            return state, action, self.grid, reward, done
        else:
         
             # First time dwelling
            if self.grid[start,t_start-1] != 1: # If previous time_step is not occupied by train(that is 1), do this
                if (self.grid[start, t_start:t_start+tau_min+1] != 0).any()== True:
                    reward = P # Give negative reward
                    return state, action, self.grid, reward, done
                self.grid[start, t_start:t_start+tau_min+1] = 1
                reward = R_halt * tau_min * self.train_state[train_id][2]
                return state, action, self.grid, reward, done
            # If next section is free and it is not the first time halting do one step
            elif start+1 < self.num_nodes and t_start+1 < self.num_time_steps and self.grid[start+1,t_start+1] == 0:
                if (self.grid[start,t_start:t_start+2] != 0).any()== 0:
                    reward = P # Give negative reward
                    return state, action, self.grid, reward, done
                self.grid[start,t_start:t_start+2] = 1
                reward = R_halt * 1 * self.train_state[train_id][2]
                return state, action, self.grid, reward, done
            # If next section is not free find the timestep corresponding to when next section is available.
            else:
                if (self.grid[start, t_start:non_zero_indices+2] != 0).any()== True:
                    reward = P # Give negative reward
                    return state, action, self.grid, reward, done
                row = self.grid[start+1] # Have to handle the out of bound case for start+1
                # Search for the first non-zero column index from the t_start index onwards
                non_zero_indices = (row[t_start+1:] != 0).nonzero(as_tuple=True)[0]
                self.grid[start, t_start:non_zero_indices+2] = 1   
                reward =  R_halt * (non_zero_indices+1-t_start) * self.train_state[train_id][2]
                return state, action, self.grid, reward, done


 
    def reset(self):
       
        return self.grid

    def render(self,):
        print("Current State:", self.grid)


    def create_station_section_array(self,args):
        stations = args.get('stations', [])
        sections = args.get('sections', [])
        result = []
        num_sections = len(sections)
       
        for i, station_dict in enumerate(stations):
            station_name = next(iter(station_dict))
            capacity = station_dict[station_name].get('capacity', 1)
            result.extend([station_name] * capacity)
            if i < num_sections:
                section_name = next(iter(sections[i]))
                result.append(section_name)
       
        self.GRID_ROW_INFO = result
        return result
   
    def create_string_to_indices_map(self,result_array):
   
        mapping = defaultdict(list)
        for idx, item in enumerate(result_array):
            mapping[item].append(idx)
        return mapping

    def get_next_index(self,mapping, counters, ss):
   
        if ss in mapping and counters[ss] < len(mapping[ss]):
            index = mapping[ss][counters[ss]]
            counters[ss] += 1
            return index
        else:
            raise ValueError("More trains at a station than allowed")

    def find_indices(self,result_array, ss_queries): # given the GRID_ROW_INFO find indices of row
        ss_queries = [ss_queries]
        mapping = self.create_string_to_indices_map(result_array)
        counters = defaultdict(int)
        indices = []
       
        for ss in ss_queries:
            index = self.get_next_index(mapping, counters, ss)
            indices.append(index)
       
        return indices

    def where_is_my_train(self,row,col):
        print('Train present at :',self.GRID_ROW_INFO[row],' at time step :', col*self.args['time_step'])

def create_graph(args, flag):
    if flag not in [1, -1]:
        raise ValueError("Flag must be either 1 (left-to-right) or -1 (right-to-left).")
   
    G = nx.MultiDiGraph()
   
    # Add section edges
    for section in args['sections']:
        for section_name, details in section.items():
            start = details['start']
            end = details['end']
            length = details['length']
           
            if flag == 1:
                source, target = start, end
            else:
                source, target = end, start
           
            G.add_edge(source, target, name=section_name, length=length)
   
    # Add station capacity edges (self-loops)
    for station in args['stations']:
        for station_name, attrs in station.items():
            capacity = attrs['capacity']
            for i in range(1, capacity + 1):
                edge_name = f"{station_name}_l{i}"
                G.add_edge(station_name, station_name, name=edge_name)
   
    return G

def visualize_graph(G):
    nodes = sorted(G.nodes(), key=lambda x: int(x[1:]))
    pos = {node: (i, 0) for i, node in enumerate(nodes)}
    plt.figure(figsize=(12, 4))
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=100)
    nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20, edge_color='gray', width=1)
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
    edge_labels = nx.get_edge_attributes(G, 'length')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=3, label_pos=0.5)
   

    plt.axis('off')
    plt.title("Graph Visualization", fontsize=16)
    plt.tight_layout()
    plt.show()

def get_next_edges(G, current_pos):  
    if G.has_node(current_pos):
        outgoing_edges = G.out_edges(current_pos, keys=True, data=True)

        next_edge_names = [data['name'] for _, _, key, data in G.edges(current_pos, keys=True, data=True)]
        return next_edge_names

    else:

        edge_found = False
        target_node = None

        for u, v, key, data in G.edges(keys=True, data=True):
            if data.get('name') == current_pos:
                edge_found = True
                target_node = v
                break
       
        if edge_found:
            outgoing_edges = G.out_edges(target_node, keys=True, data=True)

            next_edge_names = [data['name'] for _, _, key, data in G.edges(target_node, keys=True, data=True)]
            return next_edge_names
        else:

            raise ValueError(f"'{current_pos}' is neither a valid station name nor an edge name in the graph.")


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

    G_lr = create_graph(args, 1)
    G_rl = create_graph(args, -1)



    env = GridEnv(args)
    env.reset()
    env.render()
