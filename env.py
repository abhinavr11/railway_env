import gym
import torch
import numpy as np
import random
import re
import networkx as nx
import matplotlib.pyplot as plt

class GridEnv(gym.Env):
    def __init__(self,args):
        super(GridEnv, self).__init__()
        self.args = args

        self.grid = self.create_network_grid(args)
        
        
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
            state_tensor[row,col] = 1
        
        self.grid = state_tensor
        return state_tensor

    def step(self, start, connecting_edge, end, time_step):
        # Implement Logic of Transition
        time = section_length/speed
        # 3, 4, 5, 6 3:7
        # Calculate Transition Track ID
        self.grid[start,time_step:time_step+time+1] = 1
        self.grid[end,time_step+time] = 1

 
    def reset(self):
        
        return self.grid

    def render(self,):
        print("Current State:", self.grid)

    
    def step(self, action):
        pass

def generate_track_mapping(num_stations, tracks_per_station):
    track_mapping = {}
    track_id = 0

    for station in range(1, num_stations + 1):
        # Map station tracks to track_id
        num_tracks_per_station = tracks_per_station[station-1]
        for track in range(1, num_tracks_per_station + 1):
            track_mapping[track_id] = ('station', station, track)
            track_id += 1

        # Add section between this station and the next station
        if station < num_stations:
            track_mapping[track_id] = ('section', station, station + 1)
            track_id += 1

    return track_mapping



def create_graph(args, flag):
    """
    Creates a directed NetworkX graph based on the provided arguments and flag.

    Parameters:
    - args (dict): Dictionary containing 'stations' and 'sections' data.
    - flag (int): Direction flag. Use 1 for left-to-right and -1 for right-to-left.

    Returns:
    - G (networkx.MultiDiGraph): The resulting directed graph.
    """
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
    # Sort nodes based on their numerical order (assuming nodes are named like 'S1', 'S2', ...)
    nodes = sorted(G.nodes(), key=lambda x: int(x[1:]))
    
    # Assign positions: place nodes in a straight horizontal line
    pos = {node: (i, 0) for i, node in enumerate(nodes)}
    
    plt.figure(figsize=(12, 4))
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=100)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20, edge_color='gray', width=1)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
    
    # Draw edge labels (length)
    edge_labels = nx.get_edge_attributes(G, 'length')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=3, label_pos=0.5)
    
    # Remove axes
    plt.axis('off')
    plt.title("Graph Visualization", fontsize=16)
    plt.tight_layout()
    plt.show()

def get_next_edges(G, current_pos):
    """
    Given a NetworkX graph and a current position (station or edge name),
    returns the names of the next possible edges.

    Parameters:
    - G (networkx.MultiDiGraph): The graph containing stations and sections.
    - current_pos (str): The current position, either a station name or an edge name.

    Returns:
    - List[str]: A list of edge names that are the next possible moves.

    Raises:
    - ValueError: If the current_pos is neither a valid station nor a valid edge name.
    """
    # Check if current_pos is a node (station name)
    if G.has_node(current_pos):
        # Get all outgoing edges from the station
        outgoing_edges = G.out_edges(current_pos, keys=True, data=True)
        # Extract the 'name' attribute of each outgoing edge
        next_edge_names = [data['name'] for _, _, key, data in G.edges(current_pos, keys=True, data=True)]
        return next_edge_names
    
    # If not a node, check if it's an edge name
    else:
        # Initialize variables to store edge information
        edge_found = False
        target_node = None
        
        # Iterate through all edges to find the edge with the given name
        for u, v, key, data in G.edges(keys=True, data=True):
            if data.get('name') == current_pos:
                edge_found = True
                # Determine the target node based on edge direction
                target_node = v
                break
        
        if edge_found:
            # Get all outgoing edges from the target node
            outgoing_edges = G.out_edges(target_node, keys=True, data=True)
            # Extract the 'name' attribute of each outgoing edge
            next_edge_names = [data['name'] for _, _, key, data in G.edges(target_node, keys=True, data=True)]
            return next_edge_names
        else:
            # current_pos is neither a valid node nor a valid edge name
            raise ValueError(f"'{current_pos}' is neither a valid station name nor an edge name in the graph.")


if __name__ == "__main__":
    tracks_per_station = [3,3,3,3,3,3,3,3,3,5]
    print(generate_track_mapping(10,tracks_per_station))
   
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

    G_lr = create_graph(args['sections'], 1)
    G_rl = create_graph(args['sections'], -1)



    env = GridEnv(args)
    env.reset()
    env.render()
