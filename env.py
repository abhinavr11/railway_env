
# import gym
import torch
import numpy as np
import random
import re
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import copy


class GridEnv():
    def __init__(self,args):
        super(GridEnv, self).__init__()
        self.args = args
        self.grid = self.create_network_grid(args)
        self.train_state = {} # Dictionary of "train_id : (Curr_location, destination, priority)"
        self.parallel_tracks = {} # List of parallel tracks of a station
        self.populate_trains(args)
        self.stations_list, self.station_info = self.build_station_info(args)
        self.horizontal_size = 13
        self.vertical_size = 41


    
    def create_network_grid(self,args):
        self.num_time_steps = args["max_time"] // args["time_step"] + 1
        self.num_nodes = len(self.create_station_section_array(args))
        state_tensor = torch.zeros(self.num_nodes,self.num_time_steps)
        return state_tensor

    def get_station_tracks(self, station_name, station_info):
        """
        Given a station name and the dictionary station_info (which includes start_track and capacity),
        return the list of track IDs that this station occupies.
        """
        start = station_info[station_name]["start_track"]
        cap = station_info[station_name]["capacity"]
        return list(range(start, start + cap))

    def build_station_info(self, args):
        """
        Given the 'args' dictionary containing station-capacity mappings,
        return two things:
        1) A list of station names in order (e.g. ["S1", "S2", ...])
        2) A dict mapping station name -> (start_track, capacity)
            where start_track is the first track ID used by that station.
        """
        # 1. Extract station names (S1, S2, ...) and capacities in order
        stations_list = []
        capacities = []
        for station_dict in args["stations"]:
            # Each element of station_dict looks like {"S1": {"capacity": 3}}
            # Extract station name (e.g. "S1") and station data {"capacity": 3}
            for station_name, station_data in station_dict.items():
                stations_list.append(station_name)
                capacities.append(station_data["capacity"])

        # 2. Compute start_track for each station
        #    Station i has track IDs from start_track[i] to start_track[i] + capacity[i] - 1
        #    The connecting track to the next station is then start_track[i] + capacity[i]
        station_name_to_info = {}
        current_track = 0
        for i, station_name in enumerate(stations_list):
            cap = capacities[i]
            station_name_to_info[station_name] = {
                "start_track": current_track,
                "capacity": cap
            }
            # Move current_track forward by capacity + 1 (the +1 is for the section track to the next station)
            current_track += cap + 1

        return stations_list, station_name_to_info

    def pass_station(self, current_station, direction, stations_list, station_info):
        """
        Given:
        - current_station: e.g. "S1"
        - direction: "down" (S1 -> S2 -> S3 -> ...) or "up" (S2 -> S1 -> S0 -> ...)
        - stations_list: ordered list of stations e.g. ["S1", "S2", ..., "S10"]
        - station_info: mapping station_name -> {start_track, capacity}

        If direction == "down" and current_station == "S1", we want S2's track IDs, etc.
        If direction == "up" and current_station == "S2", we want S1's track IDs, etc.

        Returns a list of the neighbor station's track IDs.
        """
        # Find index of current_station in the ordered stations_list
        idx = stations_list.index(current_station)

        if direction == 1:
            # If going "down", the neighbor is station idx+1 (if it exists)
            if idx < len(stations_list) - 1:
                neighbor_station = stations_list[idx + 1]
                return self.get_station_tracks(neighbor_station, station_info)
            else:
                # No station further down; return empty or handle boundary
                return []

        else :
            # If going "up", the neighbor is station idx-1 (if it exists)
            if idx > 0:
                neighbor_station = stations_list[idx - 1]
                return self.get_station_tracks(neighbor_station, station_info)
            else:
                # No station further up; return empty or handle boundary
                return []

        

    def find_station_by_track_id(self,track_id, stations_list, station_info):
        """
        Given a track_id (an integer), return the station name in which
        this track ID belongs. If it doesn't belong to any station, return None.
        """
        for station_name in stations_list:
            start = station_info[station_name]["start_track"]
            cap = station_info[station_name]["capacity"]

            # Tracks for this station go from 'start' up to 'start + cap - 1'
            if start <= track_id < (start + cap):
                return station_name
        return None

    def populate_trains(self,args):
        state_tensor = self.grid

        train_init_states = []
        for t in args["train_configuration"]:
            train_init_states.append((t['name'],t['origin'],t['starting_time'],t['destination'],t['priority']))
        current_index = 0

        for station in args["stations"]:
            capacity = list(station.values())[0]["capacity"]
            station_tracks = list(range(current_index, current_index + capacity))

            for track in station_tracks:
                self.parallel_tracks[track] = [t for t in station_tracks if t != track]

            current_index += capacity + 1  # +1 to account for the section

        train_id = 0
        for nm, ss, st, ds, pr in train_init_states:
            row = self.find_indices(self.GRID_ROW_INFO,ss)[0]
            destination = self.find_indices(self.GRID_ROW_INFO,ds)[0]
            col = st//args['time_step']
            state_tensor[row,col] = 1
            if ds > ss:
                 direction = 1 # Destination > Origin means train is going from left to right
            else:
                 direction = -1
            self.train_state[train_id] = [row, col, destination, pr, direction]
            train_id += 1

        self.grid = state_tensor

        return state_tensor

    def calculate_end(self, action, start, t_start):
        if action == 'move':
            time_section = section_length//speed
            t_end = t_start + time_section
        else:
            if self.grid[start,t_start-1] > 1:
                t_end = t_start+tau_min
            elif start+1 < self.num_nodes and t_start+1 < self.num_time_steps and self.grid[start+1,t_start+1] == 0:
                t_end = t_start + 1
            else:
                row = self.grid[start+1] # Have to handle the out of bound case for start+1
                # Search for the first non-zero column index from the t_start index onwards
                non_zero_indices = (row[t_start+1:] != 0).nonzero(as_tuple=True)[0]
                t_end = t_start+non_zero_indices+1
        return t_end
         

    def select_action(self, train_id, curr_track, t_start, station_no, direction, epsilon):
        arrival_track = -1
        if np.random.random() < epsilon:
            arrival_track = random.choice(self.pass_station(station_no, direction, self.stations_list, self.station_info) + [curr_track])
            if arrival_track == curr_track:
                action = 'halt'
            else:
                action = 'move'
        else:
            all_act_space = self.pass_station(station_no, direction, self.stations_list, self.station_info) + [curr_track]
            max_Q_value = -10000000
            for track in range(len(all_act_space)-1):
                t_end = self.calculate_end('move', curr_track, t_start)
                obs_space = self.cropping_window(curr_track, track, t_start, t_end)
              
                if self.Q_net(obs_space) > max_Q_value:
                    max_Q_value = self.Q_net(obs_space)[0] # [0] indicates 'move'
                    action = 'move'
                    arrival_track = track

            t_end = self.calculate_end('halt', curr_track, t_start)
            obs_space = self.cropping_window(curr_track, curr_track, t_start, t_end)
            if self.Q_net(obs_space) > max_Q_value:
                    max_Q_value = self.Q_net(obs_space)[1] # [0] indicates 'halt'
                    action = 'halt'
                    arrival_track = curr_track

        return action, arrival_track

    
    def cropping_window(self, current_track, arrival_track, t_start, t_end):
        # Calculate vertical half-size
        vertical_half_size = (self.vertical_size - 1) // 2
        # Calculate horizontal half-size
        horizontal_half_size = (self.horizontal_size - 1) // 2
        
        # Extract window for the current_track centered vertically and horizontally around current_track and t_start
        o_left = self.grid[
            current_track - vertical_half_size : current_track + vertical_half_size + 1,
            t_start - horizontal_half_size : t_start + horizontal_half_size + 1
        ]
        
        # Extract window for the arrival_track centered vertically and horizontally around arrival_track and t_end
        o_right = self.grid[
            arrival_track - vertical_half_size : arrival_track + vertical_half_size + 1,
            t_end - horizontal_half_size : t_end + horizontal_half_size + 1
        ]
        
        return o_left, o_right

    
    def step(self, action, arrival_track, train_id):
        # Implement Logic of Transition
        tau_d = 2 # Block section from t_end+1 to t_end+tau_d and t_start-tau_d to t_start-1
        tau_arr = 1 # Block all other tracks of the destination station from t_end-tau_arr to t_end+tau_arr
        tau_pre = 2 # Block the destination track from t_end-tau_pre to t_end-1.
        tau_min = 3 # Minimum Dwelling Time
        
        # temp = self.grid
        state = copy.deepcopy(self.grid)
        P = -100
        alpha = 1
        beta = 4000
        DP = 200 # DP = prioirty*Halt
        R_done = 10
        R_halt = -0.01
        R_move = 0
        done = False
        section_length = 1000
        speed = 100
        # row, col, destination, pr, direction
        start = self.train_state[train_id][0] # Starting track 
        direction = self.train_state[train_id][4]
        station_no = self.find_station_by_track_id(start, self.stations_list, self.station_info)
        
        # if self.pass_station(station_no, direction, self.stations_list, self.station_info):
            
        #     end = random.choice(self.pass_station(station_no, direction, self.stations_list, self.station_info))
        if direction == 1:
            connecting_edge = min(self.pass_station(station_no, direction, self.stations_list, self.station_info)) - 1
        else:
            connecting_edge = max(self.pass_station(station_no, direction, self.stations_list, self.station_info)) + 1
        t_start = self.train_state[train_id][1]

        if action == 'move':
            time_section = section_length//speed
            t_end = t_start + time_section

            if (self.grid[connecting_edge, t_start:t_end+1] != 0).any()== True:
                reward = P # Give negative reward
                return state, action, self.grid, reward, done


            # self.grid[start, t_start] = 1 # Starting track
            self.grid[connecting_edge, t_start:t_end+1] = 1 # Populate section

            self.grid[connecting_edge, t_start-tau_d:t_start] = 255 # Block section from t_start-tau_d to t_start-1
            self.grid[connecting_edge, t_end+1:t_end+tau_d+1] = 255 # Block section from t_end+1 to t_end+tau_d



            for track in self.parallel_tracks[arrival_track]:
                self.grid[track, t_end-tau_arr:t_end+tau_arr+1] = 255 # Block all tracks of the destination station from t_end-tau_arr to t_end+tau_arr

            self.grid[arrival_track, t_end-tau_pre:t_end] = 200 # Block the destination track from t_end-tau_pre to t_end-1.
            self.grid[arrival_track,t_end+1] = 255
            self.grid[arrival_track, t_end] = 1 # Ending track
            self.train_state[train_id][0] = arrival_track
            self.train_state[train_id][1] = t_end
            reward = R_move * self.train_state[train_id][3]

            if all(value[0] == value[2] for value in self.train_state.values()): ### If all trains reach their respective destinations do this.
                reward = alpha*(beta-DP)
                return state, action, self.grid, reward, done

            if self.train_state[train_id][0] == self.train_state[train_id][2] :
                reward = R_done * self.train_state[train_id][3]
                return state, action, self.grid, reward, done

            return state, action, self.grid, reward, done
        else:

             # First time dwelling
            if self.grid[start,t_start-1] > 1: #  REVIEW THIS !!!!!!!!!!!  If previous time_step is not occupied by train(that is 1), do this
                if (self.grid[start, t_start:t_start+tau_min+1] != 0).any()== True:
                    reward = P # Give negative reward
                    return state, action, self.grid, reward, done
                self.grid[start, t_start:t_start+tau_min+1] = 1
                reward = R_halt * tau_min * self.train_state[train_id][3]
                self.train_state[train_id][1] = t_start+tau_min
                return state, action, self.grid, reward, done
            # If next section is free and it is not the first time halting do one step
            elif start+1 < self.num_nodes and t_start+1 < self.num_time_steps and self.grid[start+1,t_start+1] == 0:
                if (self.grid[start,t_start:t_start+2] != 0).any()== 0:
                    reward = P # Give negative reward
                    return state, action, self.grid, reward, done
                self.grid[start,t_start:t_start+2] = 1
                reward = R_halt * 1 * self.train_state[train_id][3]
                self.train_state[train_id][1] = t_start+1
                return state, action, self.grid, reward, done
            # If next section is not free find the timestep corresponding to when next section is available.
            else:
                # REVIEW THIS.......................................
                if (self.grid[start, t_start:non_zero_indices+2] != 0).any()== True:
                    reward = P # Give negative reward
                    return state, action, self.grid, reward, done
                row = self.grid[start+1] # Have to handle the out of bound case for start+1
                # Search for the first non-zero column index from the t_start index onwards
                non_zero_indices = (row[t_start+1:] != 0).nonzero(as_tuple=True)[0]
                self.grid[start, t_start:non_zero_indices+2] = 1 # Fill the station row
                reward =  R_halt * (non_zero_indices+1-t_start) * self.train_state[train_id][3]
                self.train_state[train_id][0] = start
                self.train_state[train_id][1] = non_zero_indices+1
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
        print(ss_queries)
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


    a = env.step('move',0) #action, start, connecting_edge, end, t_start, train_id, destination
