from network_base import *
from replay_buffer import *
import torch
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np

M_ep          = 1000   # Number of episodes
M_sr          = 200    # Max steps per episode
buffer_size   = 10000
batch_size    = 32
lr            = 1e-4
epsilon       = 1.0
epsilon_decay = 0.01
epsilon_min   = 0.05
M_up          = 50     # Update target network every M_up steps
xi            = 0.1    # Weight for soft update of target network
num_agents    = 5      # Number of agents

device = 'cuda' if torch.cuda.is_available() else 'cpu'

env = ...  # Your eTGM environment with reset(), step(), etc.

net = QNet_Base(
    in_channels_left=1,
    in_channels_right=1,
    left_input_shape=(1, 84, 84),
    right_input_shape=(1, 84, 84),
    fc_input_dim=10,
).to(device)

# Initialize target network for double-network approach
net_target = QNet_Base(
    in_channels_left=1,
    in_channels_right=1,
    left_input_shape=(1, 84, 84),
    right_input_shape=(1, 84, 84),
    fc_input_dim=10,
).to(device)
net_target.load_state_dict(net.state_dict())  # Initialize target with same weights

# Initialize Replay Buffer and Optimizer
replay_buffer = ReplayBuffer(buffer_size=buffer_size, batch_size=batch_size, device=device)
optimizer = optim.Adam(net.parameters(), lr=lr)

def soft_update(net, net_target, xi):
    """Perform a soft update of the target network parameters."""
    for target_param, param in zip(net_target.parameters(), net.parameters()):
        target_param.data.copy_(xi * param.data + (1.0 - xi) * target_param.data)

def is_RDone_p_i(reward, info):
    """
    Determine if the reward was generated by R_Done_p_i.
    This function should be defined based on how R_Done_p_i is represented in your environment.
    For example:
    return info.get('reward_type') == 'R_Done_p_i'
    """
    # Placeholder implementation; replace with actual condition
    return info.get('reward_type', None) == 'R_Done_p_i'


agents = list(range(num_agents))  # Agent identifiers (e.g., [0, 1, 2, ..., num_agents-1])
agent_now = 0  # Start with the first agent

global_step = 0  # Count total steps across episodes for scheduling updates

for episode in range(M_ep):
    # Reset environment and get initial state
    left_img, right_img, train_props = env.reset()
    left_img   = torch.tensor(left_img,   dtype=torch.float, device=device).unsqueeze(0)  # shape: (1, 1, 84, 84)
    right_img  = torch.tensor(right_img,  dtype=torch.float, device=device).unsqueeze(0) # shape: (1, 1, 84, 84)
    train_props= torch.tensor(train_props,dtype=torch.float, device=device).unsqueeze(0) # shape: (1, 10)

    # Reset Agent_Now at the start of each episode
    agent_now = 0  # Reset to the first agent for each new episode

    for step in range(M_sr):
        global_step += 1

        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action = random.randint(0, 1)  # Replace with the correct action space
        else:
            with torch.no_grad():
                q_values = net(left_img, right_img, train_props)  # shape: (1, num_actions)
                action   = q_values.argmax(dim=1).item()


        # Suppose env.step(action) returns next observations, reward, done, info
        next_left_img, next_right_img, next_train_props, reward, done, info = env.step(action)

        # Convert the next state to tensors
        next_left_img   = torch.tensor(next_left_img,   dtype=torch.float, device=device).unsqueeze(0)
        next_right_img  = torch.tensor(next_right_img,  dtype=torch.float, device=device).unsqueeze(0)
        next_train_props= torch.tensor(next_train_props,dtype=torch.float, device=device).unsqueeze(0)


        if is_RDone_p_i(reward, info):
            agent_now = (agent_now + 1) % num_agents
            # Optionally, log the agent switch
            print(f"Episode {episode}, Step {step}: Switching to Agent {agent_now}")

        replay_buffer.push(
            state=(left_img, right_img, train_props),
            action=action,
            reward=reward,
            next_state=(next_left_img, next_right_img, next_train_props),
            done=done
        )

        # Update the current state
        left_img   = next_left_img
        right_img  = next_right_img
        train_props= next_train_props


        if len(replay_buffer) >= batch_size:
            states, actions, rewards, next_states, dones = replay_buffer.sample()
            # states: (B,) of (left_img, right_img, train_props)
            # next_states: (B,) of (left_img, right_img, train_props)

            # Unpack each part of the state tuple
            left_imgs     = torch.cat([s[0] for s in states], dim=0)      # shape: (B, 1, 84, 84)
            right_imgs    = torch.cat([s[1] for s in states], dim=0)     # shape: (B, 1, 84, 84)
            train_propss  = torch.cat([s[2] for s in states], dim=0)     # shape: (B, 10)

            # Similarly for next states
            next_left_imgs    = torch.cat([ns[0] for ns in next_states], dim=0)
            next_right_imgs   = torch.cat([ns[1] for ns in next_states], dim=0)
            next_train_propss = torch.cat([ns[2] for ns in next_states], dim=0)

            # Current Q-values
            q_values = net(left_imgs, right_imgs, train_propss)  # shape: (B, num_actions)
            # Gather Q-values corresponding to the taken actions
            actions_tensor = actions.unsqueeze(1)  # shape: (B, 1)
            q_values_action = q_values.gather(1, actions_tensor)        # shape: (B, 1)

            # Compute target Q-values
            with torch.no_grad():
                # Double DQN: Use the main network to select the action, and target network to compute the Q-value
                next_q_values = net(next_left_imgs, next_right_imgs, next_train_propss)  # shape: (B, num_actions)
                next_actions = next_q_values.argmax(dim=1, keepdim=True)  # shape: (B, 1)
                next_q_values_target = net_target(next_left_imgs, next_right_imgs, next_train_propss)
                q_values_next = next_q_values_target.gather(1, next_actions)  # shape: (B, 1)
                target = rewards + (1 - dones) * 0.99 * q_values_next  # Assuming discount factor gamma=0.99


            done_mask   = dones.unsqueeze(1)   # shape: (B, 1), float {0.0, 1.0}
            loss = F.mse_loss(q_values_action, target, reduction='none')  # shape: (B,1)
            loss = (loss * done_mask).mean()                              # reduce over batch

            # Gradient descent step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        if global_step % M_up == 0:
            soft_update(net, net_target, xi)
            # Since all agents share the same network, no need to share parameters explicitly
            print(f"Global Step {global_step}: Performed soft update of target network.")

        epsilon = max(epsilon * (1 - epsilon_decay), epsilon_min)
        if done:
            print(f"Episode {episode} finished after {step+1} steps.")
            break
    if (episode + 1) % 100 == 0:
        print(f"Completed {episode + 1} episodes.")


torch.save(net.state_dict(), "trained_q_network.pth")
