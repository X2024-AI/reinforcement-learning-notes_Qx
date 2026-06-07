import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

# Hyperparameters
ENV_NAME = "CartPole-v1"
LEARNING_RATE = 3e-4
TOTAL_TIMESTEPS = 100000
TRAIN_COEFFICIENTS = {"value": 0.5, "entropy": 0.01}

# PPO Specific Hyperparameters
ROLLOUT_STEPS = 2048      # N * T (Total data pool size per iteration)
MINIBATCH_SIZE = 64       # M
PPO_EPOCHS = 10           # K
CLIP_EPSILON = 0.2        # epsilon
GAMMA = 0.99              # Discount factor
GAE_LAMBDA = 0.95         # Smoothing factor for GAE

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim)
        )
        
    def get_action_distribution(self, state):
        logits = self.network(state)
        return Categorical(logits=logits)

class CriticNetwork(nn.Module):
    def __init__(self, state_dim):
        super(CriticNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
    def forward(self, state):
        return self.network(state)

class PPOAgent:
    def __init__(self, state_dim, action_dim):
        self.actor = ActorNetwork(state_dim, action_dim)
        self.critic = CriticNetwork(state_dim)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LEARNING_RATE)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LEARNING_RATE)

    def train_step(self, states, actions, old_log_probs, returns, advantages):
        # Convert data to PyTorch tensors
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(np.array(actions), dtype=torch.long)
        old_log_probs = torch.tensor(np.array(old_log_probs), dtype=torch.float32)
        returns = torch.tensor(np.array(returns), dtype=torch.float32).unsqueeze(1)
        advantages = torch.tensor(np.array(advantages), dtype=torch.float32)
        
        # Normalize advantages for training stability
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        dataset_size = len(states)
        
        # K Epochs Loop
        for _ in range(PPO_EPOCHS):
            permutation = torch.randperm(dataset_size)
            
            # Minibatch Loop
            for start_idx in range(0, dataset_size, MINIBATCH_SIZE):
                batch_indices = permutation[start_idx:start_idx + MINIBATCH_SIZE]
                
                b_states = states[batch_indices]
                b_actions = actions[batch_indices]
                b_old_log_probs = old_log_probs[batch_indices]
                b_returns = returns[batch_indices]
                b_advantages = advantages[batch_indices]
                
                # Actor Loss Calculation
                dist = self.actor.get_action_distribution(b_states)
                new_log_probs = dist.log_prob(b_actions)
                entropy = dist.entropy().mean()
                
                # Probability Ratio r_t(theta)
                ratios = torch.exp(new_log_probs - b_old_log_probs)
                
                # Clipped Surrogate Objective
                surr1 = ratios * b_advantages
                surr2 = torch.clamp(ratios, 1.0 - CLIP_EPSILON, 1.0 + CLIP_EPSILON) * b_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Critic Loss Calculation (MSE)
                state_values = self.critic(b_states)
                critic_loss = nn.MSELoss()(state_values, b_returns)
                
                # Total Objective Optimization
                self.actor_optimizer.zero_grad()
                actor_total_loss = actor_loss - TRAIN_COEFFICIENTS["entropy"] * entropy
                actor_total_loss.backward()
                self.actor_optimizer.step()
                
                self.critic_optimizer.zero_grad()
                critic_total_loss = TRAIN_COEFFICIENTS["value"] * critic_loss
                critic_total_loss.backward()
                self.critic_optimizer.step()

def main():
    env = gym.make(ENV_NAME)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = PPOAgent(state_dim, action_dim)
    
    state, _ = env.reset()
    global_step = 0
    episode_rewards = []
    current_episode_reward = 0
    
    while global_step < TOTAL_TIMESTEPS:
        # 1. Storage buffers for Rollout Data Pool (N * T)
        b_states, b_actions, b_log_probs, b_rewards, b_dones, b_values = [], [], [], [], [], []
        
        # 2. Data Collection Phase
        for _ in range(ROLLOUT_STEPS):
            global_step += 1
            state_t = torch.tensor(state, dtype=torch.float32)
            
            with torch.no_grad():
                dist = agent.actor.get_action_distribution(state_t)
                action = dist.sample().item()
                log_prob = dist.log_prob(torch.tensor(action)).item()
                value = agent.critic(state_t).item()
                
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            current_episode_reward += reward
            
            b_states.append(state)
            b_actions.append(action)
            b_log_probs.append(log_prob)
            b_rewards.append(reward)
            b_dones.append(done)
            b_values.append(value)
            
            state = next_state
            if done:
                state, _ = env.reset()
                episode_rewards.append(current_episode_reward)
                if len(episode_rewards) % 10 == 0:
                    print(f"Step: {global_step} | Avg Reward (Last 10 Eps): {np.mean(episode_rewards[-10:]):.2f}")
                current_episode_reward = 0
                
        # Compute bootstrap value for the final unfinished step
        with torch.no_grad():
            next_value = agent.critic(torch.tensor(state, dtype=torch.float32)).item() if not done else 0
            
        # 3. Generalized Advantage Estimation (GAE) Calculation
        b_advantages = np.zeros_like(b_rewards, dtype=np.float32)
        b_returns = np.zeros_like(b_rewards, dtype=np.float32)
        last_gae_lam = 0
        
        for t in reversed(range(ROLLOUT_STEPS)):
            next_non_terminal = 1.0 - b_dones[t]
            next_val = b_values[t + 1] if t < ROLLOUT_STEPS - 1 else next_value
            
            # Delta (temporal difference error)
            delta = b_rewards[t] + GAMMA * next_val * next_non_terminal - b_values[t]
            
            # GAE tracking formula
            b_advantages[t] = last_gae_lam = delta + GAMMA * GAE_LAMBDA * next_non_terminal * last_gae_lam
            b_returns[t] = b_advantages[t] + b_values[t]
            
        # 4. Policy Optimization Phase
        agent.train_step(b_states, b_actions, b_log_probs, b_returns, b_advantages)

    env.close()

if __name__ == "__main__":
    main()