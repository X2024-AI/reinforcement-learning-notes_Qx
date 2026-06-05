import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython import display
import time

# 1. 定义策略网络 (Policy Network)
class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=-1)

# 2. 定义 REINFORCE 算法智能体
class REINFORCE:
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99):
        self.policy_net = PolicyNet(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.gamma = gamma
        
        self.log_probs = []
        self.rewards = []

    def choose_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.policy_net(state)
        
        m = Categorical(probs)
        action = m.sample()
        
        self.log_probs.append(m.log_prob(action))
        return action.item()

    def update(self):
        discounted_rewards = []
        G = 0
        
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            discounted_rewards.insert(0, G)
            
        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32)
        
        # 标准化
        if len(discounted_rewards) > 1:
            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)
        
        policy_loss = []
        for log_prob, G_t in zip(self.log_probs, discounted_rewards):
            policy_loss.append(-log_prob * G_t)
            
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        
        self.log_probs = []
        self.rewards = []

# 3. 训练主循环（带可视化）
class Visualizer:
    def __init__(self):
        self.episodes = []
        self.rewards = []
        self.running_avg = []
        
    def update_plot(self, episode, reward, avg_reward):
        self.episodes.append(episode)
        self.rewards.append(reward)
        self.running_avg.append(avg_reward)
        
        plt.clf()
        plt.plot(self.episodes, self.rewards, 'b-', alpha=0.3, label='Episode Reward')
        plt.plot(self.episodes, self.running_avg, 'r-', linewidth=2, label='Running Average (20 episodes)')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('REINFORCE Algorithm on CartPole-v1')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.pause(0.01)

def train(render=False):
    env = gym.make('CartPole-v1', render_mode='human' if render else None)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = REINFORCE(state_dim, action_dim, lr=0.001)
    visualizer = Visualizer()
    
    # 用于存储最佳模型
    best_reward = 0
    rewards_history = []
    
    print("开始训练 REINFORCE 算法...")
    print("="*50)
    
    for episode in range(500):
        state, _ = env.reset()
        episode_reward = 0
        step_count = 0
        
        while True:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            agent.rewards.append(reward)
            state = next_state
            episode_reward += reward
            step_count += 1
            
            if terminated or truncated:
                break
                
        agent.update()
        rewards_history.append(episode_reward)
        
        # 计算运行平均
        running_avg = np.mean(rewards_history[-20:]) if len(rewards_history) >= 20 else np.mean(rewards_history)
        
        # 更新可视化
        visualizer.update_plot(episode + 1, episode_reward, running_avg)
        
        # 保存最佳模型
        if episode_reward > best_reward:
            best_reward = episode_reward
            torch.save(agent.policy_net.state_dict(), 'best_policy_net.pth')
        
        # 打印进度
        if (episode + 1) % 20 == 0:
            print(f"Episode {episode+1:3d} | Reward: {episode_reward:4.0f} | "
                  f"Avg(20): {running_avg:.1f} | Best: {best_reward:.0f} | "
                  f"Steps: {step_count:3d}")
            
            if running_avg >= 450:
                print(f"\n🎉 成功解决任务！在 Episode {episode+1} 达到平均奖励 450+")
                print(f"最佳模型已保存为 'best_policy_net.pth'")
                break
    
    env.close()
    return agent, rewards_history

# 测试训练好的模型
def test_trained_model(episodes=5):
    env = gym.make('CartPole-v1', render_mode='human')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = REINFORCE(state_dim, action_dim)
    agent.policy_net.load_state_dict(torch.load('best_policy_net.pth'))
    agent.policy_net.eval()
    
    print("\n" + "="*50)
    print("测试训练好的模型...")
    
    for ep in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        step_count = 0
        
        while True:
            # 测试时不使用随机采样，直接选概率最大的动作
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)
            with torch.no_grad():
                probs = agent.policy_net(state_tensor)
            action = torch.argmax(probs).item()
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            state = next_state
            episode_reward += reward
            step_count += 1
            
            if terminated or truncated:
                print(f"测试 Episode {ep+1}: Reward = {episode_reward:.0f}, Steps = {step_count}")
                break
            time.sleep(0.01)  # 减慢速度，便于观察
    
    env.close()

if __name__ == "__main__":
    # 训练模型（不渲染，加快速度）
    trained_agent, history = train(render=False)
    
    # 显示最终结果统计
    plt.figure(figsize=(10, 6))
    plt.plot(history)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('REINFORCE Algorithm Training History')
    plt.grid(True, alpha=0.3)
    
    # 计算移动平均
    window = 20
    moving_avg = np.convolve(history, np.ones(window)/window, mode='valid')
    plt.plot(range(window-1, len(history)), moving_avg, 'r', linewidth=2, label='Moving Average')
    plt.legend()
    plt.show()
    
    # 询问是否测试可视化
    print("\n是否要观看训练好的模型演示？(y/n)")
    choice = input().lower()
    if choice == 'y':
        test_trained_model(episodes=3)