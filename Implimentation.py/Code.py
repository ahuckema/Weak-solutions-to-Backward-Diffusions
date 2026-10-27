import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# ==========================================
# 1. Environment & Free Energy Landscape
# ==========================================
def free_energy(r13, r14):
    """Toy biomolecular free energy landscape."""
    return np.sin(4 * r13) + np.cos(4 * r14) + 1.5 * (r13 - 1.2)**2 + 1.5 * (r14 - 1.2)**2

def in_polytope(r13, r14):
    """Evaluates the linear hyperplanes defining valid polytope P."""
    return (0 <= r13 <= 2 and 0 <= r14 <= 2 and 
            r14 - r13 <= 1 and r13 - r14 <= 1 and 
            r13 + r14 >= 1)

# ==========================================
# 2. Geometry-Aware Q-Learning Agent
# ==========================================
class GeometryAwareQLearning:
    def __init__(self, grid_resolution=41):
        self.grid_x = np.linspace(0, 2, grid_resolution)
        self.grid_y = np.linspace(0, 2, grid_resolution)
        
        self.states = []
        for i in range(grid_resolution):
            for j in range(grid_resolution):
                if in_polytope(self.grid_x[i], self.grid_y[j]):
                    self.states.append((i, j))
                    
        self.state_idx = {s: idx for idx, s in enumerate(self.states)}
        self.actions = [(1, 0), (-1, 0), (0, 1), (0, -1), (0, 0)]
        self.q_table = np.zeros((len(self.states), len(self.actions)))

    def step(self, state, action_idx):
        i, j = state
        di, dj = self.actions[action_idx]
        new_i, new_j = i + di, j + dj

        if (new_i, new_j) in self.state_idx:
            next_state = (new_i, new_j)
            r13, r14 = self.grid_x[new_i], self.grid_y[new_j]
            reward = -free_energy(r13, r14)
        else:
            next_state = state
            r13, r14 = self.grid_x[i], self.grid_y[j]
            reward = -free_energy(r13, r14) - 10.0

        return next_state, reward

    def train(self, total_episodes=2000, episode_length=100, alpha=0.1, gamma=0.99, epsilon=0.2):
        reward_history = []
        
        for ep in range(total_episodes):
            state = self.states[np.random.choice(len(self.states))]
            ep_reward = 0
            
            for _ in range(episode_length):
                s_idx = self.state_idx[state]
                if np.random.rand() < epsilon:
                    action = np.random.choice(len(self.actions))
                else:
                    action = np.argmax(self.q_table[s_idx])

                next_state, reward = self.step(state, action)
                ns_idx = self.state_idx[next_state]

                best_next_q = np.max(self.q_table[ns_idx])
                td_target = reward + gamma * best_next_q
                self.q_table[s_idx, action] += alpha * (td_target - self.q_table[s_idx, action])
                
                state = next_state
                ep_reward += reward
                
            reward_history.append(ep_reward)
        return reward_history

# ==========================================
# 3. Geometry-Unaware PPO Networks & Agent
# ==========================================
class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 2)  # Continuous mean vector (dr13, dr14)
        )
        self.critic = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.log_std = nn.Parameter(torch.zeros(2) - 1.0) # Initial std ~ 0.36

    def get_action(self, state):
        mean = torch.tanh(self.actor(state)) * 0.05 # Step size scale
        std = torch.exp(self.log_std)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, dist.entropy().sum(dim=-1)

    def evaluate(self, state, action):
        mean = torch.tanh(self.actor(state)) * 0.05
        std = torch.exp(self.log_std)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        value = self.critic(state).squeeze(-1)
        return log_prob, value, dist.entropy().sum(dim=-1)

class GeometryUnawarePPO:
    def __init__(self, lr=3e-4, gamma=0.99, clip_eps=0.2):
        self.policy = ActorCritic()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_eps = clip_eps

    def train(self, total_episodes=2000, episode_length=100):
        reward_history = []

        for ep in range(total_episodes):
            # Uniformly sample valid initial state
            r13 = np.random.uniform(0.0, 2.0)
            r14 = np.random.uniform(0.0, 2.0)
            while not in_polytope(r13, r14):
                r13 = np.random.uniform(0.0, 2.0)
                r14 = np.random.uniform(0.0, 2.0)

            states, actions, log_probs, rewards, values, masks = [], [], [], [], [], []
            ep_reward = 0

            for _ in range(episode_length):
                state_t = torch.tensor([r13, r14], dtype=torch.float32)
                action_t, log_prob_t, _ = self.policy.get_action(state_t)
                value_t = self.policy.critic(state_t)

                dr13, dr14 = action_t.numpy()
                next_r13, next_r14 = r13 + dr13, r14 + dr14

                # Environment step with out-of-bounds rejection penalty
                if in_polytope(next_r13, next_r14):
                    r13, r14 = next_r13, next_r14
                    reward = -free_energy(r13, r14)
                else:
                    # Boundary violation penalty
                    reward = -free_energy(r13, r14) - 10.0 

                states.append(state_t)
                actions.append(action_t)
                log_probs.append(log_prob_t)
                rewards.append(reward)
                values.append(value_t.squeeze(-1))
                masks.append(1.0)
                ep_reward += reward

            reward_history.append(ep_reward)

            # PPO Update Step
            states_t = torch.stack(states)
            actions_t = torch.stack(actions)
            old_log_probs_t = torch.stack(log_probs).detach()
            rewards_t = torch.tensor(rewards, dtype=torch.float32)
            
            # Compute Returns
            returns = []
            discounted_sum = 0
            for r in reversed(rewards):
                discounted_sum = r + self.gamma * discounted_sum
                returns.insert(0, discounted_sum)
            returns_t = torch.tensor(returns, dtype=torch.float32)
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

            # Update Policy over multiple epochs
            for _ in range(4):
                new_log_probs, state_values, entropy = self.policy.evaluate(states_t, actions_t)
                ratios = torch.exp(new_log_probs - old_log_probs_t)
                advantages = returns_t - state_values.detach()

                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = nn.MSELoss()(state_values, returns_t)

                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy.mean()
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return reward_history

# ==========================================
# 4. Execution & Convergence Comparison
# ==========================================
if __name__ == "__main__":
    TOTAL_EPISODES = 1500
    EPISODE_LENGTH = 100

    print("Training Geometry-Aware Q-Learning...")
    q_agent = GeometryAwareQLearning()
    q_rewards = q_agent.train(total_episodes=TOTAL_EPISODES, episode_length=EPISODE_LENGTH)

    print("Training Geometry-Unaware PPO...")
    ppo_agent = GeometryUnawarePPO()
    ppo_rewards = ppo_agent.train(total_episodes=TOTAL_EPISODES, episode_length=EPISODE_LENGTH)

    # Calculate Moving Average (Window = 50)
    window = 50
    q_smooth = np.convolve(q_rewards, np.ones(window)/window, mode='valid')
    ppo_smooth = np.convolve(ppo_rewards, np.ones(window)/window, mode='valid')

    # Plot Convergence Benchmark
    plt.figure(figsize=(10, 6))
    plt.plot(q_smooth, label="Geometry-Aware Q-Learning (Constrained Action Graph)", color="blue", linewidth=2)
    plt.plot(ppo_smooth, label="Geometry-Unaware PPO (Continuous Gaussian Policy + Penalty)", color="orange", linewidth=2)
    plt.axhline(y=-EPISODE_LENGTH * (-1.78), color="green", linestyle="--", label="Theoretical Optimal Return")

    plt.xlabel("Training Episodes")
    plt.ylabel("Cumulative Episode Return")
    plt.title("Convergence Speed: Geometry-Aware RL vs. Unaware PPO")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()