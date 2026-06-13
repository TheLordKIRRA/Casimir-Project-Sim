import numpy as np
import os
import glob
import json
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor, DummyVecEnv
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import pandas as pd
import subprocess
import sys

# --- Simple Predictive Model (Curiosity Module) ---
class SimplePredictor(nn.Module):
    def __init__(self, input_dim):
        super(SimplePredictor, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        x = x.view(1, -1)
        return self.model(x).view(-1)

# --- Self-Preservation Logic ---
class SelfPreservationSystem:
    def __init__(self, error_threshold=0.02, instability_threshold=1e-4):
        self.error_threshold = error_threshold
        self.instability_threshold = instability_threshold
        self.threat_level = 0.0

    def assess_threat(self, error, current_state, previous_state):
        if error > self.error_threshold:
            self.threat_level += 0.05 * (error - self.error_threshold)
        instability = np.mean(np.abs(current_state - previous_state))
        if instability > self.instability_threshold:
            self.threat_level += 0.05 * (instability / self.instability_threshold)
        self.threat_level = max(0.0, min(1.0, self.threat_level))
        return self.threat_level

    def adjust_action(self, action):
        return action * (1.0 - 0.5 * self.threat_level)

# --- Casimir Environment (Evolved) ---
class CasimirEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, dim=16):
        super(CasimirEnv, self).__init__()
        self.dim = dim
        self.grid = np.random.normal(loc=-1e-3, scale=1e-4, size=(dim, dim, dim))
        self.action_space = spaces.Box(low=-0.2, high=0.2, shape=(dim, dim, dim), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(dim, dim, dim), dtype=np.float32)
        self.total_extracted_energy = 0
        self.total_input_energy = 0
        self.time_step = 0
        self.extraction_history = []
        self.input_history = []
        self.episodic_memory = []
        self.predictor = SimplePredictor(input_dim=dim**3)
        self.predictor_optimizer = torch.optim.Adam(self.predictor.parameters(), lr=1e-3)
        self.self_preservation = SelfPreservationSystem()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.grid = np.random.normal(loc=-1e-3, scale=1e-4, size=(self.dim, self.dim, self.dim))
        self.total_extracted_energy = 0
        self.total_input_energy = 0
        self.time_step = 0
        self.extraction_history.clear()
        self.input_history.clear()
        self.episodic_memory.clear()
        return self.grid.copy(), {}

    def cluster_reward(self, coordinates):
        if len(coordinates) == 0:
            return 0
        clustering = DBSCAN(eps=2, min_samples=2).fit(coordinates)
        labels = clustering.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        return n_clusters

    def step(self, action):
        self.time_step += 1
        action = np.clip(action, -0.2, 0.2)
        previous_state = torch.tensor(self.grid.flatten(), dtype=torch.float32)

        action_cost_modifier = np.where(self.grid < 0, 1.0 / (1e-10 - self.grid), 1.0)
        energy_input = np.sum((action**2) * action_cost_modifier) * 1e-7
        self.grid += 1e-7 * action
        self.grid += np.random.normal(0, 1e-6, self.grid.shape)

        neg_energy_coords = np.argwhere(self.grid < -1e-4)
        extracted_energy = 0
        for x, y, z in neg_energy_coords:
            local_energy = -self.grid[x, y, z]
            if local_energy > 0:
                extracted = np.log1p(local_energy) ** 2
                self.grid[x, y, z] += extracted
                extracted_energy += extracted

        self.total_extracted_energy += extracted_energy
        self.total_input_energy += energy_input
        self.extraction_history.append(extracted_energy)
        self.input_history.append(energy_input)

        predicted_state = self.predictor(previous_state)
        true_state = torch.tensor(self.grid.flatten(), dtype=torch.float32)
        prediction_error = torch.mean((predicted_state - true_state) ** 2)
        self.predictor_optimizer.zero_grad()
        prediction_error.backward()
        self.predictor_optimizer.step()

        efficiency = extracted_energy / (energy_input + 1e-10)
        spatial_reward = self.cluster_reward(neg_energy_coords)
        curiosity_bonus = min(prediction_error.item(), 10.0)
        stability_penalty = np.var(self.grid)

        reward = (
            (extracted_energy - energy_input)
            + 0.5 * efficiency
            + 0.1 * spatial_reward
            + 0.01 * curiosity_bonus
            - 0.001 * stability_penalty
            - 1e-6 * self.time_step
        )

        self.episodic_memory.append({
            "grid_snapshot": self.grid.copy(),
            "reward": reward,
            "extracted_energy": extracted_energy,
            "input_energy": energy_input
        })

        return self.grid.copy(), reward, False, False, {
            "net_energy": self.total_extracted_energy - self.total_input_energy
        }

    def save_memory(self):
        os.makedirs("memory", exist_ok=True)
        os.makedirs("semantic", exist_ok=True)
        existing = sorted(glob.glob("memory/Stage2memory_*.npz"))
        index = int(existing[-1].split("_")[-1].split(".")[0]) + 1 if existing else 0
        memory_file = f"memory/Stage2memory_{index}.npz"
        semantic_file = f"semantic/Stage2semantic_{index}.json"

        np.savez(memory_file,
                 extraction_history=np.array(self.extraction_history),
                 input_history=np.array(self.input_history),
                 final_grid=self.grid,
                 total_extracted_energy=self.total_extracted_energy,
                 total_input_energy=self.total_input_energy,
                 episodic_memory=self.episodic_memory)

        semantic_data = {
            "avg_reward": np.mean([e["reward"] for e in self.episodic_memory]),
            "net_energy": self.total_extracted_energy - self.total_input_energy,
            "rounds": self.time_step
        }
        with open(semantic_file, "w") as f:
            json.dump(semantic_data, f, indent=2)

        print(f"[Memory] Saved episodic to {memory_file}")
        print(f"[Memory] Saved semantic to {semantic_file}")

# --- Training ---
def make_env():
    return CasimirEnv()

def continuous_training_loop(max_rounds=5000):
    vec_env = DummyVecEnv([make_env])
    vec_env = VecMonitor(vec_env)
    model = PPO("MlpPolicy", vec_env, verbose=1)

    for round_counter in range(max_rounds):
        print(f"\n=== Training Round {round_counter + 1} ===")
        model.learn(total_timesteps=100000)
        env = vec_env.envs[0]
        net_energy = env.total_extracted_energy - env.total_input_energy
        print(f"[Result] Net Energy: {net_energy:.6f}")
        env.save_memory()

if __name__ == "__main__":
    print("\n[AI-BOOT]: Semantic + Chronological Memory Mode Enabled")
    continuous_training_loop()
