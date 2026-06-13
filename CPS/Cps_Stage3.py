# === Cps_Stage3.py (Stage 4 Hybrid: Counterfactuals + Emotions + Symbolic Chaining) ===

import numpy as np
import subprocess
import sys
import os
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor, DummyVecEnv
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import glob

# --- Emotion System ---
class EmotionSystem:
    def __init__(self):
        self.emotions = {"hope": 0.5, "fear": 0.5, "confidence": 0.5}

    def update(self, net_energy, prediction_error):
        self.emotions["hope"] = max(0, min(1, self.emotions["hope"] + 0.01 * net_energy))
        self.emotions["fear"] = max(0, min(1, self.emotions["fear"] + 0.01 * prediction_error))
        self.emotions["confidence"] = max(0, min(1, self.emotions["confidence"] + 0.005 * (net_energy - prediction_error)))

# --- Deep Predictor ---
class DeepPredictor(nn.Module):
    def __init__(self, input_dim):
        super(DeepPredictor, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
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


# --- Symbolic Memory ---
class SymbolicMemory:
    def __init__(self):
        self.symbols = []
        self.chain_memory = []

    def classify(self, net_energy, variance):
        if net_energy > 0 and variance < 1e-5:
            return "Ideal Stability"
        elif net_energy > 0:
            return "High Energy"
        elif variance < 1e-5:
            return "Perfect Stability"
        return "Unstable"

    def add(self, label):
        self.symbols.append(label)
        if len(self.symbols) > 1:
            self.chain_memory.append((self.symbols[-2], label))
        self.symbols = self.symbols[-100:]
        self.chain_memory = self.chain_memory[-100:]

# --- Goal Planner ---
class GoalPlanner:
    def __init__(self):
        self.goals = ["maximize_net_energy", "minimize_variance", "maximize_prediction_accuracy"]
        self.current_goal = None

    def new_goal(self):
        if np.random.rand() < 0.1:
            hybrid = "hybrid_" + np.random.choice(self.goals)
            self.goals.append(hybrid)
            self.current_goal = hybrid
        else:
            self.current_goal = np.random.choice(self.goals)

    def evaluate(self, info):
        score = 0
        if "net_energy" in self.current_goal:
            score += info["net_energy"]
        if "variance" in self.current_goal:
            score -= np.var(info["grid"])
        if "accuracy" in self.current_goal:
            score += -info["prediction_error"]
        return score

# --- Environment ---
class HybridEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, dim=16):
        super().__init__()
        self.dim = dim
        self.grid = np.random.normal(-1e-3, 1e-4, (dim, dim, dim))
        self.action_space = spaces.Box(low=-0.2, high=0.2, shape=(dim, dim, dim), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(dim, dim, dim), dtype=np.float32)
        self.total_input_energy = 0
        self.total_extracted_energy = 0
        self.memory_counter = 0
        self.predictor = DeepPredictor(input_dim=dim**3)
        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=1e-3)
        self.symbolic_memory = SymbolicMemory()
        self.goal_planner = GoalPlanner()
        self.emotions = EmotionSystem()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.grid = np.random.normal(-1e-3, 1e-4, self.grid.shape)
        self.total_input_energy = 0
        self.total_extracted_energy = 0
        self.goal_planner.new_goal()
        return self.grid.copy(), {}

    def step(self, action):
        action = np.clip(action, -0.2, 0.2)
        prev_state = torch.tensor(self.grid.flatten(), dtype=torch.float32)
        self.grid += 1e-7 * action + np.random.normal(0, 1e-6, self.grid.shape)
        input_energy = np.sum((action ** 2) * np.where(self.grid < 0, 1.0 / (1e-10 - self.grid), 1)) * 1e-7
        extracted_energy = 0

        for x, y, z in np.argwhere(self.grid < -1e-4):
            energy = -self.grid[x, y, z]
            gain = np.log1p(energy) ** 2
            self.grid[x, y, z] += gain
            extracted_energy += gain

        self.total_input_energy += input_energy
        self.total_extracted_energy += extracted_energy

        predicted = self.predictor(prev_state)
        true = torch.tensor(self.grid.flatten(), dtype=torch.float32)
        error = torch.mean((predicted - true) ** 2)
        self.optimizer.zero_grad()
        error.backward()
        self.optimizer.step()

        net_energy = self.total_extracted_energy - self.total_input_energy
        variance = np.var(self.grid)
        label = self.symbolic_memory.classify(net_energy, variance)
        self.symbolic_memory.add(label)

        self.emotions.update(net_energy, error.item())

        info = {"grid": self.grid, "net_energy": net_energy, "prediction_error": error.item()}
        goal_score = self.goal_planner.evaluate(info)

        reward = (
            net_energy + goal_score + self.emotions.emotions["hope"] - self.emotions.emotions["fear"]
            - 0.001 * variance
        )
        return self.grid.copy(), reward, False, False, info

    def save_memory(self):
        os.makedirs("memory", exist_ok=True)
        np.savez(f"memory/hybrid_mem_{self.memory_counter}.npz",
                 grid=self.grid,
                 extracted=self.total_extracted_energy,
                 input=self.total_input_energy)
        self.memory_counter += 1

# --- Training ---
def make_env():
    return HybridEnv()

def run():
    env = make_env()
    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecMonitor(vec_env)
    model = PPO("MlpPolicy", vec_env, verbose=1)

    for round in range(500):
        print(f"\n[Round {round+1}] Training...")
        model.learn(total_timesteps=100000)
        env.save_memory()

        if round + 1 == 500:
            print("[Upgrade] Evolving to Stage 5...")
            np.savez("snapshots/stage4_snapshot.npz",
                     final_grid=env.grid,
                     total_extracted_energy=env.total_extracted_energy,
                     total_input_energy=env.total_input_energy)
            subprocess.Popen([sys.executable, "Cps_Stage4.py"])
            break

if __name__ == "__main__":
    run()
