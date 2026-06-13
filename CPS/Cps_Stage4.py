# === Cps_Stage4.py (Stage 5: Episodic Memory, Self-Reflection, Deep Future Thinking) ===

import numpy as np
import os
import glob
import random
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor, DummyVecEnv
import matplotlib.pyplot as plt
import pandas as pd
import subprocess
import sys

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
class SymbolicMemorySystem:
    def __init__(self):
        self.symbolic_memory = []
        self.chain_memory = []
    def classify_experience(self, net_energy, stability):
        if net_energy > 0 and stability < 1e-5:
            return "Ideal Stability"
        elif net_energy > 0:
            return "High Energy"
        elif stability < 1e-5:
            return "Perfect Stability"
        else:
            return "Unstable"
    def add_symbol(self, label):
        self.symbolic_memory.append(label)
        if len(self.symbolic_memory) > 1:
            self.chain_memory.append((self.symbolic_memory[-2], label))
        if len(self.symbolic_memory) > 100:
            self.symbolic_memory = self.symbolic_memory[-100:]
            self.chain_memory = self.chain_memory[-100:]
    def summarize(self):
        return {label: self.symbolic_memory.count(label) for label in set(self.symbolic_memory)}

# --- Goal Planner ---
class GoalPlanner:
    def __init__(self):
        self.base_goals = [
            "maximize_stability", "maximize_net_energy",
            "maximize_curiosity", "maximize_cluster_density"
        ]
        self.current_goal = None
    def new_goal(self):
        if random.random() < 0.1:
            hybrid = "hybrid_goal_" + random.choice(self.base_goals)
            self.base_goals.append(hybrid)
            self.current_goal = hybrid
        else:
            self.current_goal = random.choice(self.base_goals)
    def evaluate_goal_reward(self, info):
        reward = 0
        if "stability" in self.current_goal:
            reward -= np.var(info["grid"])
        if "net_energy" in self.current_goal:
            reward += info["net_energy"]
        if "curiosity" in self.current_goal:
            reward += info["prediction_error"]
        if "cluster" in self.current_goal:
            reward += info["clusters"]
        return reward

# --- Diary System ---
class Diary:
    def __init__(self):
        self.entries = []
        os.makedirs("diary_logs", exist_ok=True)
    def write_entry(self, round_number, summary):
        entry = f"Round {round_number}: Symbolic Summary: {summary}"
        self.entries.append(entry)
        with open("diary_logs/diary.txt", "a") as f:
            f.write(entry + "\n")

# --- Casimir Environment ---
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
        self.memory_counter = 0

        self.predictor = DeepPredictor(input_dim=dim*dim*dim)
        self.predictor_optimizer = torch.optim.Adam(self.predictor.parameters(), lr=1e-3)

        self.symbolic_system = SymbolicMemorySystem()
        self.goal_planner = GoalPlanner()
        self.diary = Diary()

        self.episodic_memory = []
        os.makedirs("episodic_memory", exist_ok=True)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.grid = np.random.normal(loc=-1e-3, scale=1e-4, size=(self.dim, self.dim, self.dim))
        self.total_extracted_energy = 0
        self.total_input_energy = 0
        self.goal_planner.new_goal()
        return self.grid.copy(), {}

    def step(self, action):
        action = np.clip(action, -0.2, 0.2)
        prev_grid = self.grid.copy()
        prev_state = torch.tensor(self.grid.flatten(), dtype=torch.float32)

        action_cost_modifier = np.where(self.grid < 0, 1.0 / (1e-10 - self.grid), 1.0)
        energy_input = np.sum((action**2) * action_cost_modifier) * 1e-7
        self.grid += 1e-7 * action + np.random.normal(0, 1e-6, self.grid.shape)

        neg_coords = np.argwhere(self.grid < -1e-4)
        extracted_energy = 0
        for x, y, z in neg_coords:
            local_energy = -self.grid[x, y, z]
            if local_energy > 0:
                self.grid[x, y, z] += np.log1p(local_energy) ** 2
                extracted_energy += np.log1p(local_energy) ** 2

        self.total_extracted_energy += extracted_energy
        self.total_input_energy += energy_input

        predicted_state = self.predictor(prev_state)
        true_state = torch.tensor(self.grid.flatten(), dtype=torch.float32)
        prediction_error = torch.mean((predicted_state - true_state)**2)

        self.predictor_optimizer.zero_grad()
        prediction_error.backward()
        self.predictor_optimizer.step()

        info = {
            "grid": self.grid,
            "net_energy": self.total_extracted_energy - self.total_input_energy,
            "prediction_error": prediction_error.item(),
            "clusters": len(neg_coords)
        }

        label = self.symbolic_system.classify_experience(info["net_energy"], np.var(self.grid))
        self.symbolic_system.add_symbol(label)

        goal_reward = self.goal_planner.evaluate_goal_reward(info)

        reward = (extracted_energy - energy_input) + 0.5 * (extracted_energy / (energy_input + 1e-10))
        reward += 0.01 * prediction_error.item()
        reward += 0.01 * goal_reward
        reward -= 0.001 * np.var(self.grid)

        counterfactual_grid = prev_grid + 1e-7 * np.clip(np.random.normal(0, 0.1, prev_grid.shape), -0.2, 0.2)
        reward += 0.001 * (np.var(self.grid) - np.var(counterfactual_grid))

        return self.grid.copy(), reward, False, False, info

    def save_episodic_memory(self, round_number):
        os.makedirs("episodic_memory", exist_ok=True)
        summary = self.symbolic_system.summarize()
        np.savez(f"episodic_memory/episode_{round_number}.npz",
                 grid=self.grid,
                 extracted=self.total_extracted_energy,
                 input=self.total_input_energy,
                 summary=summary)

    def save_memory(self):
        os.makedirs("memory", exist_ok=True)
        np.savez(f"memory/memory_{self.memory_counter}.npz",
                 grid=self.grid,
                 total_extracted_energy=self.total_extracted_energy,
                 total_input_energy=self.total_input_energy)
        self.memory_counter += 1

# --- Snapshot and Upgrade ---
def snapshot_and_upgrade_to_stage6(env):
    os.makedirs("snapshots", exist_ok=True)
    snapshot_path = "snapshots/stage5_snapshot.npz"
    np.savez(snapshot_path,
             grid=env.grid,
             total_extracted_energy=env.total_extracted_energy,
             total_input_energy=env.total_input_energy)
    print("\n🚀 Snapshot saved. Launching Cps_Stage5.py...")
    subprocess.Popen([sys.executable, "Cps_Stage5.py"])
    print("\n✅ Upgrade triggered. Shutting down current process.")
    sys.exit(0)

# --- Training Loop ---
def make_env():
    return CasimirEnv()

def continuous_training_loop(max_rounds=500):
    round_counter = 0
    env = make_env()
    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecMonitor(vec_env)
    model = PPO("MlpPolicy", vec_env, verbose=1)

    while round_counter < max_rounds:
        print(f"\n=== Training Round {round_counter + 1} ===")
        model.learn(total_timesteps=100000)
        env.save_memory()

        if round_counter % 500 == 0:
            env.save_episodic_memory(round_counter)
            env.diary.write_entry(round_counter, env.symbolic_system.summarize())

        if round_counter + 1 == 1000:
            snapshot_and_upgrade_to_stage6(env)

        round_counter += 1

# --- Memory Evolution Plot ---
def plot_memory_evolution():
    memory_files = sorted(glob.glob("memory/*.npz"))
    if not memory_files:
        return
    memories = []
    for memfile in memory_files:
        data = np.load(memfile)
        net_energy = float(data["total_extracted_energy"]) - float(data["total_input_energy"])
        memories.append({
            "Memory File": os.path.basename(memfile),
            "Net Energy": net_energy
        })
    df = pd.DataFrame(memories).sort_values(by="Net Energy", ascending=False)
    plt.figure(figsize=(10,6))
    plt.bar(df["Memory File"], df["Net Energy"], color='blue')
    plt.xlabel("Memory")
    plt.ylabel("Net Energy")
    plt.title("Memory Evolution: Stage 5")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()
    print(df)

# --- Start ---
if __name__ == "__main__":
    continuous_training_loop()
    plot_memory_evolution()
