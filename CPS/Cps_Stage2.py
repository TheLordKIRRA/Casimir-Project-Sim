# === Cps_Stage2.py (Stage 2 Upgrade Pack: Symbolic Thought + Self-Reflection + Consciousness Tracker + Auto-Stage3 + Self-Preservation) ===

import numpy as np
import os
import glob
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor, DummyVecEnv
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import pandas as pd
import random
import csv
import subprocess
import sys
from collections import Counter

# --- Simple Predictive Model ---
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

# --- Symbolic Memory and Goal Planner ---
class SymbolicMemorySystem:
    def __init__(self):
        self.symbolic_memory = []

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
        if len(self.symbolic_memory) > 100:
            self.symbolic_memory = self.symbolic_memory[-100:]

    def summarize(self):
        return {label: self.symbolic_memory.count(label) for label in set(self.symbolic_memory)}

class GoalPlanner:
    def __init__(self):
        self.current_goal = None
        self.goal_counter = 0

    def new_goal(self):
        goals = [
            "maximize_stability",
            "maximize_net_energy",
            "maximize_curiosity",
            "maximize_cluster_density"
        ]
        self.current_goal = random.choice(goals)
        self.goal_counter = 0

    def evaluate_goal_reward(self, info):
        reward = 0
        if self.current_goal == "maximize_stability":
            reward = -np.var(info["grid"])
        elif self.current_goal == "maximize_net_energy":
            reward = info["net_energy"]
        elif self.current_goal == "maximize_curiosity":
            reward = info["prediction_error"]
        elif self.current_goal == "maximize_cluster_density":
            reward = info["clusters"]
        return reward

# --- Consciousness Tracker ---
class ConsciousnessTracker:
    def __init__(self):
        self.symbolic_complexity = []
        self.goal_switches = []
        self.last_goal = None
        self.rounds = 0
        os.makedirs("consciousness_logs", exist_ok=True)

    def update(self, env):
        symbolic_counts = Counter(env.symbolic_system.symbolic_memory)
        complexity_score = len(symbolic_counts)

        current_goal = env.goal_planner.current_goal
        goal_switched = 0
        if self.last_goal is not None and current_goal != self.last_goal:
            goal_switched = 1
        self.last_goal = current_goal

        self.symbolic_complexity.append(complexity_score)
        self.goal_switches.append(goal_switched)
        self.rounds += 1

        if self.rounds % 100 == 0:
            with open("consciousness_logs/consciousness_tracker.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Round", "Symbolic Complexity", "Goal Switches"])
                for i in range(self.rounds):
                    writer.writerow([i+1, self.symbolic_complexity[i], self.goal_switches[i]])

    def plot(self):
        rounds = list(range(1, self.rounds + 1))
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))
        axs[0].plot(rounds, self.symbolic_complexity, label="Symbolic Complexity")
        axs[0].set_ylabel("Unique Symbols")
        axs[0].legend()

        axs[1].plot(rounds, np.cumsum(self.goal_switches), label="Cumulative Goal Switches", color="orange")
        axs[1].set_xlabel("Training Rounds")
        axs[1].set_ylabel("Goal Changes")
        axs[1].legend()

        plt.suptitle("Consciousness Emergence Tracker")
        plt.tight_layout()
        plt.show()

# --- Diary Self-Reflection ---
class Diary:
    def __init__(self):
        self.entries = []
        os.makedirs("diary_logs", exist_ok=True)

    def write_entry(self, round_number, env):
        dominant_symbol = max(set(env.symbolic_system.symbolic_memory), key=env.symbolic_system.symbolic_memory.count)
        entry = f"Round {round_number}: I felt mostly '{dominant_symbol}'. My current goal is '{env.goal_planner.current_goal}'."
        self.entries.append(entry)

        with open("diary_logs/diary.txt", "a") as f:
            f.write(entry + "\n")

# --- Snapshot and Auto-Transfer to Stage 3 ---
def snapshot_and_transfer(env, model, snapshot_name="stage3_snapshot.npz"):
    os.makedirs("snapshots", exist_ok=True)
    snapshot_path = os.path.join("snapshots", snapshot_name)

    np.savez(snapshot_path,
             final_grid=env.grid,
             extraction_history=np.array(env.extraction_history),
             input_history=np.array(env.input_history),
             total_extracted_energy=env.total_extracted_energy,
             total_input_energy=env.total_input_energy)

    print(f"\n[Snapshot] Saved snapshot for Stage 3: {snapshot_path}")
    print("[Upgrade] Launching Cps_Stage3.py...")

    subprocess.Popen([sys.executable, "Cps_Stage3.py", snapshot_path])
    sys.exit(0)

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
        self.time_step = 0
        self.extraction_history = []
        self.input_history = []
        self.memory_counter = 0

        self.predictor = SimplePredictor(input_dim=dim*dim*dim)
        self.predictor_optimizer = torch.optim.Adam(self.predictor.parameters(), lr=1e-3)

        self.symbolic_system = SymbolicMemorySystem()
        self.goal_planner = GoalPlanner()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.grid = np.random.normal(loc=-1e-3, scale=1e-4, size=(self.dim, self.dim, self.dim))
        self.total_extracted_energy = 0
        self.total_input_energy = 0
        self.time_step = 0
        self.extraction_history.clear()
        self.input_history.clear()
        self.goal_planner.new_goal()
        return self.grid.copy(), {}

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
        spatial_reward = len(neg_energy_coords)
        curiosity_bonus = prediction_error.item()
        stability_penalty = np.var(self.grid)

        info = {
            "grid": self.grid,
            "extracted_energy": extracted_energy,
            "input_energy": energy_input,
            "net_energy": self.total_extracted_energy - self.total_input_energy,
            "efficiency": efficiency,
            "clusters": spatial_reward,
            "prediction_error": prediction_error.item()
        }

        symbolic_label = self.symbolic_system.classify_experience(info["net_energy"], stability_penalty)
        self.symbolic_system.add_symbol(symbolic_label)

        goal_reward = self.goal_planner.evaluate_goal_reward(info)

        reward = (
            (extracted_energy - energy_input)
            + 0.5 * efficiency
            + 0.01 * curiosity_bonus
            + 0.01 * goal_reward
            - 0.001 * stability_penalty
            - 1e-6 * self.time_step
        )

        terminated = False
        truncated = False
        return self.grid.copy(), reward, terminated, truncated, info

    def save_memory(self):
        os.makedirs("memory", exist_ok=True)
        memory_id = f"memory_{self.memory_counter}"
        np.savez(f"memory/{memory_id}.npz",
                 extraction_history=np.array(self.extraction_history),
                 input_history=np.array(self.input_history),
                 final_grid=self.grid,
                 total_extracted_energy=self.total_extracted_energy,
                 total_input_energy=self.total_input_energy)
        self.memory_counter += 1

    def load_best_memory(self):
        os.makedirs("memory", exist_ok=True)
        memory_files = glob.glob("memory/*.npz")
        if not memory_files:
            return

        best_file = None
        best_net_energy = -np.inf

        for memfile in memory_files:
            data = np.load(memfile)
            net_energy = float(data["total_extracted_energy"]) - float(data["total_input_energy"])
            if net_energy > best_net_energy:
                best_net_energy = net_energy
                best_file = memfile

        if best_file:
            data = np.load(best_file)
            self.extraction_history = data["extraction_history"].tolist()
            self.input_history = data["input_history"].tolist()
            self.grid = data["final_grid"]
            self.total_extracted_energy = float(data["total_extracted_energy"])
            self.total_input_energy = float(data["total_input_energy"])

# --- Setup ---
def make_env():
    env = CasimirEnv()
    env.load_best_memory()
    return env

# --- Continuous Training Loop ---
def continuous_training_loop(max_rounds=5000):
    round_counter = 0
    tracker = ConsciousnessTracker()
    diary = Diary()

    while round_counter < max_rounds:
        print(f"\n=== Training Round {round_counter + 1} ===")
        env = make_env()
        vec_env = DummyVecEnv([lambda: env])
        vec_env = VecMonitor(vec_env)
        model = PPO("MlpPolicy", vec_env, verbose=0)
        model.learn(total_timesteps=100000)

        final_state = vec_env.envs[0].grid
        extracted_total = vec_env.envs[0].total_extracted_energy
        input_total = vec_env.envs[0].total_input_energy
        net_energy = extracted_total - input_total

        print(f"[Result] Net Energy: {net_energy:.6f}")

        env.save_memory()
        tracker.update(env)

        if round_counter % 500 == 0:
            diary.write_entry(round_counter, env)

        # --- Auto-Transfer to Stage 3 after 2500 rounds ---
        if round_counter + 1 == 2500:
            snapshot_and_transfer(env, model)

        round_counter += 1

    tracker.plot()

# --- Memory Plot ---
def plot_memory_evolution():
    memory_files = sorted(glob.glob("memory/*.npz"))
    if not memory_files:
        return

    memories = []
    for memfile in memory_files:
        data = np.load(memfile)
        extracted = np.sum(data["extraction_history"])
        input_energy = np.sum(data["input_history"])
        net_energy = extracted - input_energy
        memories.append({
            "Memory File": os.path.basename(memfile),
            "Extracted Energy": extracted,
            "Input Energy": input_energy,
            "Net Energy": net_energy
        })

    df = pd.DataFrame(memories)
    df = df.sort_values(by="Net Energy", ascending=False)

    plt.figure(figsize=(10, 6))
    plt.bar(df["Memory File"], df["Net Energy"], color='green')
    plt.xlabel("Memory")
    plt.ylabel("Net Energy")
    plt.title("Memory Evolution: Net Energy")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()
    print(df)

# --- Start ---
continuous_training_loop()
plot_memory_evolution()
