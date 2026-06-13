# === AICps.py (Full Evolutionary Model + Karma + Prediction + Emotion) ===

import numpy as np
import os
import glob
import json
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor, DummyVecEnv
import torch
import torch.nn as nn
import subprocess
import sys
import atexit

# --- Create save folder if needed ---
os.makedirs("checkpoints", exist_ok=True)

# --- Safe exit handler ---
def save_on_exit():
    global model, round_counter
    if 'model' in globals() and 'round_counter' in globals():
        print(f"[Exit] Saving PPO model and round_counter={round_counter}")
        model.save("checkpoints/latest_ppo_model")
        with open("checkpoints/last_round.json", "w") as f:
            json.dump({"round_counter": round_counter}, f)

atexit.register(save_on_exit)

# --- Predictive Curiosity Model ---
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

# --- Advanced Casimir Environment ---
class CasimirEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, dim=16):
        super(CasimirEnv, self).__init__()
        self.dim = dim
        self.grid = self._init_grid()
        self.action_space = spaces.Box(low=-0.2, high=0.2, shape=(dim, dim, dim), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(dim, dim, dim), dtype=np.float32)
        self.total_extracted_energy = 0
        self.total_input_energy = 0
        self.time_step = 0
        self.extraction_history = []
        self.input_history = []
        self.memory_counter = 0
        self.personality = {"curiosity": 0.5, "focus": 0.5, "adaptability": 0.5}
        self.emotional_state = {"hope": 0.5, "stress": 0.5}
        self.karma_traits = self.load_karma_profile()
        self.predictor = SimplePredictor(dim ** 3)
        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=1e-3)

    def _init_grid(self):
        base_noise = np.random.normal(loc=-1e-3, scale=1e-4, size=(self.dim, self.dim, self.dim))
        quantum_fluctuation = np.random.normal(0.0, 5e-5, size=(self.dim, self.dim, self.dim))
        return base_noise + quantum_fluctuation

    def load_karma_profile(self):
        if os.path.exists("karma_profile.json"):
            with open("karma_profile.json", "r") as f:
                print("[Karma] karma_profile.json loaded.")
                return json.load(f)
        return {"compassion": 0.5, "curiosity": 0.5, "stability": 0.5}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.grid = self._init_grid()
        self.total_extracted_energy = 0
        self.total_input_energy = 0
        self.time_step = 0
        self.extraction_history.clear()
        self.input_history.clear()
        return self.grid.copy(), {}

    def step(self, action):
        self.time_step += 1
        action = np.clip(action, -0.2, 0.2)
        cost_mod = np.where(self.grid < 0, 1.0 / (1e-10 - self.grid), 1.0)
        energy_input = np.sum((action**2) * cost_mod) * 1e-7
        self.grid += 1e-7 * action
        self.grid += np.random.normal(0, 1e-6, self.grid.shape)

        neg_coords = np.argwhere(self.grid < -1e-4)
        extracted_energy = 0
        for x, y, z in neg_coords:
            val = -self.grid[x, y, z]
            if val > 0:
                delta = np.log1p(val) ** 2
                self.grid[x, y, z] += delta
                extracted_energy += delta

        self.total_extracted_energy += extracted_energy
        self.total_input_energy += energy_input
        self.extraction_history.append(extracted_energy)
        self.input_history.append(energy_input)

        efficiency = extracted_energy / (energy_input + 1e-10)
        memory_bonus = max(0, self.total_extracted_energy - self.total_input_energy)
        stability_penalty = np.var(self.grid)

        # --- Predictive Curiosity ---
        prev_state = torch.tensor(self.grid.flatten(), dtype=torch.float32)
        predicted = self.predictor(prev_state)
        true_state = torch.tensor(self.grid.flatten(), dtype=torch.float32)
        prediction_error = torch.mean((predicted - true_state) ** 2)
        self.optimizer.zero_grad()
        prediction_error.backward()
        self.optimizer.step()
        curiosity_bonus = min(prediction_error.item(), 10.0)

        # --- Emotional State Evolution ---
        self.emotional_state["stress"] = min(1.0, self.emotional_state["stress"] + 0.01 * prediction_error.item())
        self.emotional_state["hope"] = max(0.0, self.emotional_state["hope"] + 0.01 * efficiency)

        # --- Final Reward with All Factors ---
        karma = self.karma_traits
        reward = (
            (extracted_energy - energy_input)
            + self.personality["focus"] * 0.5 * efficiency
            + self.personality["curiosity"] * 0.001 * memory_bonus
            + 0.01 * curiosity_bonus * karma.get("curiosity", 0.5)
            - 1e-6 * self.time_step
            - 0.001 * stability_penalty * (1 - karma.get("stability", 0.5))
        )
        return self.grid.copy(), reward, False, False, {}

    def save_memory(self):
        os.makedirs("memory", exist_ok=True)
        memory_id = f"memory_{self.memory_counter}"
        np.savez(f"memory/{memory_id}.npz",
                 extraction_history=np.array(self.extraction_history),
                 input_history=np.array(self.input_history),
                 final_grid=self.grid,
                 total_extracted_energy=self.total_extracted_energy,
                 total_input_energy=self.total_input_energy)
        with open(f"memory/{memory_id}_personality.json", "w") as f:
            json.dump(self.personality, f)
        with open(f"memory/{memory_id}_emotion.json", "w") as f:
            json.dump(self.emotional_state, f)
        self.memory_counter += 1

    def load_latest_memory(self):
        files = sorted(glob.glob("memory/memory_*.npz"))
        if not files:
            return 0
        latest = max(files, key=lambda x: int(x.split("_")[-1].split(".")[0]))
        data = np.load(latest)
        self.grid = data["final_grid"]
        self.total_extracted_energy = float(data["total_extracted_energy"])
        self.total_input_energy = float(data["total_input_energy"])
        self.extraction_history = data["extraction_history"].tolist()
        self.input_history = data["input_history"].tolist()
        self.memory_counter = int(latest.split("_")[-1].split(".")[0]) + 1
        try:
            with open(latest.replace(".npz", "_personality.json")) as f:
                self.personality = json.load(f)
            with open(latest.replace(".npz", "_emotion.json")) as f:
                self.emotional_state = json.load(f)
        except FileNotFoundError:
            pass
        print(f"[Resume] Loaded: {latest}")
        return self.memory_counter

    def adaptive_mutation(self, net_energy, round_counter):
        delta = 0.01
        decay = 0.999 ** round_counter

        energy_trend = (self.total_extracted_energy - self.total_input_energy) / (self.total_input_energy + 1e-10)
        self.personality["curiosity"] += delta * energy_trend * decay
        self.personality["curiosity"] = np.clip(self.personality["curiosity"], 0.0, 1.0)

        if net_energy > 0:
            self.personality["focus"] += delta * decay
        else:
            self.personality["focus"] -= delta * decay
        self.personality["focus"] = np.clip(self.personality["focus"], 0.0, 1.0)

        volatility = np.var(self.grid)
        if volatility > 1e-8:
            self.personality["adaptability"] += delta * decay
        else:
            self.personality["adaptability"] -= delta * decay
        self.personality["adaptability"] = np.clip(self.personality["adaptability"], 0.0, 1.0)

        print(f"[Mutation] Adaptive traits: {self.personality}")

# --- Breakthrough Export ---
def export_breakthrough(env, model):
    os.makedirs("breakthrough", exist_ok=True)
    np.savez("breakthrough/Breakthrough_principles.npz",
             extraction_history=np.array(env.extraction_history),
             input_history=np.array(env.input_history),
             final_grid=env.grid,
             total_extracted_energy=env.total_extracted_energy,
             total_input_energy=env.total_input_energy)
    model.save("breakthrough/ppo_model_at_breakthrough")
    with open("breakthrough/env_state.json", "w") as f:
        json.dump({
            "time_step": env.time_step,
            "personality": env.personality,
            "emotion": env.emotional_state,
            "karma": env.karma_traits,
            "net_energy": env.total_extracted_energy - env.total_input_energy
        }, f, indent=2)
    print("[Breakthrough] Exported to 'Breakthrough_principles.npz'")
    
def save_breakthrough_summary(env, round_counter):
    summary_path = f"breakthrough/summary_round_{round_counter}.txt"
    with open(summary_path, "w") as f:
        f.write(f"=== Breakthrough Summary — Round {round_counter} ===\n")
        f.write(f"Net Energy: {env.total_extracted_energy - env.total_input_energy:.6f}\n")
        f.write(f"Total Extracted Energy: {env.total_extracted_energy:.6f}\n")
        f.write(f"Total Input Energy: {env.total_input_energy:.6f}\n")
        f.write(f"Time Steps: {env.time_step}\n\n")

        f.write("Personality Traits:\n")
        for trait, value in env.personality.items():
            f.write(f"  {trait}: {value:.3f}\n")

        f.write("\nEmotional State:\n")
        for state, value in env.emotional_state.items():
            f.write(f"  {state}: {value:.3f}\n")

        f.write("\nKarma Traits:\n")
        for trait, value in env.karma_traits.items():
            f.write(f"  {trait}: {value:.3f}\n")
        
    print(f"[Summary] Saved breakthrough summary to: {summary_path}")

# --- Training ---
def make_env():
    return CasimirEnv()

vec_env = DummyVecEnv([make_env])
vec_env = VecMonitor(vec_env)
env_instance = vec_env.envs[0]

# --- Load from upgrade_checkpoint.npz if available ---
checkpoint_path = "upgrade_checkpoint.npz"
if os.path.exists(checkpoint_path):
    print(f"[Upgrade] Loading checkpoint from {checkpoint_path}")
    data = np.load(checkpoint_path)
    env_instance.grid = data["final_grid"]
    env_instance.total_extracted_energy = float(data["total_extracted_energy"])
    env_instance.total_input_energy = float(data["total_input_energy"])
    env_instance.extraction_history = data["extraction_history"].tolist()
    env_instance.input_history = data["input_history"].tolist()
    env_instance.memory_counter = 1499  # Or the correct round number if known
round_counter = env_instance.load_latest_memory()
model = PPO("MlpPolicy", vec_env, learning_rate=3e-4, verbose=1)

# --- Load round counter from file if available ---
round_counter = env_instance.load_latest_memory()
if os.path.exists("checkpoints/last_round.json"):
    with open("checkpoints/last_round.json", "r") as f:
        round_counter = json.load(f).get("round_counter", round_counter)


# --- PPO Load if Available ---
if os.path.exists("checkpoints/latest_ppo_model.zip"):
    print("[Model] Restoring PPO from checkpoints/latest_ppo_model.zip")
    model = PPO.load("checkpoints/latest_ppo_model", env=vec_env)
else:
    model = PPO("MlpPolicy", vec_env, learning_rate=3e-4, verbose=1)


while True:
    print(f"\n=== Training Round {round_counter + 1} ===")
    model.learn(total_timesteps=100000)

    extracted_total = env_instance.total_extracted_energy
    input_total = env_instance.total_input_energy
    net_energy = extracted_total - input_total
    print(f"[Result] Net Energy: {net_energy:.6f}")

    env_instance.save_memory()
    round_counter += 1

    if net_energy > 0 and round_counter % 10 == 0:
        export_breakthrough(env_instance, model)
        save_breakthrough_summary(env_instance, round_counter)


    if round_counter % 100 == 0:
        env_instance.adaptive_mutation(net_energy, round_counter)

    if round_counter % 500 == 0:
        env_instance.grid += np.random.normal(0, 1e-4, env_instance.grid.shape)
    if round_counter % 10 == 0:
        model.save("checkpoints/latest_ppo_model")


# --- PPO Model Save ---
os.makedirs("checkpoints", exist_ok=True)
model.save("checkpoints/latest_ppo_model")


print("Training in AICps.py Complete.")