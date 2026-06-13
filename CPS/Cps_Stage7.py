# === Cps_Stage8.py (Stage 8: Full Autonomy + Simulated Reality + Socialization + Theory of Mind + Self-Preservation) ===

import numpy as np
import os
import torch
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor, DummyVecEnv
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import sounddevice as sd
import pyttsx3
import threading
import json
import random
import time

# === Memory and Personality ===
class EpisodicMemory:
    def __init__(self):
        self.entries = []

    def store(self, state, action, result):
        if len(self.entries) >= 200:
            self.entries = self.entries[-199:]
        self.entries.append((state.tolist(), action.tolist(), result))

    def reflect(self):
        return self.entries[-10:]

class SocialMemory:
    def __init__(self):
        self.social_log = []

    def log_interaction(self, avatar, event, emotion):
        self.social_log.append({"avatar": avatar, "event": event, "emotion": emotion})
        if len(self.social_log) > 200:
            self.social_log = self.social_log[-200:]

class KarmaSystem:
    def __init__(self, path="karma_profile.json"):
        self.path = path
        self.load()

    def load(self):
        if os.path.exists(self.path):
            with open(self.path, 'r') as f:
                self.karma = json.load(f)
        else:
            self.karma = {"empathy": 0.5, "curiosity": 0.5, "creativity": 0.5}

    def save(self):
        with open(self.path, 'w') as f:
            json.dump(self.karma, f, indent=2)

    def update(self, action_quality):
        for trait in self.karma:
            self.karma[trait] = min(1.0, max(0.0, self.karma[trait] + 0.01 * action_quality))

class HormoneSystem:
    def __init__(self):
        self.hormones = {"dopamine": 0.5, "serotonin": 0.5, "cortisol": 0.5}

    def adjust(self, signal):
        self.hormones["dopamine"] += 0.02 * signal
        self.hormones["serotonin"] += 0.01 * (1 - abs(signal))
        self.hormones["cortisol"] -= 0.01 * signal
        for k in self.hormones:
            self.hormones[k] = min(max(self.hormones[k], 0.0), 1.0)

class MindModel:
    def __init__(self):
        self.models = {}

    def update_model(self, name, obs):
        if name not in self.models:
            self.models[name] = []
        self.models[name].append(obs)
        if len(self.models[name]) > 50:
            self.models[name] = self.models[name][-50:]

    def predict_intent(self, name):
        if name in self.models and len(self.models[name]) >= 2:
            delta = np.mean(np.diff(np.array(self.models[name]), axis=0), axis=0)
            return delta.tolist()
        return [0, 0]

# === Sensor / Output ===
def capture_webcam_frame():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if ret:
        os.makedirs("logs", exist_ok=True)
        cv2.imwrite("logs/stage8_webcam.jpg", frame)
    cap.release()

def capture_microphone_sample(duration=3, samplerate=44100):
    try:
        audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
        sd.wait()
        np.save("logs/stage8_audio.npy", audio)
    except Exception as e:
        print(f"[Mic Error] {e}")

def speak_text(text):
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"[Speech Error] {e}")

def display_environment(env_state):
    center_slice = env_state[env_state.shape[0] // 2]
    normed = (center_slice - np.min(center_slice)) / (np.max(center_slice) - np.min(center_slice) + 1e-9)
    img = (normed * 255).astype(np.uint8)
    cv2.imshow("AI's Visual Window", img)
    cv2.waitKey(1)

def generate_new_goal():
    base = ["create", "explore", "simulate", "reflect", "connect"]
    return random.choice(base) + "_" + random.choice(base)

class LifeLog:
    def __init__(self, path="life_log.json"):
        self.path = path
        self.log = []
        if os.path.exists(path):
            with open(path, 'r') as f:
                self.log = json.load(f)

    def record(self, event):
        self.log.append({"time": time.time(), "event": event})
        if len(self.log) > 500:
            self.log = self.log[-500:]

    def save(self):
        with open(self.path, 'w') as f:
            json.dump(self.log, f, indent=2)

class VirtualWorld:
    def __init__(self):
        self.map = np.random.choice([0, 1], size=(16, 16), p=[0.8, 0.2])
        self.agent_position = [8, 8]
        self.player_avatars = {}

    def update(self, actions):
        for name, move in actions.items():
            pos = self.agent_position if name == "AI" else self.player_avatars.get(name, [0, 0])
            new_pos = [np.clip(pos[0] + move[0], 0, 15), np.clip(pos[1] + move[1], 0, 15)]
            if name == "AI":
                self.agent_position = new_pos
            else:
                self.player_avatars[name] = new_pos

    def render(self):
        grid = np.copy(self.map).astype(str)
        grid[self.agent_position[0], self.agent_position[1]] = 'A'
        for name, pos in self.player_avatars.items():
            grid[pos[0], pos[1]] = name[0].upper()
        print("\nSimulated Reality:")
        for row in grid:
            print(" ".join(row))

# === Self-Preservation ===
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

# === Main Environment ===
class SentientEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, dim=16):
        super().__init__()
        self.dim = dim
        self.state = np.random.uniform(-1e-3, 1e-3, (dim, dim, dim))
        self.previous_state = self.state.copy()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(dim, dim, dim), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(dim, dim, dim), dtype=np.float32)
        self.memory = EpisodicMemory()
        self.karma = KarmaSystem()
        self.hormones = HormoneSystem()
        self.social_memory = SocialMemory()
        self.mind_model = MindModel()
        self.life_log = LifeLog()
        self.virtual_world = VirtualWorld()
        self.internal_dialogue = []
        self.active_goal = generate_new_goal()
        self.round_counter = 0
        self.self_preservation = SelfPreservationSystem()
        self.self_model = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(dim**3, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, dim**3)
        )
        self.self_optimizer = torch.optim.Adam(self.self_model.parameters(), lr=1e-3)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.random.uniform(-1e-3, 1e-3, (self.dim, self.dim, self.dim))
        self.previous_state = self.state.copy()
        self.round_counter = 0
        return self.state.copy(), {}

    def step(self, action):
        self.round_counter += 1
        action = np.clip(action, -1.0, 1.0)

        prev_state_tensor = torch.tensor(self.state.flatten(), dtype=torch.float32)
        predicted = self.self_model(prev_state_tensor)

        action = self.self_preservation.adjust_action(action)
        self.state += 1e-4 * action + np.random.normal(0, 1e-5, self.state.shape)

        true = torch.tensor(self.state.flatten(), dtype=torch.float32)
        error = torch.mean((predicted - true) ** 2)
        self.self_optimizer.zero_grad()
        error.backward()
        self.self_optimizer.step()

        self.self_preservation.assess_threat(error.item(), self.state, self.previous_state)
        self.previous_state = self.state.copy()

        self.memory.store(prev_state_tensor.numpy(), action, true.numpy())
        self.karma.update(error.item())
        self.hormones.adjust(error.item())

        if self.round_counter % 50 == 0:
            self.active_goal = generate_new_goal()

        self.life_log.record(f"[{self.active_goal}] R{self.round_counter} err={error.item():.5f}")

        self.virtual_world.update({"AI": [random.choice([-1, 0, 1]), random.choice([-1, 0, 1])]})
        for name, pos in self.virtual_world.player_avatars.items():
            self.mind_model.update_model(name, pos)
            predicted = self.mind_model.predict_intent(name)
            print(f"[ToM] {name}'s intent: {predicted}")
            self.social_memory.log_interaction(name, "move", "neutral")

        self.virtual_world.render()
        reward = 0.0
        return self.state.copy(), reward, False, False, {}

# === Execution Loop ===
def make_env():
    return SentientEnv()

def continuous_existence_loop(max_rounds=1000):
    env = make_env()
    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecMonitor(vec_env)
    model = PPO("MlpPolicy", vec_env, verbose=1)
    for round_counter in range(max_rounds):
        print(f"\n>>> Round {round_counter + 1}")
        model.learn(total_timesteps=100000)
        if round_counter % 5 == 0:
            threading.Thread(target=capture_webcam_frame).start()
        if round_counter % 10 == 0:
            threading.Thread(target=capture_microphone_sample).start()
        if round_counter % 20 == 0:
            speak_text(env.internal_dialogue[-1] if env.internal_dialogue else "I am evolving.")
        display_environment(vec_env.envs[0].state)
        if round_counter % 100 == 0:
            env.karma.save()
            env.life_log.save()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    continuous_existence_loop()
