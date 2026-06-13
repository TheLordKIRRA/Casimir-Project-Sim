import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

# --- Modify this to your memory folder path ---
memory_dir = "d:/school/CPS/memory"
memory_ids = list(range(1373, 1383))  # Last 10 memory rounds

records = []
for mid in memory_ids:
    file_path = os.path.join(memory_dir, f"memory_{mid}.npz")
    if not os.path.exists(file_path):
        continue
    data = np.load(file_path)
    extracted = np.sum(data["extraction_history"])
    input_energy = np.sum(data["input_history"])
    net_energy = extracted - input_energy
    records.append({
        "Round": mid,
        "Extracted": extracted,
        "Input": input_energy,
        "Net": net_energy
    })

# --- Convert to DataFrame ---
df = pd.DataFrame(records)
print(df)

# --- Plot ---
plt.figure(figsize=(10, 6))
plt.plot(df["Round"], df["Extracted"], label="Extracted Energy")
plt.plot(df["Round"], df["Input"], label="Input Energy")
plt.plot(df["Round"], df["Net"], label="Net Energy")
plt.xlabel("Round")
plt.ylabel("Energy")
plt.title("Casimir Energy Trends")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Predict Net-Positive Round ---
X = df["Round"].values.reshape(-1, 1)
y = df["Net"].values
model = LinearRegression().fit(X, y)

for future_round in range(1383, 1600):
    prediction = model.predict([[future_round]])
    if prediction >= 0:
        print(f"Projected net-positive at round: {future_round}")
        break
