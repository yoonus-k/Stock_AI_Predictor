import numpy as np
import matplotlib.pyplot as plt
import os

# Path to evaluations file
eval_file = './logs/evaluations.npz'

# Ensure file exists
if not os.path.exists(eval_file):
    raise FileNotFoundError(f"Evaluation file not found at: {eval_file}")

# Load evaluation data
data = np.load(eval_file)

# Keys: timesteps, results, ep_lengths
timesteps = data['timesteps']
results = data['results']
ep_lengths = data['ep_lengths']

# Plot average reward over time
mean_rewards = results.mean(axis=1)
plt.figure(figsize=(12, 6))
plt.plot(timesteps, mean_rewards, label='Mean Reward')
plt.title("Evaluation: Mean Reward Over Timesteps")
plt.xlabel("Timesteps")
plt.ylabel("Mean Reward")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Optional: plot episode lengths
mean_ep_lengths = ep_lengths.mean(axis=1)
plt.figure(figsize=(12, 4))
plt.plot(timesteps, mean_ep_lengths, label='Mean Episode Length', color='orange')
plt.title("Evaluation: Mean Episode Length Over Timesteps")
plt.xlabel("Timesteps")
plt.ylabel("Length")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
