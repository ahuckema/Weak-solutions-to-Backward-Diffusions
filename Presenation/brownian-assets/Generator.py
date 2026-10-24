import numpy as np
import matplotlib.pyplot as plt
import os

# Create the folder your LaTeX code expects
output_dir = "brownian-assets"
os.makedirs(output_dir, exist_ok=True)

# Simulation parameters
n_paths = 10000       # Large N for a smooth empirical distribution
n_frames = 40
T = 1.0
t = np.linspace(0, T, n_frames)
dt = t[1] - t[0]

# 1. Simulate Brownian paths: W(t+dt) = W(t) + sqrt(dt)*N(0,1)
dW = np.random.normal(0, np.sqrt(dt), (n_paths, n_frames - 1))
W = np.hstack([np.zeros((n_paths, 1)), np.cumsum(dW, axis=1)])

# Spatial grid for the exact Heat Equation solution
x_grid = np.linspace(-3, 3, 200)
hist_bins = np.linspace(-3, 3, 60)

for i in range(n_frames):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    current_time = t[i]

    # LEFT PANEL: Stochastic view (Brownian Paths)
    # We plot only 100 paths to avoid visual clutter
    ax1.plot(t[:i+1], W[:100, :i+1].T, color="#16B8A6", alpha=0.15)
    ax1.axvline(current_time, color="#F47763", linestyle="--", linewidth=1.5)
    ax1.set_xlim(0, T)
    ax1.set_ylim(-3, 3)
    ax1.set_title(r"Stochastic: Sample Paths $W(t)$")
    ax1.set_xlabel(r"Time $t$")
    ax1.set_ylabel(r"Position $x$")

    # RIGHT PANEL: Deterministic view (Heat Equation)
    # Empirical histogram of all 10,000 particles
    ax2.hist(W[:, i], bins=hist_bins, density=True, color="#16B8A6", alpha=0.6, label="Empirical Density")
    ax2.set_xlim(-3, 3)
    ax2.set_ylim(0, 1.5)
    ax2.set_title(rf"Deterministic: Density at $t = {current_time:.3f}$")
    ax2.set_xlabel(r"Position $x$")

    # Exact solution to the heat equation
    if current_time > 0:
        heat_eq = np.exp(-x_grid**2 / (2 * current_time)) / np.sqrt(2 * np.pi * current_time)
        ax2.plot(x_grid, heat_eq, color="#F47763", linewidth=2.5, label="Heat Eq Solution")
    
    ax2.legend(loc="upper right", frameon=False)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/frame-{i}.png", dpi=300)
    plt.close()

print(f"Successfully generated {n_frames} frames in '{output_dir}/'.")