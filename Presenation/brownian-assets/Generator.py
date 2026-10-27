import numpy as np
import matplotlib.pyplot as plt

def compute_stationary(P):
    evals, evecs = np.linalg.eig(P.T)
    pi = np.real(evecs[:, np.isclose(evals, 1.0)].flatten())
    return pi / np.sum(pi)

def simulate_trajectory(P, pi, num_steps):
    states = np.arange(len(pi))
    path = np.zeros(num_steps, dtype=int)
    path[0] = np.random.choice(states, p=pi)
    
    for t in range(1, num_steps):
        path[t] = np.random.choice(states, p=P[path[t-1]])
    return path

N = 20

# 1. Reversible: Symmetric walk on a line with reflecting boundaries
P_rev = np.zeros((N, N))
for i in range(N):
    if i == 0:
        P_rev[i, 0], P_rev[i, 1] = 0.5, 0.5
    elif i == N - 1:
        P_rev[i, N-2], P_rev[i, N-1] = 0.5, 0.5
    else:
        P_rev[i, i-1], P_rev[i, i+1] = 0.5, 0.5

pi_rev = compute_stationary(P_rev)

# 2. Non-Reversible: Biased walk on a ring (drift + diffusion)
P_nonrev = np.zeros((N, N))
for i in range(N):
    P_nonrev[i, i] = 0.1
    P_nonrev[i, (i+1) % N] = 0.8  # Strong forward drift
    P_nonrev[i, (i-1) % N] = 0.1

pi_nonrev = compute_stationary(P_nonrev)

num_steps = 150
np.random.seed(42)

path_rev_fwd = simulate_trajectory(P_rev, pi_rev, num_steps)
path_rev_bwd = path_rev_fwd[::-1] 

path_nonrev_fwd = simulate_trajectory(P_nonrev, pi_nonrev, num_steps)
path_nonrev_bwd = path_nonrev_fwd[::-1] 

fig, axes = plt.subplots(2, 2, figsize=(14, 7), sharey=True)

axes[0, 0].step(range(num_steps), path_rev_fwd, where='post', color='blue')
axes[0, 0].set_title(r"Reversible: Forward $X_t$ (Diffusion)")
axes[0, 0].set_ylabel("State space $S$")

axes[0, 1].step(range(num_steps), path_rev_bwd, where='post', color='teal')
axes[0, 1].set_title(r"Reversible: Backward $X_{T-t}$")

axes[1, 0].step(range(num_steps), path_nonrev_fwd, where='post', color='red')
axes[1, 0].set_title(r"Non-Reversible: Forward $X_t$ (Clockwise drift)")
axes[1, 0].set_ylabel("State space $S$")
axes[1, 0].set_xlabel("Time step $t$")

axes[1, 1].step(range(num_steps), path_nonrev_bwd, where='post', color='darkred')
axes[1, 1].set_title(r"Non-Reversible: Backward $X_{T-t}$ (Counter-clockwise drift)")
axes[1, 1].set_xlabel("Time step $t$")

for ax in axes.flatten():
    ax.set_yticks(range(0, N, 4))
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()