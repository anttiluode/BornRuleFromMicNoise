import numpy as np
import matplotlib.pyplot as plt

# --- Parameters ---
TOTAL_TIME_STEPS = 1000
NUM_PARTICLES = 5000 # Simulate an ensemble

# Natural decay probability per time step if NOT observed
# If observed, this natural decay is overridden by the observation outcome
PROB_NATURAL_DECAY = 0.01 # 1% chance to decay per step if left alone

# "Observation" parameters (mimicking your Born rule setup)
# Let I_A be the "intensity" for being in state A, I_B for state B
# If in state A, ideal I_A = 1, ideal I_B = 0
# If in state B, ideal I_A = 0, ideal I_B = 1 (not used in this simple Zeno)

# Noise standard deviation, crucial for your CV ~ 0.5 idea.
# If ideal I_A=1, I_B=0, the difference is 1.
# For CV ~ 0.5, if mean diff is 1, std_dev of noise on diff is ~0.5.
# If n_A and n_B are independent with std_dev sigma, then (n_A - n_B) has std_dev sqrt(2)*sigma.
# So, if sqrt(2)*sigma / 1 ~ 0.5 (for the erf function argument scaling), then sigma ~ 0.5/sqrt(2) ~ 0.35.
# Let's use the value that worked well for your Born rule:
AETHER_NOISE_STD = 0.522 # Applied to each intensity before comparison
# This means the effective std_dev on the *difference* of noisy intensities is sqrt(2)*0.522 ~ 0.738

OBSERVATION_INTERVALS = [0, 1, 2, 5, 10, 20, 50, 100] # 0 means no observation (only natural decay)
                                                     # 1 means observe every step

# --- Simulation ---

all_survival_curves = {}

for obs_interval in OBSERVATION_INTERVALS:
    # particle_state = 1 (in state A), 0 (in state B/decayed)
    particles_in_state_A = np.ones(NUM_PARTICLES, dtype=int)
    survival_count = np.zeros(TOTAL_TIME_STEPS)

    for t in range(TOTAL_TIME_STEPS):
        # Count how many are still in state A
        survival_count[t] = np.sum(particles_in_state_A)

        # --- Natural Decay Process (for particles not yet decayed) ---
        # This happens IF NOT OBSERVED in this step OR if observation "fails" to keep it in A
        # Let's refine: Natural decay applies if an observation *doesn't* occur *this specific timestep*
        
        currently_A_mask = (particles_in_state_A == 1)
        num_currently_A = np.sum(currently_A_mask)
        
        if num_currently_A == 0: # All decayed
            survival_count[t:] = 0 # Fill rest of survival curve
            break

        # --- "Observation" Process ---
        if obs_interval > 0 and (t % obs_interval == 0):
            # These particles are being "observed"
            observing_mask = currently_A_mask # Observe all that are currently in A
            num_to_observe = np.sum(observing_mask)

            if num_to_observe > 0:
                # Ideal intensities for being in state A
                I_A_ideal = 1.0
                I_B_ideal = 0.0 # Intensity for "decaying to B" or "being in B"

                noise_A = np.random.normal(0, AETHER_NOISE_STD, num_to_observe)
                noise_B = np.random.normal(0, AETHER_NOISE_STD, num_to_observe)

                noisy_I_A = I_A_ideal + noise_A
                noisy_I_B = I_B_ideal + noise_B

                # Decision: if noisy_I_A > noisy_I_B, it "remains" in state A due to observation
                # Otherwise, the observation "caused" a transition (or failed to prevent it)
                remains_A_due_to_obs = (noisy_I_A > noisy_I_B)
                
                # Update particles_in_state_A for those observed
                # Get original indices of particles being observed
                observed_indices = np.where(observing_mask)[0]
                
                # Those that did not remain A (due to observation) decay
                particles_in_state_A[observed_indices[~remains_A_due_to_obs]] = 0
        
        # --- Natural Decay for those NOT observed THIS specific timestep ---
        # Re-calculate currently_A_mask after potential observation-induced decays
        currently_A_mask_after_obs = (particles_in_state_A == 1)
        
        # Determine which particles were NOT subject to observation this step
        if obs_interval == 0: # No observations at all
            not_observed_this_step_mask = currently_A_mask_after_obs
        else:
            was_observed_this_step = (t % obs_interval == 0)
            if was_observed_this_step:
                not_observed_this_step_mask = np.zeros_like(particles_in_state_A, dtype=bool) # None, all were handled
            else:
                not_observed_this_step_mask = currently_A_mask_after_obs # All that are A and not observed

        num_eligible_for_natural_decay = np.sum(not_observed_this_step_mask)

        if num_eligible_for_natural_decay > 0:
            # Apply natural decay probability
            rand_for_decay = np.random.rand(num_eligible_for_natural_decay)
            natural_decay_occurs = (rand_for_decay < PROB_NATURAL_DECAY)
            
            # Get original indices of particles eligible for natural decay
            natural_decay_indices = np.where(not_observed_this_step_mask)[0]
            
            # Update particles_in_state_A for those that naturally decayed
            particles_in_state_A[natural_decay_indices[natural_decay_occurs]] = 0
            
    all_survival_curves[obs_interval] = survival_count / NUM_PARTICLES

# --- Plotting ---
plt.figure(figsize=(12, 8))
for obs_interval, survival_curve in all_survival_curves.items():
    if obs_interval == 0:
        label = "No Observation (Natural Decay)"
        plt.plot(survival_curve, label=label, linestyle='--', linewidth=2)
    else:
        label = f"Observed every {obs_interval} steps"
        plt.plot(survival_curve, label=label)

plt.xlabel("Time Steps")
plt.ylabel("Fraction of Particles in State A (Survival)")
plt.title(f"Classical Zeno Effect Analogue (Simulated Aether Noise)\nNatural Decay Prob={PROB_NATURAL_DECAY}, Noise Std={AETHER_NOISE_STD}")
plt.legend()
plt.grid(True)
plt.ylim(0, 1.05)
plt.show()

# --- Analysis of "Half-Life" (Time to 50% survival) ---
print("Approximate 'Half-Life' (Time for 50% to decay):")
for obs_interval, survival_curve in all_survival_curves.items():
    half_life_idx = np.where(survival_curve <= 0.5)[0]
    if len(half_life_idx) > 0:
        half_life = half_life_idx[0]
    else:
        half_life = TOTAL_TIME_STEPS # Survived longer than simulation
    
    obs_label = "No Obs" if obs_interval == 0 else f"Obs Int: {obs_interval}"
    print(f"{obs_label:<15}: {half_life} time steps")