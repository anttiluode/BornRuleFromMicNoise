import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm # For erf related calculations if needed

# --- Parameters ---
NUM_ALPHA_POINTS = 50
alphas_instrument = np.linspace(0, np.pi, NUM_ALPHA_POINTS) # External measurement setting

# CV values to test (representing different local membrane noise states)
CV_values_membrane = [0.3, 0.4, 0.5, 0.522, 0.6, 0.7] 

# For "compare two noisy intensities" simulation (if we do it that way)
TRIALS_PER_POINT = 2000

# --- Helper Functions ---

def effective_noise_std_from_cv(cv_membrane, ideal_intensity_diff_scale=1.0):
    """
    Convert a CV value to an effective noise standard deviation for the
    "compare two noisy intensities" mechanism.
    This is a heuristic link: if CV = std_env / mean_env, and mean_env ~ ideal_intensity_diff_scale,
    then std_env ~ CV * ideal_intensity_diff_scale.
    The value 0.522 was found empirically to give good Born fit when diff_scale=1.
    So, we can scale relative to that.
    If CV=0.5, we want noise_std ~ 0.522. So, noise_std = CV * (0.522/0.5)
    """
    return cv_membrane * (0.522 / 0.5) * ideal_intensity_diff_scale

def simulate_born_with_cv_noise(alpha_instrument, cv_membrane, trials):
    """
    Simulates P(0|alpha) using the "compare two noisy intensities"
    where the noise_std is derived from cv_membrane.
    """
    noise_std = effective_noise_std_from_cv(cv_membrane)
    
    I0_ideal = np.cos(alpha_instrument / 2.0)**2
    I1_ideal = np.sin(alpha_instrument / 2.0)**2
    wins = 0
    for _ in range(trials):
        n0 = np.random.normal(0, noise_std)
        n1 = np.random.normal(0, noise_std)
        if (I0_ideal + n0) > (I1_ideal + n1):
            wins += 1
    return wins / trials

# --- Store Results ---
all_P0_curves = {}

# --- Simulation ---
print("Simulating Born rule for different membrane CV values...")
for cv_val in CV_values_membrane:
    print(f"  Running for CV_membrane = {cv_val:.3f} (effective noise_std = {effective_noise_std_from_cv(cv_val):.3f})")
    current_P0s = []
    for alpha_instr in alphas_instrument:
        # Here, the "lump" is assumed to be perfectly prepared along the direction
        # that would ideally correspond to alpha_instr if CV was perfect.
        # The CV of the membrane noise then affects the measurement outcome.
        p0 = simulate_born_with_cv_noise(alpha_instr, cv_val, TRIALS_PER_POINT)
        current_P0s.append(p0)
    all_P0_curves[cv_val] = np.array(current_P0s)

# --- Plotting ---
plt.figure(figsize=(12, 8))
# Plot ideal QM Born rule
plt.plot(alphas_instrument, np.cos(alphas_instrument/2)**2, 'k--', lw=2, label='Ideal QM: cos²(α/2)')

for cv_val, p0_curve in all_P0_curves.items():
    eff_noise = effective_noise_std_from_cv(cv_val)
    plt.plot(alphas_instrument, p0_curve, 'o-', ms=4, alpha=0.7,
             label=f'CV_mem={cv_val:.3f} (Eff. Noise Std≈{eff_noise:.3f})')

plt.xlabel("Instrument Preparation Angle α (radians)")
plt.ylabel("Probability P(Outcome 0)")
plt.title("Born Rule Emergence vs. Membrane Envelope CV\n(Frequency Lump Angle Modulated by CV-derived Noise)")
plt.legend(loc='upper right')
plt.grid(True)
plt.ylim(-0.05, 1.05)
plt.show()

print("\nNote on interpretation:")
print("This simulation directly uses the 'compare two noisy intensities' mechanism.")
print("The 'CV_membrane' translates to an 'effective_noise_std' for that comparison.")
print("A lower CV (and thus lower effective_noise_std) will make the P(0|α) curve sharper.")
print("A higher CV (and higher effective_noise_std) will make it flatter.")
print("The curve matches cos²(α/2) best when effective_noise_std is around 0.522 (derived from CV≈0.5).")

# --- Conceptual CHSH with CV-dependent local measurements ---
def run_chsh_with_cv_noise(cv_membrane_alice, cv_membrane_bob, ftl_effectiveness, num_pairs=10000):
    noise_std_A = effective_noise_std_from_cv(cv_membrane_alice)
    noise_std_B = effective_noise_std_from_cv(cv_membrane_bob)

    # CHSH Angles
    alpha_settings = [0.0, np.pi / 2.0]
    beta_settings = [np.pi / 4.0, 3.0 * np.pi / 4.0]
    E_matrix = np.zeros((2,2))
    counts = np.zeros((2,2), dtype=int)

    for _ in range(num_pairs):
        alice_setting_idx = np.random.randint(2)
        bob_setting_idx = np.random.randint(2)
        alpha = alpha_settings[alice_setting_idx]
        beta = beta_settings[bob_setting_idx]

        # Alice's local measurement
        I_A0 = np.cos(alpha/2)**2; I_A1 = np.sin(alpha/2)**2
        nA0 = np.random.normal(0, noise_std_A); nA1 = np.random.normal(0, noise_std_A)
        outcome_A_val = 1 if (I_A0 + nA0) > (I_A1 + nA1) else -1
        alice_collapse_phase = 0.0 if outcome_A_val == 1 else np.pi

        # FTL Influence on Bob's *ideal probabilities*
        # Bob's particle is now effectively oriented at (alice_collapse_phase + pi) relative to Alice's setting axis (alpha)
        effective_angle_for_bob_born = (beta - (alpha + ((0.0 if outcome_A_val == 1 else np.pi) + np.pi))) # (A_phase+pi) is Bob's state axis
        
        I_B0_ideal_ftl = np.cos(effective_angle_for_bob_born / 2.0)**2
        I_B1_ideal_ftl = np.sin(effective_angle_for_bob_born / 2.0)**2
        
        I_B0_ideal_local = np.cos(beta/2)**2 # Bob's original ideal if no FTL
        I_B1_ideal_local = np.sin(beta/2)**2

        # Mix based on FTL effectiveness
        I_B0_final = (1 - ftl_effectiveness) * I_B0_ideal_local + ftl_effectiveness * I_B0_ideal_ftl
        I_B1_final = (1 - ftl_effectiveness) * I_B1_ideal_local + ftl_effectiveness * I_B1_ideal_ftl

        # Bob's local measurement
        nB0 = np.random.normal(0, noise_std_B); nB1 = np.random.normal(0, noise_std_B)
        outcome_B_val = 1 if (I_B0_final + nB0) > (I_B1_final + nB1) else -1
        
        E_matrix[alice_setting_idx, bob_setting_idx] += outcome_A_val * outcome_B_val
        counts[alice_setting_idx, bob_setting_idx] += 1
    
    E_matrix /= counts
    S_value = E_matrix[0,0] - E_matrix[0,1] + E_matrix[1,0] + E_matrix[1,1]
    return S_value

print("\n--- Running CHSH with CV-dependent local noise and FTL ---")
# Example: Assume Alice and Bob experience slightly different local membrane CVs
cv_A_local = 0.5
cv_B_local = 0.55 
ftl_eff = 0.95 # Strong FTL effectiveness

S_test = run_chsh_with_cv_noise(cv_A_local, cv_B_local, ftl_eff)
print(f"CHSH S-value for CV_A={cv_A_local}, CV_B={cv_B_local}, FTL_eff={ftl_eff}: {S_test:.4f}")
if abs(S_test) > 2.0: print("  Bell Inequality VIOLATED!")

# What if FTL is perfect (eff=1.0) and local noise is ideal (CV=0.5)?
S_ideal_ftl_ideal_noise = run_chsh_with_cv_noise(0.5, 0.5, 1.0)
print(f"CHSH S for CV_A=0.5, CV_B=0.5, FTL_eff=1.0: {S_ideal_ftl_ideal_noise:.4f}")
if abs(S_ideal_ftl_ideal_noise) > 2.0: print("  Bell Inequality VIOLATED!")


# What if FTL is perfect but local noise is higher (e.g., CV=0.6 for both)?
S_ideal_ftl_higher_noise = run_chsh_with_cv_noise(0.6, 0.6, 1.0)
print(f"CHSH S for CV_A=0.6, CV_B=0.6, FTL_eff=1.0: {S_ideal_ftl_higher_noise:.4f}")
if abs(S_ideal_ftl_higher_noise) > 2.0: print("  Bell Inequality VIOLATED!")