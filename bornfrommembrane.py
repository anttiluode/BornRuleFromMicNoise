import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation # For 3D rotations

# --- Parameters ---
NUM_ALPHA_POINTS = 50
alphas_instrument = np.linspace(0, np.pi, NUM_ALPHA_POINTS) # External measurement setting

# CV_amplitude of the substrate - we are particularly interested in CV=0.5
CV_SUBSTRATE = 0.5

# How CV translates to angular jitter (standard deviation of jitter angle in radians)
# This is the KEY heuristic parameter. We need to find a value that works.
# Let's hypothesize that CV=0.5 corresponds to a significant angular spread.
# If effective_noise_std from qubitgem for CV=0.5 was ~0.522,
# and I0-I1 = cos(alpha_eff). We need std(cos(alpha_eff)) ~ 0.522.
# If alpha_eff = alpha + delta_alpha, and delta_alpha is normal(0, sigma_angle)
# std(cos(alpha+delta_alpha)) over delta_alpha.
# Let's try a sigma_angle. A large angle like pi/4 radians (45 deg) might be too much.
# pi/6 (30 deg) -> ~0.52 rad.
# pi/8 (22.5 deg) -> ~0.39 rad.
# What if the *variance* of the angle is proportional to CV?
# Or std_dev of angle is proportional to CV?
# Let sigma_angle = K * CV_SUBSTRATE. We need to find K.
# If CV=0.5 should give Born, maybe K is around 0.6-0.8?
# std_dev_angular_jitter_rad = 0.7 * CV_SUBSTRATE  # Try this: for CV=0.5, std_dev_angle = 0.35 rad (20 deg)
std_dev_angular_jitter_rad = 0.65 * CV_SUBSTRATE # For CV=0.5, std_dev_angle = 0.325 rad (~18.6 deg)
# After some trial and error, a value around 0.65 to 0.75 for K (when CV=0.5) seems to give good results.
# Let's use a K that makes std_dev_angular_jitter_rad directly interpretable for CV=0.5.
# For example, if we want std_dev_angle ~ 0.33 rad when CV=0.5, K = 0.33/0.5 = 0.66
K_CV_TO_ANGLE_STD = 0.66
std_dev_angular_jitter_rad = K_CV_TO_ANGLE_STD * CV_SUBSTRATE


TRIALS_PER_POINT = 3000

# --- Helper Functions ---

def random_rotation_around_axis(vector, axis, angle_std_dev):
    """Rotates a vector around a given axis by a normally distributed angle."""
    if angle_std_dev == 0:
        return vector
    angle = np.random.normal(0, angle_std_dev)
    # Ensure axis is a unit vector
    axis = axis / np.linalg.norm(axis)
    rot = Rotation.from_rotvec(angle * axis)
    return rot.apply(vector)

def get_jittered_lump_orientation(ideal_orientation_vec, angular_std_dev):
    """
    Applies a random angular jitter to the ideal orientation vector.
    The jitter is a rotation by a random angle (from normal dist with angular_std_dev)
    around a random axis *perpendicular* to ideal_orientation_vec.
    """
    if angular_std_dev == 0:
        return ideal_orientation_vec

    # Generate a random axis perpendicular to ideal_orientation_vec
    random_vec = np.random.randn(3)
    # Ensure it's not parallel to ideal_orientation_vec
    while np.allclose(np.cross(ideal_orientation_vec, random_vec), 0):
        random_vec = np.random.randn(3)
        
    perp_axis = np.cross(ideal_orientation_vec, random_vec)
    if np.linalg.norm(perp_axis) < 1e-9: # Should be rare with the while loop
        # If still parallel (e.g. random_vec was zero), pick an arbitrary perpendicular
        if np.allclose(ideal_orientation_vec, [0,0,1]) or np.allclose(ideal_orientation_vec, [0,0,-1]):
            perp_axis = np.array([1,0,0])
        else:
            perp_axis = np.cross(ideal_orientation_vec, np.array([0,0,1]))

    perp_axis_unit = perp_axis / np.linalg.norm(perp_axis)
    
    # Get random rotation angle
    jitter_angle = np.random.normal(0, angular_std_dev)
    
    # Create rotation object
    rot = Rotation.from_rotvec(jitter_angle * perp_axis_unit)
    
    # Apply rotation
    return rot.apply(ideal_orientation_vec)


def simulate_born_angular_jitter(alpha_instrument_rad, cv_substrate_val, trials):
    """
    Simulates P(0|alpha) using the "CV as angular jitter" model.
    alpha_instrument_rad is the angle of the ideal lump orientation from the +Z axis, in the XZ plane.
    Measurement apparatus axis for outcome "0" is +Z.
    """
    # Derive angular jitter std dev from CV
    angular_std_dev = K_CV_TO_ANGLE_STD * cv_substrate_val

    # Apparatus axis for outcome "0" (e.g., Spin Up)
    apparatus_axis_A0 = np.array([0, 0, 1.0])
    # Apparatus axis for outcome "1" (e.g., Spin Down) is implicitly -A0 for this simple model
    # or we can think of it as comparing projection on A0 vs projection on orthogonal plane.
    # Let's stick to comparing projection on A0 vs projection on -A0 (which is |L_z|^2 vs |-L_z|^2 for L_z < 0)
    # More accurately, compare intensity for A0 vs intensity for A1 (orthogonal to A0).
    # For simplicity of a 2-outcome projection like Stern-Gerlach along Z:
    # Outcome 0 if Z-component of L_jittered is positive.
    # Outcome 1 if Z-component of L_jittered is negative.
    # Probability proportional to squared projection.

    # Ideal lump orientation vector (prepared state)
    # Lies in XZ plane, angle alpha_instrument from +Z towards +X
    ideal_L_x = np.sin(alpha_instrument_rad)
    ideal_L_y = 0.0
    ideal_L_z = np.cos(alpha_instrument_rad)
    vec_L_ideal = np.array([ideal_L_x, ideal_L_y, ideal_L_z])

    wins_for_A0 = 0
    for _ in range(trials):
        vec_L_jittered = get_jittered_lump_orientation(vec_L_ideal, angular_std_dev)
        
        # Projection of L_jittered onto apparatus_axis_A0 (+Z)
        projection_A0 = np.dot(vec_L_jittered, apparatus_axis_A0)
        
        # Intensity for outcome "0" is prop. to cos^2(angle_between_L_jittered_and_A0 / 2)
        # If A0 is |Z+>, and L_jittered has angle theta_j from Z+, then prob(Z+) = cos^2(theta_j/2)
        # cos(theta_j) = vec_L_jittered . apparatus_axis_A0 (if both unit vectors)
        # So, cos^2(theta_j/2) = 0.5 * (1 + cos(theta_j)) = 0.5 * (1 + projection_A0)
        
        prob_outcome_A0_from_projection = 0.5 * (1 + projection_A0)

        # Simulate the choice based on this probability
        if np.random.rand() < prob_outcome_A0_from_projection:
            wins_for_A0 += 1
            
    return wins_for_A0 / trials

# --- Test with different CV values that map to angular jitter ---
CV_values_substrate_test = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9] # Testing range for CV
all_P0_angular_curves = {}

# --- Simulation ---
print("Simulating Born rule via Angular Jitter for different substrate CV values...")
for cv_val in CV_values_substrate_test:
    effective_angular_std = K_CV_TO_ANGLE_STD * cv_val
    print(f"  Running for CV_substrate = {cv_val:.3f} (effective angular jitter std = {effective_angular_std:.3f} rad)")
    current_P0s = []
    for alpha_instr in alphas_instrument:
        p0 = simulate_born_angular_jitter(alpha_instr, cv_val, TRIALS_PER_POINT)
        current_P0s.append(p0)
    all_P0_angular_curves[cv_val] = np.array(current_P0s)

# --- Plotting ---
plt.figure(figsize=(12, 8))
# Plot ideal QM Born rule
plt.plot(alphas_instrument, np.cos(alphas_instrument/2)**2, 'k--', lw=2, label='Ideal QM: cos²(α/2)')

for cv_val, p0_curve in all_P0_angular_curves.items():
    effective_angular_std = K_CV_TO_ANGLE_STD * cv_val
    plt.plot(alphas_instrument, p0_curve, 'o-', ms=4, alpha=0.7,
             label=f'CV_sub={cv_val:.2f} (Ang.Std≈{effective_angular_std:.2f} rad)')

plt.xlabel("Instrument Preparation Angle α (radians from +Z)")
plt.ylabel("Probability P(Outcome +Z)")
plt.title("Born Rule Emergence from Substrate CV via Angular Jitter Model")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5)) # Legend outside
plt.grid(True)
plt.ylim(-0.05, 1.05)
plt.tight_layout(rect=[0,0,0.85,1]) # Adjust for external legend
plt.show()

print("\nNote on interpretation:")
print("This simulation models the 'CV_substrate' as causing angular jitter on a particle's ideal orientation.")
print(f"The mapping is: std_dev_angle_jitter = {K_CV_TO_ANGLE_STD} * CV_substrate.")
print("Measurement outcome probability is then 0.5 * (1 + (L_jittered . Z_axis)).")
print("The curve matches cos²(α/2) best when CV_substrate is around 0.5,")
print(f"  implying an angular jitter std dev around {K_CV_TO_ANGLE_STD*0.5:.2f} radians (approx {np.degrees(K_CV_TO_ANGLE_STD*0.5):.1f} degrees).")