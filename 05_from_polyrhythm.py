import numpy as np
from scipy.signal import convolve2d, welch, hilbert
from scipy.stats import linregress
import matplotlib.pyplot as plt
import time

# --- Parameters for the Substrate ---
GRID_SIZE = 32  # Start smaller for faster iteration
NUM_FIELDS = 200 # Number of coupled phi fields
DT = 0.1       # Time step

# Field-specific parameters (can be tuned)
BASE_FREQUENCIES = np.linspace(0.5, 2.5, NUM_FIELDS) # Base oscillation tendency
DIFFUSION_COEFFS = np.linspace(0.1, 0.05, NUM_FIELDS) # How much they spread

# Interaction parameters
NONLINEARITY_A = 1.0  # Coeff for phi term in V'(phi) = a*phi - b*phi^3
NONLINEARITY_B = 1.0  # Coeff for phi^3 term
POLYRHYTHM_COUPLING_STRENGTH = 0.1 # How strongly fields interact
DAMPING = 0.005 # Global damping

# Analysis parameters
PSD_WINDOW_SIZE = 1024 # For Welch's method
CV_WINDOW_SIZE = 512   # For calculating running CV

class SubstrateEmergenceSimulator:
    def __init__(self):
        self.phi = [(np.random.rand(GRID_SIZE, GRID_SIZE) - 0.5) * 0.1 for _ in range(NUM_FIELDS)]
        self.phi_vel = [np.zeros((GRID_SIZE, GRID_SIZE)) for _ in range(NUM_FIELDS)]
        self.t = 0

        # For analysis
        self.local_activity_history = [] # Store sum(phi[mid_point]) over time
        self.psd_history = []
        self.cv_history = []
        self.spectral_exponent_history = []

        self.laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)

    def _non_linear_potential_deriv(self, field_component):
        return NONLINEARITY_A * field_component - NONLINEARITY_B * (field_component**3)

    def step(self):
        new_phi = [np.zeros_like(p) for p in self.phi]
        
        # Calculate coupling term (sum of all other fields influencing the current one)
        sum_all_phi = np.sum(self.phi, axis=0)

        for i in range(NUM_FIELDS):
            lap_phi_i = convolve2d(self.phi[i], self.laplacian_kernel, mode='same', boundary='wrap')
            
            potential_deriv_i = self._non_linear_potential_deriv(self.phi[i])
            
            # Polyrhythmic coupling: each field is pulled towards the average of others,
            # and also pushed by its own deviation from a "mean field" derived from others.
            # This is a simple model; more complex frequency-dependent coupling could be added.
            coupling_force = POLYRHYTHM_COUPLING_STRENGTH * (sum_all_phi - self.phi[i] * NUM_FIELDS) / (NUM_FIELDS -1 if NUM_FIELDS > 1 else 1)


            acceleration = (DIFFUSION_COEFFS[i] * lap_phi_i - 
                            potential_deriv_i + 
                            coupling_force - # Base frequencies can be added here if desired
                            DAMPING * self.phi_vel[i])
            
            self.phi_vel[i] += acceleration * DT
            new_phi[i] = self.phi[i] + self.phi_vel[i] * DT
        
        self.phi = new_phi
        self.t += DT

        # --- Analysis (done periodically) ---
        mid_point_activity = np.sum([p[GRID_SIZE//2, GRID_SIZE//2] for p in self.phi])
        self.local_activity_history.append(mid_point_activity)

        if len(self.local_activity_history) > PSD_WINDOW_SIZE:
            self.local_activity_history.pop(0) # Keep window size

            # Calculate PSD
            segment = np.array(self.local_activity_history)
            if np.std(segment) > 1e-6: # Avoid issues with flat signals
                freqs, psd = welch(segment, fs=1.0/DT, nperseg=min(len(segment), 256)) # Use shorter segments for Welch
                if len(freqs) > 1:
                    self.psd_history.append((freqs, psd))
                    if len(self.psd_history) > 100: self.psd_history.pop(0)

                    # Fit spectral exponent (log-log linear regression)
                    # Avoid zero frequencies and very low PSD values for log fit
                    valid_idx = (freqs > 0) & (psd > 1e-9) 
                    if np.sum(valid_idx) > 5:
                        log_freqs = np.log(freqs[valid_idx])
                        log_psd = np.log(psd[valid_idx])
                        slope, intercept, r_value, p_value, std_err = linregress(log_freqs, log_psd)
                        self.spectral_exponent_history.append(slope)
                        if len(self.spectral_exponent_history) > 200: self.spectral_exponent_history.pop(0)


            # Calculate Envelope and CV
            if len(segment) > CV_WINDOW_SIZE: # Need enough points for Hilbert
                analytic_signal = hilbert(segment[-CV_WINDOW_SIZE:]) # Use recent history for CV
                envelope = np.abs(analytic_signal)
                if np.mean(envelope) > 1e-6: # Avoid division by zero
                    cv = np.std(envelope) / np.mean(envelope)
                    self.cv_history.append(cv)
                    if len(self.cv_history) > 200: self.cv_history.pop(0)


# --- Visualization (Simplified for now) ---
if __name__ == "__main__":
    sim = SubstrateEmergenceSimulator()
    
    plt.ion()
    fig, axes = plt.subplots(3, 1, figsize=(8, 10))
    line_activity, = axes[0].plot([], [])
    line_psd, = axes[1].loglog([], []) # PSD on log-log
    line_cv, = axes[2].plot([], [])
    line_exponent, = axes[0].plot([], [], color='r', linestyle='--') # For spectral exponent

    axes[0].set_xlim(0, 200 * DT)
    axes[0].set_title("Local Activity (Blue) & Spectral Exp (Red)")
    axes[1].set_title("Power Spectral Density")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Power")
    axes[2].set_xlim(0, 200 * DT) # Assuming 200 CV points
    axes[2].set_ylim(0, 2.0) # Expected CV range
    axes[2].axhline(0.5, color='r', linestyle='--', label="CV=0.5")
    axes[2].set_title("Coefficient of Variation (Envelope)")
    axes[2].legend()
    plt.tight_layout()

    print_interval = 100
    num_steps_total = 9000

    for step in range(num_steps_total):
        sim.step()

        if step % print_interval == 0:
            print(f"Step: {step}, Time: {sim.t:.2f}")
            
            if sim.cv_history: print(f"  Current CV: {sim.cv_history[-1]:.3f}")
            if sim.spectral_exponent_history: print(f"  Current Spectral Exp: {sim.spectral_exponent_history[-1]:.3f}")

            # Update Activity Plot
            axes[0].set_xlim(sim.t - len(sim.local_activity_history)*DT, sim.t)
            if sim.local_activity_history:
                 line_activity.set_data(np.linspace(sim.t - len(sim.local_activity_history)*DT, sim.t, len(sim.local_activity_history)), sim.local_activity_history)
                 axes[0].set_ylim(np.min(sim.local_activity_history), np.max(sim.local_activity_history))
            
            # Update Spectral Exponent Plot on the same axes
            if sim.spectral_exponent_history:
                 # Scale exponent to fit on activity plot or use a twin axis
                 # Simple scaling for now:
                 scaled_exponents = np.array(sim.spectral_exponent_history) * (-0.1 * (np.max(sim.local_activity_history) if sim.local_activity_history else 1)) # Arbitrary scaling
                 line_exponent.set_data(np.linspace(sim.t - len(scaled_exponents)*DT, sim.t, len(scaled_exponents)), scaled_exponents)


            # Update PSD plot
            if sim.psd_history:
                freqs, psd = sim.psd_history[-1]
                line_psd.set_data(freqs, psd)
                if len(freqs)>0: axes[1].set_xlim(np.min(freqs[freqs>0]), np.max(freqs))
                if len(psd)>0: axes[1].set_ylim(np.min(psd[psd>0]), np.max(psd))


            # Update CV plot
            if sim.cv_history:
                cv_times = np.linspace(sim.t - len(sim.cv_history)*DT, sim.t, len(sim.cv_history)) # Approximate time axis
                line_cv.set_data(cv_times, sim.cv_history)
                axes[2].set_xlim(cv_times[0], cv_times[-1] + DT) # Adjust x-axis for time

            fig.canvas.draw()
            plt.pause(0.001)

    plt.ioff()
    plt.show()