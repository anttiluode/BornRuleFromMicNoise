import numpy as np
from scipy.signal import convolve2d, welch, hilbert
from scipy.stats import linregress
import matplotlib.pyplot as plt
import time

# --- Parameters from the "Oh wee!" run ---
GRID_SIZE = 32
NUM_FIELDS = 200
DT = 0.1 # Note: This was 0.1 in the discussion for the "Oh wee!" plot

# Field-specific parameters (can be tuned)
# These were the defaults in the script, let's assume they were used unless specified otherwise
BASE_FREQUENCIES = np.linspace(0.5, 2.5, NUM_FIELDS)
DIFFUSION_COEFFS = np.linspace(0.1, 0.05, NUM_FIELDS)

# Interaction parameters
NONLINEARITY_A = 1.0
NONLINEARITY_B = 1.0
POLYRHYTHM_COUPLING_STRENGTH = 0.1 # This is a key parameter to explore
DAMPING = 0.005

# Analysis parameters
PSD_WINDOW_SIZE = 1024
CV_WINDOW_SIZE = 512

class SubstrateEmergenceSimulator:
    def __init__(self):
        self.phi = [(np.random.rand(GRID_SIZE, GRID_SIZE) - 0.5) * 0.1 for _ in range(NUM_FIELDS)]
        self.phi_vel = [np.zeros((GRID_SIZE, GRID_SIZE)) for _ in range(NUM_FIELDS)]
        self.t = 0

        self.local_activity_history = []
        self.psd_history = [] # Stores tuples of (freqs, psd)
        self.cv_history = []
        self.spectral_exponent_history = []
        
        # Store times for CV and exponent history for accurate plotting
        self.cv_times_history = []
        self.exponent_times_history = []


        self.laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)

    def _non_linear_potential_deriv(self, field_component):
        return NONLINEARITY_A * field_component - NONLINEARITY_B * (field_component**3)

    def step(self):
        new_phi = [np.zeros_like(p) for p in self.phi]
        sum_all_phi = np.sum(self.phi, axis=0)

        for i in range(NUM_FIELDS):
            lap_phi_i = convolve2d(self.phi[i], self.laplacian_kernel, mode='same', boundary='wrap')
            potential_deriv_i = self._non_linear_potential_deriv(self.phi[i])
            
            coupling_denominator = (NUM_FIELDS - 1) if NUM_FIELDS > 1 else 1
            coupling_force = POLYRHYTHM_COUPLING_STRENGTH * (sum_all_phi - self.phi[i] * NUM_FIELDS) / coupling_denominator

            acceleration = (DIFFUSION_COEFFS[i] * lap_phi_i -
                            potential_deriv_i +
                            coupling_force -
                            DAMPING * self.phi_vel[i])
            
            self.phi_vel[i] += acceleration * DT
            new_phi[i] = self.phi[i] + self.phi_vel[i] * DT
        
        self.phi = new_phi
        self.t += DT

        mid_point_activity = np.sum([p[GRID_SIZE//2, GRID_SIZE//2] for p in self.phi])
        self.local_activity_history.append(mid_point_activity)

        if len(self.local_activity_history) > max(PSD_WINDOW_SIZE, CV_WINDOW_SIZE): # Ensure enough history for longest window
            if len(self.local_activity_history) > PSD_WINDOW_SIZE :
                 self.local_activity_history.pop(0)


            # Calculate PSD
            segment_psd = np.array(self.local_activity_history[-PSD_WINDOW_SIZE:]) # Use full PSD window
            if np.std(segment_psd) > 1e-9: # Avoid issues with flat signals
                # Ensure nperseg is not greater than segment length
                nperseg_val = min(len(segment_psd), 256) 
                if nperseg_val > 0 : # nperseg must be > 0
                    freqs, psd = welch(segment_psd, fs=1.0/DT, nperseg=nperseg_val)
                    if len(freqs) > 1:
                        self.psd_history.append((freqs, psd))
                        if len(self.psd_history) > 100: self.psd_history.pop(0)

                        valid_idx = (freqs > 1e-6) & (psd > 1e-18) # More robust filtering
                        if np.sum(valid_idx) > 5: # Need at least a few points for robust fit
                            log_freqs = np.log(freqs[valid_idx])
                            log_psd = np.log(psd[valid_idx])
                            if len(log_freqs) > 1: # linregress needs at least 2 points
                                slope, intercept, r_value, p_value, std_err = linregress(log_freqs, log_psd)
                                self.spectral_exponent_history.append(slope)
                                self.exponent_times_history.append(self.t)
                                if len(self.spectral_exponent_history) > 200: 
                                    self.spectral_exponent_history.pop(0)
                                    self.exponent_times_history.pop(0)


            # Calculate Envelope and CV
            segment_cv = np.array(self.local_activity_history[-CV_WINDOW_SIZE:]) # Use full CV window
            if len(segment_cv) >= CV_WINDOW_SIZE and np.std(segment_cv) > 1e-9: # Need enough points for Hilbert & non-flat
                analytic_signal = hilbert(segment_cv)
                envelope = np.abs(analytic_signal)
                if np.mean(envelope) > 1e-9:
                    cv = np.std(envelope) / np.mean(envelope)
                    self.cv_history.append(cv)
                    self.cv_times_history.append(self.t)
                    if len(self.cv_history) > 200: 
                        self.cv_history.pop(0)
                        self.cv_times_history.pop(0)

if __name__ == "__main__":
    sim = SubstrateEmergenceSimulator()
    
    plt.ion()
    fig, axes = plt.subplots(3, 1, figsize=(8, 12)) # Increased figure height
    
    # --- Plot 1: Local Activity and Spectral Exponent ---
    line_activity, = axes[0].plot([], [], label="Local Activity")
    axes[0].set_title("Local Activity (Blue) & Spectral Exp (Red)")
    axes[0].set_ylabel("Activity Amplitude", color='tab:blue')
    axes[0].tick_params(axis='y', labelcolor='tab:blue')
    
    ax0_twin = axes[0].twinx() # Create a twin y-axis for spectral exponent
    line_exponent, = ax0_twin.plot([], [], color='r', linestyle='--', label="Spectral Exponent")
    ax0_twin.set_ylabel("Spectral Exponent", color='r')
    ax0_twin.tick_params(axis='y', labelcolor='r')
    ax0_twin.set_ylim(-2.5, 0) # Typical range for spectral exponents (-2 for Brownian, -1 for 1/f, 0 for white)
    ax0_twin.axhline(-1.0, color='pink', linestyle=':', label="1/f slope (-1)")
    
    # Combine legends for plot 0
    lines0, labels0 = axes[0].get_legend_handles_labels()
    lines0_twin, labels0_twin = ax0_twin.get_legend_handles_labels()
    axes[0].legend(lines0 + lines0_twin, labels0 + labels0_twin, loc='upper left')

    # --- Plot 2: Power Spectral Density ---
    line_psd, = axes[1].loglog([], [], label="PSD")
    axes[1].set_title("Power Spectral Density")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Power")
    # Add reference line for 1/f slope on PSD plot
    psd_ref_line, = axes[1].loglog([], [], 'g--', label="1/f slope ref", alpha=0.7)
    axes[1].legend(loc='upper right')


    # --- Plot 3: Coefficient of Variation ---
    line_cv, = axes[2].plot([], [], label="CV")
    axes[2].set_title("Coefficient of Variation (Envelope)")
    axes[2].set_xlabel("Time (s)") # Added x-label for time
    axes[2].set_ylabel("CV Value")
    axes[2].set_ylim(0, 2.0)
    axes[2].axhline(0.5, color='r', linestyle='--', label="Target CV=0.5")
    axes[2].legend(loc='upper right')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to prevent title overlap
    fig.suptitle("Substrate Emergence Simulation", fontsize=16)


    print_interval = 100
    # Let's use the num_steps_total from the "Oh wee!" discussion
    num_steps_total = 9000

    for step in range(num_steps_total):
        sim.step()

        if step % print_interval == 0 and step > max(PSD_WINDOW_SIZE, CV_WINDOW_SIZE) : # Start plotting after enough history
            print(f"Step: {step}, Time: {sim.t:.2f}")
            
            current_cv = sim.cv_history[-1] if sim.cv_history else float('nan')
            current_exp = sim.spectral_exponent_history[-1] if sim.spectral_exponent_history else float('nan')
            print(f"  Current CV: {current_cv:.3f}")
            print(f"  Current Spectral Exp: {current_exp:.3f}")

            # Update Activity Plot
            if sim.local_activity_history:
                activity_times = np.linspace(sim.t - len(sim.local_activity_history)*DT, sim.t, len(sim.local_activity_history))
                line_activity.set_data(activity_times, sim.local_activity_history)
                axes[0].set_xlim(activity_times[0], activity_times[-1] + DT)
                min_act, max_act = np.min(sim.local_activity_history), np.max(sim.local_activity_history)
                if max_act > min_act: # Avoid issues with flat line
                     axes[0].set_ylim(min_act - 0.1*abs(min_act), max_act + 0.1*abs(max_act))
                else:
                     axes[0].set_ylim(min_act - 0.1, max_act + 0.1)


            # Update Spectral Exponent Plot
            if sim.exponent_times_history:
                line_exponent.set_data(sim.exponent_times_history, sim.spectral_exponent_history)
                # ax0_twin will share x-axis with axes[0]

            # Update PSD plot
            if sim.psd_history:
                freqs, psd = sim.psd_history[-1]
                line_psd.set_data(freqs, psd)
                valid_freqs = freqs[freqs > 0]
                valid_psd = psd[psd > 0]
                if len(valid_freqs) > 0 and len(valid_psd) > 0:
                    axes[1].set_xlim(np.min(valid_freqs), np.max(valid_freqs))
                    axes[1].set_ylim(np.min(valid_psd), np.max(valid_psd))
                    
                    # Add 1/f reference line (slope -1)
                    # Anchor it to the first point of the actual PSD
                    ref_psd_vals = (valid_freqs[0] / valid_freqs) * valid_psd[0] 
                    psd_ref_line.set_data(valid_freqs, ref_psd_vals)


            # Update CV plot
            if sim.cv_times_history:
                line_cv.set_data(sim.cv_times_history, sim.cv_history)
                axes[2].set_xlim(sim.cv_times_history[0], sim.cv_times_history[-1] + DT)

            fig.canvas.draw()
            plt.pause(0.001)

    plt.ioff()
    print("Simulation finished. Close plot window to exit.")
    plt.show()
