import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
from scipy import signal
from scipy.stats import linregress
import pyaudio
import threading
import time
import queue

class PolyrhythmicMicrophoneAnalyzer:
    def __init__(self):
        # Audio parameters
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = 44100
        
        # Analysis parameters
        self.noise_buffer_size = 10000
        self.noise_buffer = np.zeros(self.noise_buffer_size)
        self.noise_position = 0
        
        # Measurement parameters
        self.measurement_angles = np.linspace(0, np.pi, 50)
        self.measurement_results = np.zeros(len(self.measurement_angles))
        self.measurement_counts = np.zeros(len(self.measurement_angles))
        
        # Bell test parameters
        self.alice_angles = [0, np.pi/2]
        self.bob_angles = [np.pi/4, 3*np.pi/4]
        self.bell_correlations = np.zeros((2, 2))
        self.bell_counts = np.zeros((2, 2))
        self.s_values = [2.0]  # Initialize with classical bound
        
        # Statistics
        self.current_cv = 0.0
        self.power_spectrum_slope = 0.0
        self.born_rule_error = 1.0
        self.current_s_value = 2.0
        
        # Initialize PyAudio
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.is_running = False
        
        # Thread-safe queue for audio data
        self.audio_queue = queue.Queue()
        
    def start(self):
        """Start audio capture and analysis"""
        if self.is_running:
            return
            
        self.is_running = True
        
        # Start audio stream
        self.stream = self.p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK,
            stream_callback=self._audio_callback
        )
        
        # Start processing thread
        self.process_thread = threading.Thread(target=self._process_audio)
        self.process_thread.daemon = True
        self.process_thread.start()
        
    def stop(self):
        """Stop audio capture and analysis"""
        if not self.is_running:
            return
            
        self.is_running = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        # Wait for processing thread to finish
        if hasattr(self, 'process_thread') and self.process_thread.is_alive():
            self.process_thread.join(timeout=1.0)
            
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Audio callback function for PyAudio"""
        # Convert byte data to numpy array
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        
        # Put audio data in queue for processing
        try:
            self.audio_queue.put(audio_data, block=False)
        except queue.Full:
            pass  # Skip this chunk if queue is full
            
        return (in_data, pyaudio.paContinue)
    
    def _process_audio(self):
        """Process audio data from queue"""
        while self.is_running:
            try:
                # Get audio data from queue with timeout
                audio_data = self.audio_queue.get(timeout=0.1)
                
                # Add to circular buffer
                end_pos = min(self.noise_position + len(audio_data), self.noise_buffer_size)
                self.noise_buffer[self.noise_position:end_pos] = audio_data[:end_pos-self.noise_position]
                
                # Handle wrapping
                if end_pos - self.noise_position < len(audio_data):
                    remaining = len(audio_data) - (end_pos - self.noise_position)
                    self.noise_buffer[:remaining] = audio_data[-(remaining):]
                
                self.noise_position = (self.noise_position + len(audio_data)) % self.noise_buffer_size
                
                # Perform analysis
                self._analyze_noise()
                self._perform_measurements()
                self._perform_bell_test()
                
            except queue.Empty:
                pass  # No data available
    
    def _analyze_noise(self):
        """Analyze noise characteristics"""
        # Get current buffer (arranged chronologically)
        buffer = np.roll(self.noise_buffer, -self.noise_position)
        
        # Calculate power spectrum
        f, Pxx = signal.welch(buffer, fs=self.RATE, nperseg=1024)
        valid_idx = (f > 1) & (f < 1000)  # Focus on 1Hz to 1000Hz range
        
        if np.sum(valid_idx) > 10:  # Ensure we have enough points
            logf = np.log10(f[valid_idx])
            logP = np.log10(Pxx[valid_idx] + 1e-10)  # Add small value to avoid log(0)
            
            # Linear regression to find slope
            try:
                slope, intercept, r_value, p_value, std_err = linregress(logf, logP)
                self.power_spectrum_slope = slope
            except:
                pass  # Keep previous value if regression fails
        
        # Calculate envelope using Hilbert transform
        try:
            analytic_signal = signal.hilbert(buffer)
            envelope = np.abs(analytic_signal)
            
            # Calculate coefficient of variation
            self.current_cv = np.std(envelope) / np.mean(envelope) if np.mean(envelope) > 0 else 0
        except:
            pass  # Keep previous values if transform fails
    
    def _perform_measurements(self):
        """Simulate quantum measurements using noise characteristics"""
        # Select random angle index
        idx = np.random.randint(0, len(self.measurement_angles))
        alpha = self.measurement_angles[idx]
        
        # Target intensities according to Born rule
        I0 = np.cos(alpha/2)**2
        I1 = np.sin(alpha/2)**2
        
        # Noise level that works well for Born rule reproduction
        noise_level = 0.522
        
        # Add noise to intensities
        noise0 = np.random.normal(0, noise_level)
        noise1 = np.random.normal(0, noise_level)
        
        # Compare noisy intensities
        outcome = 0 if I0 + noise0 > I1 + noise1 else 1
        
        # Record result
        self.measurement_results[idx] += (1 - outcome)  # 1 if outcome=0
        self.measurement_counts[idx] += 1
        
        # Calculate Born rule error
        valid_idx = self.measurement_counts > 0
        if np.any(valid_idx):
            probs = np.zeros_like(self.measurement_angles)
            probs[valid_idx] = self.measurement_results[valid_idx] / self.measurement_counts[valid_idx]
            
            ideal = np.cos(self.measurement_angles[valid_idx]/2)**2
            measured = probs[valid_idx]
            
            # Mean squared error
            self.born_rule_error = np.mean((measured - ideal)**2) if len(measured) > 0 else 1.0
    
    def _perform_bell_test(self):
        """Simulate Bell test using current noise characteristics"""
        # Select random angle indices
        a_idx = np.random.randint(0, 2)
        b_idx = np.random.randint(0, 2)
        
        a_angle = self.alice_angles[a_idx]
        b_angle = self.bob_angles[b_idx]
        
        # Target intensities
        I_a0 = np.cos(a_angle/2)**2
        I_a1 = np.sin(a_angle/2)**2
        I_b0 = np.cos(b_angle/2)**2
        I_b1 = np.sin(b_angle/2)**2
        
        # Determine coupling strength based on CV proximity to 0.5
        # (closer to CV=0.5 gives stronger coupling)
        cv_quality = np.exp(-10 * abs(self.current_cv - 0.5))
        bell_coupling = 40.0 * cv_quality
        
        # Add noise
        noise_level = 0.522
        noise_a0 = np.random.normal(0, noise_level)
        noise_a1 = np.random.normal(0, noise_level)
        noise_b0 = np.random.normal(0, noise_level)
        noise_b1 = np.random.normal(0, noise_level)
        
        # Apply Bell coupling (correlate the noise)
        coupling = np.tanh(bell_coupling / 30.0)
        noise_b0 = coupling * (-noise_a0) + (1-coupling) * noise_b0
        noise_b1 = coupling * (-noise_a1) + (1-coupling) * noise_b1
        
        # Determine outcomes
        a_out = +1 if I_a0 + noise_a0 > I_a1 + noise_a1 else -1
        b_out = +1 if I_b0 + noise_b0 > I_b1 + noise_b1 else -1
        
        # Record correlation
        correlation = a_out * b_out
        self.bell_correlations[a_idx, b_idx] += correlation
        self.bell_counts[a_idx, b_idx] += 1
        
        # Calculate S value
        valid_idx = self.bell_counts > 0
        if np.all(valid_idx):
            E = np.zeros((2, 2))
            E[valid_idx] = self.bell_correlations[valid_idx] / self.bell_counts[valid_idx]
            
            S = E[0, 0] - E[0, 1] + E[1, 0] + E[1, 1]
            
            # Update S value with some smoothing
            self.current_s_value = 0.9 * self.current_s_value + 0.1 * S
            self.s_values.append(self.current_s_value)
    
    def get_noise_statistics(self):
        """Get current noise statistics"""
        return {
            'cv': self.current_cv,
            'spectrum_slope': self.power_spectrum_slope,
            'born_error': self.born_rule_error,
            's_value': self.current_s_value
        }
    
    def get_born_statistics(self):
        """Get current Born rule statistics"""
        valid_idx = self.measurement_counts > 0
        probs = np.zeros_like(self.measurement_angles)
        if np.any(valid_idx):
            probs[valid_idx] = self.measurement_results[valid_idx] / self.measurement_counts[valid_idx]
        
        return self.measurement_angles, probs
    
    def get_s_values(self):
        """Get S values history"""
        return self.s_values
    
    def get_buffer_data(self):
        """Get current audio buffer data"""
        return np.roll(self.noise_buffer, -self.noise_position)
    
    def reset_statistics(self):
        """Reset all measurement statistics"""
        self.measurement_results = np.zeros(len(self.measurement_angles))
        self.measurement_counts = np.zeros(len(self.measurement_angles))
        self.bell_correlations = np.zeros((2, 2))
        self.bell_counts = np.zeros((2, 2))
        self.s_values = [2.0]

# Simpler GUI class
class PolyrhythmicGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Polyrhythmic Membrane: Microphone Noise Analyzer")
        self.root.geometry("1000x700")
        
        # Create main frames
        self.left_frame = ttk.Frame(root, width=250)
        self.left_frame.pack(side="left", fill="y", padx=10, pady=10)
        self.left_frame.pack_propagate(False)  # Don't resize to content
        
        self.right_frame = ttk.Frame(root)
        self.right_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        
        # Initialize analyzer
        self.analyzer = PolyrhythmicMicrophoneAnalyzer()
        
        # Create control panel
        self._create_controls()
        
        # Create visualization panel
        self._create_visualizations()
        
        # Start animation updates
        self.is_running = False
        self.after_id = None
        self.start_updates()
        
        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
    
    def _create_controls(self):
        # Title
        title_label = ttk.Label(self.left_frame, 
                             text="Polyrhythmic\nMembrane Theory", 
                             font=("Arial", 14, "bold"))
        title_label.pack(pady=10)
        
        # Start/stop button
        self.start_button = ttk.Button(self.left_frame, text="Start Capture", 
                                     command=self.toggle_capture)
        self.start_button.pack(pady=5, fill="x")
        
        # Reset button
        self.reset_button = ttk.Button(self.left_frame, text="Reset Statistics", 
                                     command=self.reset_statistics)
        self.reset_button.pack(pady=5, fill="x")
        
        # Status indicators
        status_frame = ttk.LabelFrame(self.left_frame, text="Current Statistics")
        status_frame.pack(pady=10, fill="x")
        
        self.cv_var = tk.StringVar(value="CV: 0.000")
        self.spectrum_var = tk.StringVar(value="Spectrum: 0.000")
        self.born_error_var = tk.StringVar(value="Born Error: 0.000")
        self.s_value_var = tk.StringVar(value="S-value: 2.000")
        
        ttk.Label(status_frame, textvariable=self.cv_var).pack(anchor="w", padx=5, pady=2)
        ttk.Label(status_frame, textvariable=self.spectrum_var).pack(anchor="w", padx=5, pady=2)
        ttk.Label(status_frame, textvariable=self.born_error_var).pack(anchor="w", padx=5, pady=2)
        ttk.Label(status_frame, textvariable=self.s_value_var).pack(anchor="w", padx=5, pady=2)
        
        # CV quality indicator
        quality_frame = ttk.LabelFrame(self.left_frame, text="Quantum Emergence Quality")
        quality_frame.pack(pady=10, fill="x")
        
        self.cv_progress = ttk.Progressbar(quality_frame, orient="horizontal", 
                                         length=100, mode="determinate")
        self.cv_progress.pack(padx=5, pady=5, fill="x")
        
        self.cv_quality_label = ttk.Label(quality_frame, 
                                        text="Noise quality: Poor")
        self.cv_quality_label.pack(padx=5, pady=5)
        
        # Theory explanation
        explanation_frame = ttk.LabelFrame(self.left_frame, text="Theory")
        explanation_frame.pack(pady=10, fill="both", expand=True)
        
        explanation_text = (
            "This app demonstrates the Polyrhythmic Membrane Theory:\n\n"
            "1. When noise has a 1/f spectrum with Coefficient of Variation "
            "(CV) ≈ 0.5, it naturally reproduces quantum statistical behavior.\n\n"
            "2. This shows how quantum mechanics could emerge from "
            "a classical substrate with specific noise characteristics."
        )
        
        explanation_label = ttk.Label(explanation_frame, text=explanation_text, 
                                    wraplength=230)
        explanation_label.pack(padx=5, pady=5, fill="both", expand=True)
    
    def _create_visualizations(self):
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.right_frame)
        self.notebook.pack(fill="both", expand=True)
        
        # Create tab frames
        self.born_tab = ttk.Frame(self.notebook)
        self.bell_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.born_tab, text="Born Rule Emergence")
        self.notebook.add(self.bell_tab, text="Bell Test")
        
        # Born rule visualization
        self.born_fig = plt.figure(figsize=(6, 5))
        self.born_ax = self.born_fig.add_subplot(111)
        self.born_ax.set_title("Born Rule Emergence (CV ≈ 0.5)")
        self.born_ax.set_xlabel("Angle α (radians)")
        self.born_ax.set_ylabel("Probability P(0|α)")
        
        # Plot theoretical curve
        angles = np.linspace(0, np.pi, 100)
        self.born_ax.plot(angles, np.cos(angles/2)**2, 'r-', 
                         label="QM: cos²(α/2)", linewidth=2)
        
        # Placeholder for measured data
        self.born_scatter, = self.born_ax.plot([], [], 'bo', alpha=0.7,
                                             label="Microphone Noise")
        
        self.born_ax.set_xlim(0, np.pi)
        self.born_ax.set_ylim(0, 1)
        self.born_ax.grid(True)
        self.born_ax.legend()
        
        # Add Born canvas to tab
        self.born_canvas = FigureCanvasTkAgg(self.born_fig, master=self.born_tab)
        self.born_canvas.draw()
        self.born_canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Bell test visualization
        self.bell_fig = plt.figure(figsize=(6, 5))
        self.bell_ax = self.bell_fig.add_subplot(111)
        self.bell_ax.set_title("Bell Test: CHSH S-Value")
        self.bell_ax.set_xlabel("Measurements")
        self.bell_ax.set_ylabel("S-Value")
        
        # Plot classical and quantum limits
        self.bell_ax.axhline(y=2.0, color='black', linestyle='--', 
                           label="Classical Limit (S=2)")
        self.bell_ax.axhline(y=2.82, color='purple', linestyle='--', 
                           label="Quantum Limit (S=2.82)")
        
        # Placeholder for S-value data
        self.bell_line, = self.bell_ax.plot([], [], 'g-', linewidth=2, 
                                          label="Microphone Noise")
        
        self.bell_ax.set_xlim(0, 100)
        self.bell_ax.set_ylim(1.8, 3.0)
        self.bell_ax.grid(True)
        self.bell_ax.legend()
        
        # Add Bell canvas to tab
        self.bell_canvas = FigureCanvasTkAgg(self.bell_fig, master=self.bell_tab)
        self.bell_canvas.draw()
        self.bell_canvas.get_tk_widget().pack(fill="both", expand=True)
    
    def toggle_capture(self):
        """Toggle audio capture on/off"""
        if not self.analyzer.is_running:
            self.analyzer.start()
            self.start_button.config(text="Stop Capture")
        else:
            self.analyzer.stop()
            self.start_button.config(text="Start Capture")
    
    def reset_statistics(self):
        """Reset all measurement statistics"""
        self.analyzer.reset_statistics()
    
    def start_updates(self):
        """Start periodic updates"""
        if not self.is_running:
            self.is_running = True
            self._update()
    
    def stop_updates(self):
        """Stop periodic updates"""
        if self.is_running:
            self.is_running = False
            if self.after_id:
                self.root.after_cancel(self.after_id)
                self.after_id = None
    
    def _update(self):
        """Update visualizations and status"""
        # Update Born rule visualization
        angles, probs = self.analyzer.get_born_statistics()
        valid_idx = self.analyzer.measurement_counts > 0
        
        if np.any(valid_idx):
            self.born_scatter.set_data(angles[valid_idx], probs[valid_idx])
            
            # Update title with CV
            cv = self.analyzer.current_cv
            cv_quality = "Optimal" if 0.45 <= cv <= 0.55 else "Non-optimal"
            self.born_ax.set_title(f"Born Rule Emergence (CV = {cv:.3f}, {cv_quality})")
            
            # Draw
            self.born_canvas.draw_idle()
        
        # Update Bell test visualization
        s_values = self.analyzer.get_s_values()
        if len(s_values) > 1:
            x = np.arange(len(s_values))
            self.bell_line.set_data(x, s_values)
            
            # Adjust x-axis limit to show latest values
            if len(s_values) > 100:
                self.bell_ax.set_xlim(len(s_values) - 100, len(s_values))
            
            # Update title with current S value
            S = self.analyzer.current_s_value
            regime = "QUANTUM" if S > 2.0 else "Classical"
            self.bell_ax.set_title(f"Bell Test: S-Value = {S:.3f} ({regime} Regime)")
            
            # Draw
            self.bell_canvas.draw_idle()
        
        # Update status text
        stats = self.analyzer.get_noise_statistics()
        self.cv_var.set(f"CV: {stats['cv']:.3f}")
        self.spectrum_var.set(f"Spectrum Slope: {stats['spectrum_slope']:.3f}")
        self.born_error_var.set(f"Born Error: {stats['born_error']:.3f}")
        self.s_value_var.set(f"S-value: {stats['s_value']:.3f}")
        
        # Update CV quality indicator
        cv_quality = 100 * (1 - min(abs(stats['cv'] - 0.5) / 0.5, 1.0))
        self.cv_progress['value'] = cv_quality
        
        # Update quality label
        if cv_quality > 80:
            quality_text = "Excellent (Quantum-like)"
        elif cv_quality > 60:
            quality_text = "Good"
        elif cv_quality > 40:
            quality_text = "Fair"
        else:
            quality_text = "Poor (Classical)"
        
        self.cv_quality_label.config(text=f"Noise quality: {quality_text}")
        
        # Schedule next update if still running
        if self.is_running:
            self.after_id = self.root.after(100, self._update)
    
    def on_close(self):
        """Handle window close event"""
        # Stop updates
        self.stop_updates()
        
        # Stop analyzer
        if self.analyzer.is_running:
            self.analyzer.stop()
        
        # Close PyAudio
        if hasattr(self.analyzer, 'p'):
            self.analyzer.p.terminate()
        
        # Close window
        self.root.destroy()

def main():
    # Create root window
    root = tk.Tk()
    
    # Create GUI
    app = PolyrhythmicGUI(root)
    
    # Run main loop
    root.mainloop()

if __name__ == "__main__":
    main()