# Polyrhythmic Membrane: Microphone Noise Analyzer

(Edit: Added qubit gem that explores it without mic) 

This application is an experimental tool designed to explore the "Polyrhythmic Membrane Theory," which proposes that quantum mechanical statistics (specifically the Born rule) can emerge from a classical substrate exhibiting specific noise characteristics.

The program analyzes live audio input from a microphone to:
1.  Calculate the real-time Coefficient of Variation (CV) of the noise envelope.
2.  Estimate the slope of the noise's power spectrum.
3.  Simulate a "quantum measurement" (Born rule probability `P(0|α) = cos²(α/2)`) using a "compare two noisy intensities" mechanism, where the noise level for the simulation is derived from the live microphone noise's CV.
4.  Simulate a Bell Test (CHSH S-value) where the local measurement noise and the correlation strength between simulated particles are influenced by the live microphone noise's CV.
5.  Display these statistics and plots in a real-time GUI.

## Theory Highlights

The core theoretical ideas being tested are:
1.  **Classical Origin of Quantum Probabilities:** When noise (from the environment or an underlying "aether") has an amplitude envelope with a Coefficient of Variation (CV) ≈ 0.5, a simple classical decision rule ("compare two noisy intensities") can reproduce the Born rule probabilities of quantum mechanics. A 1/f power spectrum for the noise is often associated with this CV.
2.  **Emergent Quantum Mechanics:** This suggests that quantum mechanics might be an effective statistical description of a deeper classical reality that possesses these specific noise characteristics.
3.  **Bell Test Limitations:** The simulation also explores if such local noise characteristics, even when correlated between two "particles," are sufficient to violate Bell's inequality (S > 2). The theory suggests that S > 2 likely requires additional non-local mechanisms (e.g., FTL links through a substrate) not directly testable with this live microphone noise setup alone for the Bell test.

## Features

*   Live audio capture from microphone.
*   Real-time calculation of noise envelope CV and power spectrum slope.
*   Real-time simulation and plotting of Born rule emergence driven by microphone noise characteristics.
*   Real-time simulation and plotting of CHSH S-value evolution for a Bell test, with parameters influenced by microphone noise CV.
*   "Quantum Emergence Quality" indicator based on how close the live CV is to the theoretical optimum of ~0.5.
*   User interface to start/stop capture and reset statistics.

## How to Run

1.  Ensure you have Python 3 installed.
2.  Install the required libraries (see `requirements.txt`).

    pip install -r requirements.txt

3.  Run the script:

    python app.py 

5.  Click "Start Capture" to begin processing microphone noise.

6.  Observe the live statistics and plots.

## Interpreting the Output

*   **CV:** Watch the "Current Statistics" panel. A CV around 0.5 is considered optimal for Born rule emergence in this theory. The "Quantum Emergence Quality" bar reflects this.
*   **Spectrum Slope:** A slope around -1.0 (for power spectrum) or -0.5 (for amplitude spectrum) indicates 1/f (pink) noise.
*   **Born Rule Plot:** Observe how well the blue dots ("Mic Noise Sim") fit the red theoretical curve (`cos²(α/2)`). The title will show the effective noise CV used for generating the blue dots. A low "Born Err" indicates a good fit.
*   **Bell Test Plot:** Observe the S-value. Values consistently above 2.0 would violate Bell's inequality (not expected with this specific simulation logic using only correlated local noise).

## Disclaimer

This is an experimental and exploratory tool based on a novel theoretical framework. It is intended for research and educational purposes. The Bell test simulation within this specific application, relying on correlating local noise, is not expected to violate Bell's inequality (S>2) due to the constraints of local realistic models.
