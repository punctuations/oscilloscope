import numpy as np
from scipy.io.wavfile import write

freq = 440
duration = 60

# Generate time values for waveform
t = np.linspace(0, duration, int(duration * 44100), endpoint=False)

# Create waveform for left channel
left_wave = np.cos(2 * np.pi * freq * t)

# Create waveform for right channel
right_wave = np.sin(2 * np.pi * freq * t)

# Combine into one audio signal
stereo_waveform = np.column_stack([left_wave, right_wave])

write('examples/circle.wav', 44100, stereo_waveform)
