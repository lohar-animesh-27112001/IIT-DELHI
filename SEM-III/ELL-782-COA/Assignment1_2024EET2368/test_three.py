import numpy as np
import math

# Generate test case 3 input
N = 64
input_real = []
input_imag = []

print("Input data for Test Case 3:")
for k in range(N):
    angle = 2 * math.pi * k / N
    real_val = math.cos(angle)
    imag_val = math.sin(angle)
    input_real.append(real_val)
    input_imag.append(imag_val)
    print(f"Sample {k}: real = {real_val:.6f}, imag = {imag_val:.6f}")

# Convert to complex array for FFT
input_complex = [complex(r, i) for r, i in zip(input_real, input_imag)]

# Compute FFT using numpy
fft_result = np.fft.fft(input_complex)

print("\nExpected FFT results for Test Case 3:")
for k in range(N):
    print(f"Bin {k}: real = {fft_result[k].real:.6f}, imag = {fft_result[k].imag:.6f}")

# The expected result should have energy only at bin 1
print(f"\nMain energy should be at bin 1: {fft_result[1].real:.6f} + {fft_result[1].imag:.6f}j")