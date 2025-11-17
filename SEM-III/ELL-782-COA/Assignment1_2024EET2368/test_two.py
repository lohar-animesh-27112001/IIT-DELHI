import math

# Test Case 2 inputs
x1_real = 2.0
x1_imag = 1.0
x2_real = -1.0
x2_imag = -1.0
w_real = 1.0
w_imag = 0.0

# Calculate W * x2
w_x2_real = w_real * x2_real - w_imag * x2_imag
w_x2_imag = w_real * x2_imag + w_imag * x2_real

# Calculate outputs
y1_real = x1_real + w_x2_real
y1_imag = x1_imag + w_x2_imag
y2_real = x1_real - w_x2_real
y2_imag = x1_imag - w_x2_imag

print("Expected results (floating-point):")
print(f"y1_real = {y1_real:.4f}")
print(f"y1_imag = {y1_imag:.4f}")
print(f"y2_real = {y2_real:.4f}")
print(f"y2_imag = {y2_imag:.4f}")

# Convert to Q8.8 fixed-point
def float_to_q88(value):
    return int(round(value * 256)) & 0xFFFF

print("\nExpected results (Q8.8 hex):")
print(f"y1_real = 0x{float_to_q88(y1_real):04X}")
print(f"y1_imag = 0x{float_to_q88(y1_imag):04X}")
print(f"y2_real = 0x{float_to_q88(y2_real):04X}")
print(f"y2_imag = 0x{float_to_q88(y2_imag):04X}")