import numpy as np
import random
import subprocess
import os
import re

# Parameters
N = 64
FRAC_BITS = 8
SCALE_FACTOR = 1 << FRAC_BITS  # 256 for Q8.8

def float_to_q8_8(value):
    """Convert float to Q8.8 fixed-point format"""
    fixed_val = int(np.round(value * SCALE_FACTOR))
    # Handle negative numbers using two's complement
    if fixed_val < 0:
        fixed_val = (1 << 16) + fixed_val
    return fixed_val & 0xFFFF

def q8_8_to_float(value):
    """Convert Q8.8 fixed-point format to float"""
    if value >= 0x8000:  # Negative number in two's complement
        value = value - 0x10000
    return value / SCALE_FACTOR

def generate_random_input():
    """Generate a random 64-point complex vector"""
    input_data = []
    for i in range(N):
        # Generate random values between -1 and 1
        real = random.uniform(-1.0, 1.0)
        imag = random.uniform(-1.0, 1.0)
        input_data.append(complex(real, imag))
    return input_data

def write_testbench_input(input_data, filename="random_input.txt"):
    """Write input data to a file for Verilog testbench"""
    with open(filename, 'w') as f:
        for i, sample in enumerate(input_data):
            real_q = float_to_q8_8(sample.real)
            imag_q = float_to_q8_8(sample.imag)
            f.write(f"{real_q:04x} {imag_q:04x} // Sample {i}: {sample.real:.4f} + {sample.imag:.4f}j\n")

def run_verilog_simulation():
    """Run the Verilog simulation"""
    # Compile the Verilog code
    compile_cmd = ["iverilog", "-o", "fft_test.out", "compute.v", "fft_butterfly.v", "fft_64.v", "test_four.v"]
    result = subprocess.run(compile_cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("Compilation failed:")
        print(result.stderr)
        return False
    
    # Run the simulation
    run_cmd = ["vvp", "fft_test.out"]
    result = subprocess.run(run_cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("Simulation failed:")
        print(result.stderr)
        return False
    
    print("Simulation completed successfully")
    print(result.stdout)
    return True

def parse_verilog_output(filename="verilog_output.txt"):
    """Parse the Verilog simulation output"""
    verilog_output = []
    try:
        with open(filename, 'r') as f:
            for line in f:
                if line.startswith("Output["):
                    # Use regular expressions to extract hex values
                    hex_pattern = r'[0-9a-fA-F]{4}'
                    hex_values = re.findall(hex_pattern, line)
                    
                    if len(hex_values) >= 2:
                        real_hex = hex_values[0]
                        imag_hex = hex_values[1]
                        real_float = q8_8_to_float(int(real_hex, 16))
                        imag_float = q8_8_to_float(int(imag_hex, 16))
                        verilog_output.append(complex(real_float, imag_float))
                    else:
                        print(f"Warning: Could not parse line: {line}")
    except FileNotFoundError:
        print(f"Error: Could not find output file {filename}")
        return []
    
    return verilog_output

def validate_results(verilog_output, numpy_output):
    """Validate Verilog output against NumPy FFT"""
    print("Validation Results:")
    print("Bin\tVerilog Real\tVerilog Imag\tNumPy Real\tNumPy Imag\tReal Error\tImag Error\tStatus")
    
    max_error = 0
    error_count = 0
    
    for i in range(N):
        verilog_val = verilog_output[i]
        numpy_val = numpy_output[i]
        
        real_error = abs(verilog_val.real - numpy_val.real)
        imag_error = abs(verilog_val.imag - numpy_val.imag)
        
        status = "PASS" if real_error <= 0.001 and imag_error <= 0.001 else "FAIL"
        
        if status == "FAIL":
            error_count += 1
            max_error = max(max_error, real_error, imag_error)
        
        print(f"{i}\t{verilog_val.real:.6f}\t{verilog_val.imag:.6f}\t"
              f"{numpy_val.real:.6f}\t{numpy_val.imag:.6f}\t"
              f"{real_error:.6f}\t{imag_error:.6f}\t{status}")
    
    print(f"\nSummary: {error_count} errors out of {N} bins")
    print(f"Maximum error: {max_error:.6f}")
    
    return error_count == 0

def main():
    # Generate random input
    print("Generating random 64-point input vector...")
    input_data = generate_random_input()
    
    # Write input for Verilog testbench
    write_testbench_input(input_data)
    
    # Compute expected FFT using NumPy
    print("Computing expected FFT using NumPy...")
    numpy_fft = np.fft.fft(input_data)
    
    # Run Verilog simulation
    print("Running Verilog simulation...")
    if not run_verilog_simulation():
        return
    
    # Parse Verilog output
    print("Parsing Verilog output...")
    verilog_output = parse_verilog_output()
    
    if len(verilog_output) != N:
        print(f"Error: Expected {N} output values, got {len(verilog_output)}")
        return
    
    # Validate results
    print("Validating results...")
    success = validate_results(verilog_output, numpy_fft)
    
    if success:
        print("\n✓ All results within tolerance (0.001)")
    else:
        print("\n✗ Some results exceed tolerance")
    
    # Save results for appendix
    with open("validation_results.txt", "w") as f:
        f.write("Random Input Test Case - Validation Results\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Input Data:\n")
        for i, sample in enumerate(input_data):
            f.write(f"Sample {i}: {sample.real:.6f} + {sample.imag:.6f}j\n")
        
        f.write("\nNumPy FFT Results:\n")
        for i, sample in enumerate(numpy_fft):
            f.write(f"Bin {i}: {sample.real:.6f} + {sample.imag:.6f}j\n")
        
        f.write("\nVerilog FFT Results:\n")
        for i, sample in enumerate(verilog_output):
            f.write(f"Bin {i}: {sample.real:.6f} + {sample.imag:.6f}j\n")
        
        f.write("\nValidation Summary:\n")
        f.write(f"All results within tolerance: {success}\n")
    
    print("Detailed results saved to validation_results.txt")

if __name__ == "__main__":
    main()