`timescale 1ns/1ps

module test_three;

    // Parameters
    parameter N = 64;
    parameter CLK_PERIOD = 10; // 100 MHz clock
    // Testbench signals
    reg clk, rst, start;
    reg signed [15:0] data_in_real, data_in_imag;
    wire signed [15:0] data_out_real, data_out_imag;
    wire valid, ready;
    // FFT module instance
    fft_64 uut (
        .clk(clk),
        .rst(rst),
        .start(start),
        .in_real(data_in_real),
        .in_imag(data_in_imag),
        .out_real(data_out_real),
        .out_imag(data_out_imag),
        .valid(valid),
        .ready(ready)
    );
    // Clock generation
    always #(CLK_PERIOD/2) clk = ~clk;
    // Test procedure
    integer k;
    real angle, cos_val, sin_val;
    real out_real, out_imag, error_real, error_imag;
    initial begin
        // Initialize
        clk = 0;
        rst = 1;
        start = 0;
        data_in_real = 0;
        data_in_imag = 0;
        // Generate dump file
        $dumpfile("./VCD/test_three.vcd");
        $dumpvars(0, test_three);
        // Reset
        #(CLK_PERIOD*2) rst = 0;
        // Wait for ready
        while (!ready) #(CLK_PERIOD);
        #(CLK_PERIOD);
        // Start loading data
        start = 1;
        #(CLK_PERIOD);
        start = 0;
        // Load input data: x[k].real = cos(2πk/64), x[k].imag = sin(2πk/64)
        for (k = 0; k < N; k = k + 1) begin
            // Calculate the floating point values
            angle = 2.0 * 3.141592653589793 * k / 64.0;
            cos_val = $cos(angle);
            sin_val = $sin(angle);
            // Convert to Q8.8 fixed-point
            data_in_real = $rtoi(cos_val * 256.0);
            data_in_imag = $rtoi(sin_val * 256.0);
            // Wait for next clock cycle
            #(CLK_PERIOD);
            // Display input values
            $display("Input[%0d]: real = %f (0x%h), imag = %f (0x%h)", 
                     k, cos_val, data_in_real, sin_val, data_in_imag);
        end
        
        // Wait for processing to complete
        while (!valid) #(CLK_PERIOD);
        // Read and verify output
        for (k = 0; k < N; k = k + 1) begin
            // Convert output to floating point for comparison
            out_real = $itor(data_out_real) / 256.0;
            out_imag = $itor(data_out_imag) / 256.0;
            // For this specific test case, we expect energy only at bin 1
            if (k == 1) begin
                // Check if we have significant energy at bin 1
                if (out_real < 30.0 || out_real > 34.0 || out_imag > 0.001 || out_imag < -0.001) begin
                    $display("ERROR: Expected energy at bin 1, got real=%f, imag=%f", out_real, out_imag);
                end
            end else begin
                // Check if other bins are near zero
                error_real = (out_real > 0) ? out_real : -out_real;
                error_imag = (out_imag > 0) ? out_imag : -out_imag;
                if (error_real > 0.001 || error_imag > 0.001) begin
                    $display("ERROR: Output[%0d] should be near zero, got real=%f, imag=%f", k, out_real, out_imag);
                end
            end
            // Display results
            $display("Output[%0d]: real = %f, imag = %f", k, out_real, out_imag);
            // Wait for next output
            #(CLK_PERIOD);
        end
        $display("Test Case 3 completed!");
        $finish;
    end
    // Simple monitor to track FFT progress
    reg [2:0] prev_state;
    initial prev_state = 0;
    always @(posedge clk) begin
        if (uut.state !== prev_state) begin
            $display("FFT state changed to: %d", uut.state);
            prev_state = uut.state;
        end
    end

endmodule