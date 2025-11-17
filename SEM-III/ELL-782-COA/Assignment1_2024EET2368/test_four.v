`timescale 1ns/1ps

module debug_fft;

    // Parameters
    parameter N = 64;
    parameter CLK_PERIOD = 10;
    
    // Testbench signals
    reg clk, rst, start;
    reg signed [15:0] in_real, in_imag;
    wire signed [15:0] out_real, out_imag;
    wire valid, ready;
    
    // FFT module instance
    fft_64 uut (
        .clk(clk),
        .rst(rst),
        .start(start),
        .in_real(in_real),
        .in_imag(in_imag),
        .out_real(out_real),
        .out_imag(out_imag),
        .valid(valid),
        .ready(ready)
    );
    
    // Clock generation
    always #(CLK_PERIOD/2) clk = ~clk;
    
    // Test procedure
    integer i;
    
    initial begin
        // Initialize
        clk = 0;
        rst = 1;
        start = 0;
        in_real = 0;
        in_imag = 0;
        
        // Create dump file
        $dumpfile("debug_fft.vcd");
        $dumpvars(0, debug_fft);
        
        // Reset
        #(CLK_PERIOD*2) rst = 0;
        
        // Wait for ready
        while (!ready) #(CLK_PERIOD);
        #(CLK_PERIOD);
        
        // Start loading data - simple impulse input
        start = 1;
        #(CLK_PERIOD);
        start = 0;
        
        // Load impulse at position 0
        for (i = 0; i < N; i = i + 1) begin
            if (i == 0) begin
                in_real = 16'h0100; // 1.0 in Q8.8
                in_imag = 16'h0000; // 0.0 in Q8.8
            end else begin
                in_real = 16'h0000;
                in_imag = 16'h0000;
            end
            #(CLK_PERIOD);
        end
        
        // Wait for processing to complete
        while (!valid) #(CLK_PERIOD);
        
        // Output results
        for (i = 0; i < N; i = i + 1) begin
            $display("Output[%0d]: real = %h, imag = %h", i, out_real, out_imag);
            #(CLK_PERIOD);
        end
        
        $finish;
    end

endmodule