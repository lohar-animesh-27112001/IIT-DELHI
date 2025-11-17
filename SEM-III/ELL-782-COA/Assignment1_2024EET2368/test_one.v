`timescale 1ns/1ps

module test_one;

    reg clk, rst;
    reg signed [15:0] x1_real, x1_imag, x2_real, x2_imag;
    reg signed [15:0] w_real, w_imag;
    wire signed [15:0] y1_real, y1_imag, y2_real, y2_imag;
    fft_butterfly test (
        .clk(clk),
        .rst(rst),
        .x1_real(x1_real),
        .x1_imag(x1_imag),
        .x2_real(x2_real),
        .x2_imag(x2_imag),
        .w_real(w_real),
        .w_imag(w_imag),
        .y1_real(y1_real),
        .y1_imag(y1_imag),
        .y2_real(y2_real),
        .y2_imag(y2_imag)
    );
    always #5 clk = ~clk;

    initial begin
        $dumpfile("./VCD/test_one.vcd");
        $dumpvars(0, test_one);
        clk = 0; rst = 1;
        #10 rst = 0;
        // test Case 1
        x1_real = 16'h0100; x1_imag = 16'h0000;
        x2_real = 16'h0000; x2_imag = 16'h0100;
        w_real  = 16'h00B5; w_imag  = 16'h00B5;
        #20;
        $display("Test Case 1 Results:");
        $display("y1 = %d + j%d", y1_real, y1_imag);
        $display("y2 = %d + j%d", y2_real, y2_imag);
        $display("Floating Point: y1 = %f + j%f", $itor(y1_real)/256.0, $itor(y1_imag)/256.0);
        $display("Floating Point: y2 = %f + j%f", $itor(y2_real)/256.0, $itor(y2_imag)/256.0);
        $finish;
    end

endmodule
