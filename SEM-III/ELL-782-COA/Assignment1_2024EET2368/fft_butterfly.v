module fft_butterfly (
    input  wire clk,
    input  wire rst,
    input  wire signed [15:0] x1_real,
    input  wire signed [15:0] x1_imag,
    input  wire signed [15:0] x2_real,
    input  wire signed [15:0] x2_imag,
    input  wire signed [15:0] w_real,
    input  wire signed [15:0] w_imag,
    output reg signed [15:0] y1_real,
    output reg signed [15:0] y1_imag,
    output reg signed [15:0] y2_real,
    output reg signed [15:0] y2_imag
    );

    wire signed [15:0] add_real, add_imag, sub_real, sub_imag;
    compute compute_main (
        .x1_real(x1_real),
        .x1_imag(x1_imag),
        .x2_real(x2_real),
        .x2_imag(x2_imag),
        .w_real(w_real),
        .w_imag(w_imag),
        .add_real(add_real),
        .add_imag(add_imag),
        .sub_real(sub_real),
        .sub_imag(sub_imag)
    );

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            y1_real <= 0;
            y1_imag <= 0;
            y2_real <= 0;
            y2_imag <= 0;
        end else begin
            y1_real <= add_real;
            y1_imag <= add_imag;
            y2_real <= sub_real;
            y2_imag <= sub_imag;
        end
    end

endmodule