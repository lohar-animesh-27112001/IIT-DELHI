module compute (
    input  wire signed [15:0] x1_real,
    input  wire signed [15:0] x1_imag,
    input  wire signed [15:0] x2_real,
    input  wire signed [15:0] x2_imag,
    input  wire signed [15:0] w_real,
    input  wire signed [15:0] w_imag,
    output reg signed [15:0] add_real,
    output reg signed [15:0] add_imag,
    output reg signed [15:0] sub_real,
    output reg signed [15:0] sub_imag
    );

    reg signed [31:0] mult1, mult2, mult3, mult4;
    reg signed [31:0] t_real, t_imag;
    always @(*) begin
        mult1 = w_real * x2_real;
        mult2 = w_imag * x2_imag;
        mult3 = w_real * x2_imag;
        mult4 = w_imag * x2_real;
        t_real = (mult1 - mult2) >>> 8;
        t_imag = (mult3 + mult4) >>> 8;
        add_real = x1_real + t_real[15:0];
        add_imag = x1_imag + t_imag[15:0];
        sub_real = x1_real - t_real[15:0];
        sub_imag = x1_imag - t_imag[15:0];
    end

endmodule