`timescale 1ns/1ps

module fft_64 (
    input  wire clk,          // system clock
    input  wire rst,          // reset signal
    input  wire start,        // start signal
    input  wire signed [15:0] in_real, // input real part
    input  wire signed [15:0] in_imag, // input imag part
    output reg  signed [15:0] out_real, // output real part
    output reg  signed [15:0] out_imag, // output imag part
    output reg  valid,        // output valid flag
    output reg  ready         // ready for new input
);

    // Parameters
    parameter N       = 64;   // FFT size
    parameter STAGES  = 6;    // log2(N) = 6 for 64-point FFT
    parameter TW_SIZE = 32;   // Twiddle table size (N/2)

    // Twiddle factor LUT (Q8.8 fixed-point format)
    reg signed [15:0] twiddle_real[0:TW_SIZE-1];
    reg signed [15:0] twiddle_imag[0:TW_SIZE-1];

    initial begin
        // Precomputed values (cos/sin scaled to Q8.8)
        twiddle_real[0]  = 16'h0100; twiddle_imag[0]  = 16'h0000; // 1.0
        twiddle_real[1]  = 16'h00FF; twiddle_imag[1]  = 16'hFFE7; // ~cos/sin(2π/64)
        twiddle_real[2]  = 16'h00FB; twiddle_imag[2]  = 16'hFFCE;
        twiddle_real[3]  = 16'h00F5; twiddle_imag[3]  = 16'hFFB6;
        twiddle_real[4]  = 16'h00ED; twiddle_imag[4]  = 16'hFF9E;
        twiddle_real[5]  = 16'h00E2; twiddle_imag[5]  = 16'hFF87;
        twiddle_real[6]  = 16'h00D5; twiddle_imag[6]  = 16'hFF72;
        twiddle_real[7]  = 16'h00C6; twiddle_imag[7]  = 16'hFF5E;
        twiddle_real[8]  = 16'h00B5; twiddle_imag[8]  = 16'hFF4B;
        twiddle_real[9]  = 16'h00A2; twiddle_imag[9]  = 16'hFF3A;
        twiddle_real[10] = 16'h008E; twiddle_imag[10] = 16'hFF2B;
        twiddle_real[11] = 16'h0079; twiddle_imag[11] = 16'hFF1E;
        twiddle_real[12] = 16'h0062; twiddle_imag[12] = 16'hFF13;
        twiddle_real[13] = 16'h004A; twiddle_imag[13] = 16'hFF0B;
        twiddle_real[14] = 16'h0032; twiddle_imag[14] = 16'hFF05;
        twiddle_real[15] = 16'h0019; twiddle_imag[15] = 16'hFF01;
        twiddle_real[16] = 16'h0000; twiddle_imag[16] = 16'hFF00;
        twiddle_real[17] = 16'hFFE7; twiddle_imag[17] = 16'hFF01;
        twiddle_real[18] = 16'hFFCE; twiddle_imag[18] = 16'hFF05;
        twiddle_real[19] = 16'hFFB6; twiddle_imag[19] = 16'hFF0B;
        twiddle_real[20] = 16'hFF9E; twiddle_imag[20] = 16'hFF13;
        twiddle_real[21] = 16'hFF87; twiddle_imag[21] = 16'hFF1E;
        twiddle_real[22] = 16'hFF72; twiddle_imag[22] = 16'hFF2B;
        twiddle_real[23] = 16'hFF5E; twiddle_imag[23] = 16'hFF3A;
        twiddle_real[24] = 16'hFF4B; twiddle_imag[24] = 16'hFF4B;
        twiddle_real[25] = 16'hFF3A; twiddle_imag[25] = 16'hFF5E;
        twiddle_real[26] = 16'hFF2B; twiddle_imag[26] = 16'hFF72;
        twiddle_real[27] = 16'hFF1E; twiddle_imag[27] = 16'hFF87;
        twiddle_real[28] = 16'hFF13; twiddle_imag[28] = 16'hFF9E;
        twiddle_real[29] = 16'hFF0B; twiddle_imag[29] = 16'hFFB6;
        twiddle_real[30] = 16'hFF05; twiddle_imag[30] = 16'hFFCE;
        twiddle_real[31] = 16'hFF01; twiddle_imag[31] = 16'hFFE7;
    end

    // Memory for intermediate results
    reg signed [15:0] mem_real[0:N-1];
    reg signed [15:0] mem_imag[0:N-1];

    // Butterfly outputs
    wire signed [15:0] bf_y1_real, bf_y1_imag;
    wire signed [15:0] bf_y2_real, bf_y2_imag;

    // FSM counters
    reg [5:0] stage_counter, butterfly_counter, read_counter, write_counter, mem_write_addr;
    reg [2:0] state;

    // Twiddle index (correct Cooley–Tukey addressing)
    wire [4:0] twiddle_index;
    assign twiddle_index = (butterfly_counter << (STAGES-1-stage_counter)) & (N/2 - 1);

    // Instantiate butterfly
    fft_butterfly butterfly_unit (
        .clk(clk),
        .rst(rst),
        .x1_real(mem_real[butterfly_counter]),
        .x1_imag(mem_imag[butterfly_counter]),
        .x2_real(mem_real[butterfly_counter + (1 << stage_counter)]),
        .x2_imag(mem_imag[butterfly_counter + (1 << stage_counter)]),
        .w_real(twiddle_real[twiddle_index]),
        .w_imag(twiddle_imag[twiddle_index]),
        .y1_real(bf_y1_real),
        .y1_imag(bf_y1_imag),
        .y2_real(bf_y2_real),
        .y2_imag(bf_y2_imag)
    );

    // FSM states
    parameter IDLE   = 3'b000;
    parameter LOAD   = 3'b001;
    parameter PROCESS= 3'b010;
    parameter OUTPUT = 3'b011;
    parameter DONE   = 3'b100;

    // FSM
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state <= IDLE;
            stage_counter <= 0;
            butterfly_counter <= 0;
            write_counter <= 0;
            read_counter <= 0;
            valid <= 0;
            ready <= 1;
        end else begin
            case (state)
                IDLE: begin
                    ready <= 1;
                    if (start) begin
                        state <= LOAD;
                        ready <= 0;
                        write_counter <= 0;
                    end
                end
                LOAD: begin
                    mem_real[write_counter] <= in_real;
                    mem_imag[write_counter] <= in_imag;
                    if (write_counter == N-1) begin
                        stage_counter <= 0;
                        butterfly_counter <= 0;
                        state <= PROCESS;
                    end else begin
                        write_counter <= write_counter + 1;
                    end
                end
                PROCESS: begin
                    if (stage_counter < STAGES) begin
                        // Write results from previous butterfly
                        if (butterfly_counter > 0) begin
                            mem_real[mem_write_addr] <= bf_y1_real;
                            mem_imag[mem_write_addr] <= bf_y1_imag;
                            mem_real[mem_write_addr + (1 << stage_counter)] <= bf_y2_real;
                            mem_imag[mem_write_addr + (1 << stage_counter)] <= bf_y2_imag;
                        end
                        mem_write_addr <= butterfly_counter;
                        if (butterfly_counter < N/2 - 1) begin
                            butterfly_counter <= butterfly_counter + 1;
                        end else begin
                            // Last butterfly in stage
                            mem_real[mem_write_addr] <= bf_y1_real;
                            mem_imag[mem_write_addr] <= bf_y1_imag;
                            mem_real[mem_write_addr + (1 << stage_counter)] <= bf_y2_real;
                            mem_imag[mem_write_addr + (1 << stage_counter)] <= bf_y2_imag;
                            butterfly_counter <= 0;
                            stage_counter <= stage_counter + 1;
                        end
                    end else begin
                        state <= OUTPUT;
                        read_counter <= 0;
                    end
                end
                OUTPUT: begin
                    out_real <= mem_real[read_counter];
                    out_imag <= mem_imag[read_counter];
                    valid <= 1;
                    if (read_counter == N-1) begin
                        state <= DONE;
                    end else begin
                        read_counter <= read_counter + 1;
                    end
                end
                DONE: begin
                    valid <= 0;
                    state <= IDLE;
                end
            endcase
        end
    end

endmodule
