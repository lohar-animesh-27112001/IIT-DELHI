module fft_64 (
    input wire clk, // system clock
    input wire rst, // reset signal
    input wire start, // start signal to begin FFT computation
    // 16 bits signed - input & output data
    input wire signed [15:0] in_real, // input real data
    input wire signed [15:0] in_imag, // input imaginary data
    output reg signed [15:0] out_real, // output real data
    output reg signed [15:0] out_imag, // output imaginary data
    output reg valid, // asserts when output data is valid
    output reg ready // asserts when module is ready for new input
);

    // parameters
    parameter size = 32; // size of the twiddle factor lookup table
    // twiddle factor LUT (precomputed Q8.8 values for 64-point FFT)
    reg signed [15:0] twiddle_real[0:size-1];
    reg signed [15:0] twiddle_imag[0:size-1];
    // initialize twiddle factors (Q8.8 format)
    initial begin
        // precomputed twiddle factors for 64-point FFT
        twiddle_real[0] = 16'h0100; twiddle_imag[0] = 16'h0000;
        twiddle_real[1] = 16'h00FF; twiddle_imag[1] = 16'hFFE7;
        twiddle_real[2] = 16'h00FB; twiddle_imag[2] = 16'hFFCE;
        twiddle_real[3] = 16'h00F5; twiddle_imag[3] = 16'hFFB6;
        twiddle_real[4] = 16'h00ED; twiddle_imag[4] = 16'hFF9E;
        twiddle_real[5] = 16'h00E2; twiddle_imag[5] = 16'hFF87;
        twiddle_real[6] = 16'h00D5; twiddle_imag[6] = 16'hFF72;
        twiddle_real[7] = 16'h00C6; twiddle_imag[7] = 16'hFF5E;
        twiddle_real[8] = 16'h00B5; twiddle_imag[8] = 16'hFF4B;
        twiddle_real[9] = 16'h00A2; twiddle_imag[9] = 16'hFF3A;
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
    // parameters
    parameter N = 64; // FFT size
    parameter stages = 6; // 6 because a radix-2 FFT has log2(N) stages
    // Memory for intermediate results
    reg signed [15:0] mem_real[0:N-1];
    reg signed [15:0] mem_imag[0:N-1];
    // Butterfly unit signals
    wire signed [15:0] bf_y1_real;
    wire signed [15:0] bf_y1_imag;
    wire signed [15:0] bf_y2_real;
    wire signed [15:0] bf_y2_imag;
    // Internal signals
    // These counters track which stage (0–5), which butterfly pair (0–31), and which memory location to read/write
    reg [5:0] butterfly_counter;
    reg [5:0] mem_read_addr1;
    reg [5:0] mem_read_addr2;
    reg [5:0] mem_write_addr;
    reg [5:0] stage_counter;
    reg [5:0] read_counter;
    reg [5:0] write_counter;
    // Instantiate butterfly unit
    fft_butterfly butterfly_unit (
        .clk(clk),
        .rst(rst),
        .x1_real(mem_real[mem_read_addr1]),
        .x1_imag(mem_imag[mem_read_addr1]),
        .x2_real(mem_real[mem_read_addr2]),
        .x2_imag(mem_imag[mem_read_addr2]),
        .w_real(twiddle_real[butterfly_counter[4:0]]),
        .w_imag(twiddle_imag[butterfly_counter[4:0]]),
        .y1_real(bf_y1_real),
        .y1_imag(bf_y1_imag),
        .y2_real(bf_y2_real),
        .y2_imag(bf_y2_imag)
    );
    // state machine states: control signals
    parameter IDLE = 3'b000; // waits for start signal
    parameter LOAD = 3'b001; // loads input data into memory
    parameter PROCESS = 3'b010; /* performs FFT stage by stage:
    each stage uses butterfly_counter to step through pairs.
    reads from two memory addresses (mem_read_addr1, mem_read_addr2).
    sends data through butterfly. writes results back into memory.
    after finishing one stage, increments stage_counter until 6 stages are done. */
    parameter OUTPUT = 3'b011; // sequentially reads FFT results from memory and drives
    parameter DONE = 3'b100; // indicates processing is complete, and goto idle
    reg [2:0] state; // current state of the FSM
    always @(posedge clk or posedge rst) begin
        if (rst) begin // reset state: if rst = 1, then reset all counters and state
            state <= IDLE; // current state of the FSM: IDLE = 3'b000. this is because it is the initial state
            stage_counter <= 0;
            butterfly_counter <= 0;
            write_counter <= 0;
            read_counter <= 0;
            mem_write_addr <= 0;
            valid <= 0; // no output valid yet
            ready <= 1; // ready to accept new input
        end else begin // main state machine
            case (state) // here state = IDLE
                IDLE: begin
                    ready <= 1; // tells external logic it can send input
                    if (start) begin // start signal received means start = 1
                        state <= LOAD; // when start=1, it moves to LOAD state
                        ready <= 0;
                        write_counter <= 0;
                    end
                end
                LOAD: begin // LOAD state
                    // loads the 64 input samples into internal memory
                    // when all inputs are loaded (write_counter == N-1), go to PROCESS state
                    if (write_counter == N-1) begin
                        mem_real[write_counter] <= in_real;
                        mem_imag[write_counter] <= in_imag;
                        stage_counter <= 0;
                        butterfly_counter <= 0;
                        state <= PROCESS;
                    end else begin
                        // using write_counter to step through addresses
                        mem_real[write_counter] <= in_real;
                        mem_imag[write_counter] <= in_imag;
                        write_counter <= write_counter + 1;
                    end
                end
                /*This is where the FFT butterflies are executed stage by stage
                stage_counter keeps track of which FFT stage which is 0 to 5
                butterfly_counter steps through butterflies in that stage which is 0 to 31
                mem_read_addr1 and mem_read_addr2 pick the two points for the butterfly which is 0 to 63
                results (bf_y1, bf_y2) are written back into memory which is 0 to 63
                when all stages (which is 0 to 5) are done, FSM moves to output. */
                PROCESS: begin
                    if (stage_counter < stages) begin
                        // calculate memory addresses for this butterfly
                        // using Cooley-Tukey algorithm addressing
                        // write results from previous butterfly operation
                        if (butterfly_counter > 0) begin
                            mem_read_addr1 <= butterfly_counter;
                            mem_read_addr2 <= butterfly_counter + (1 << stage_counter);
                            mem_real[mem_write_addr] <= bf_y1_real;
                            mem_imag[mem_write_addr] <= bf_y1_imag;
                            mem_real[mem_write_addr + (1 << stage_counter)] <= bf_y2_real;
                            mem_imag[mem_write_addr + (1 << stage_counter)] <= bf_y2_imag;
                        end else begin
                            mem_read_addr1 <= butterfly_counter;
                            mem_read_addr2 <= butterfly_counter + (1 << stage_counter);
                        end
                        // store current addresses for next cycle write
                        if (butterfly_counter < N/2 - 1) begin
                            mem_write_addr <= butterfly_counter;
                            butterfly_counter <= butterfly_counter + 1;
                        end else begin
                            // write last butterfly results
                            mem_write_addr <= butterfly_counter;
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
                // sequentially outputs the final FFT results from memory. valid=1 indicates output is usable.
                // uses read_counter to step through results. when all outputs are sent, go to done.
                OUTPUT: begin
                    // output results
                    if (read_counter == N-1) begin
                        out_real <= mem_real[read_counter];
                        out_imag <= mem_imag[read_counter];
                        valid <= 1;
                        state <= DONE;
                    end else begin
                        out_real <= mem_real[read_counter];
                        out_imag <= mem_imag[read_counter];
                        valid <= 1;
                        read_counter <= read_counter + 1;
                    end
                end
                // FFT process is complete. return to IDLE for the next run.
                DONE: begin
                    valid <= 0;
                    ready <= 1;
                    state <= IDLE;
                end
            endcase
        end
    end

endmodule