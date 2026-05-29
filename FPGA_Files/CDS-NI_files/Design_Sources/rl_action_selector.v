module rl_action_selector(    
    input clk, reset, start,
    input [7:0] n_candidates,
    input [13:0] action_data_start_addr,
    input signed [31:0] AF_real, // Q s2.30
    input signed [15:0] p_h_f,     // Q s1.15
    input signed [15:0] pgt1_recip, // Q s3.13 — reciprocal from prob_pgt1 BRAM
    input wire [15:0] action_data_out,
    
    output reg [13:0] best_action_addr,
    output reg [8:0] best_feature_idx,
    output reg done,

    // BRAM address outputs (to model_rom / sensor_interface)
    output reg [13:0] action_data_addr
);
    
    reg [3:0] state;
    reg [7:0] candidate_counter;
    reg signed [31:0] best_AF; // Q s2.30

    localparam [3:0]
        S_IDLE          = 4'd0,
        S_LOAD_ACT_W0 = 4'd1, // load action word 0 (feature idx)
        S_LOAD_ACT_W1 = 4'd2, // load action word 1 (r_j_h)
        S_CAPTURE_ACT = 4'd3,
        S_MUL      = 4'd4,
        S_MUL_WAIT = 4'd5,
        S_DIV      = 4'd6,
        S_DIV_WAIT = 4'd7,
        S_UPDATE_BEST   = 4'd8,
        S_NEXT_CAND     = 4'd9,
        S_DONE          = 4'd10;
    
   // fixedMultiply
    reg  signed [15:0] mul_a, mul_b;
    reg                mul_valid;
    wire signed [31:0] mul_product;
    wire               mul_result_valid;

    // fixedDivide
    reg  signed [31:0] div_numerator;
    reg  signed [15:0] div_recip;
    reg                div_valid;
    wire signed [31:0] div_quotient;
    wire               div_result_valid;

    reg signed [31:0] candidate_AF; // Q s2.30
    reg [8:0]         latched_feature_idx;
    reg signed [15:0] latched_r_j_h;

    fixedMultiply u_mul (
        .clk          (clk),
        .reset        (reset),
        .a            (mul_a),
        .b            (mul_b),
        .valid        (mul_valid),
        .product      (mul_product),
        .result_valid (mul_result_valid)
    );

    fixedDivide u_div (
        .clk                    (clk),
        .reset                  (reset),
        .numerator              (div_numerator),
        .reciprocal_denominator (div_recip),
        .valid                  (div_valid),
        .quotient               (div_quotient),
        .result_valid           (div_result_valid)
    );

    always @(posedge clk) begin
        if (reset) begin
            state <= S_IDLE;
            candidate_counter <= 0;
            best_AF <= -32'sh40000000; // -1.0 in Q s2.30
            best_action_addr <= 14'd0;
            best_feature_idx <= 9'd0;
            done <= 1'b0;
        end
        else begin
            mul_valid <= 1'b0;
            div_valid <= 1'b0;
            done      <= 1'b0;
            case (state)

                S_IDLE: begin
                    if (start) begin
                        candidate_counter <= 8'd0;
                        best_AF <= 32'sh80000000;
                        state <= S_LOAD_ACT_W0;
                    end
                end

                S_LOAD_ACT_W0: begin
                    action_data_addr <= {action_data_start_addr + {6'd0, candidate_counter}, 1'b0}; // each candidate has 2 words of header
                    state <= S_LOAD_ACT_W1;
                end

                S_LOAD_ACT_W1: begin
                    latched_feature_idx <= action_data_out[8:0];
                    action_data_addr <= action_data_addr + 14'd1;
                    state <= S_CAPTURE_ACT;
                end

                S_CAPTURE_ACT: begin
                    latched_r_j_h      <= action_data_out[15:0];
                    state <= S_MUL;
                end

                S_MUL: begin
                    mul_a     <= p_h_f;
                    mul_b     <= latched_r_j_h;
                    mul_valid <= 1'b1;
                    state     <= S_MUL_WAIT;
                end

                S_MUL_WAIT: begin
                    if (mul_result_valid) begin
                        candidate_AF <= mul_product;
                        state <= S_DIV;
                    end
                end

                S_DIV: begin
                    div_numerator <= candidate_AF; // Q s2.30
                    div_recip     <= pgt1_recip;              // Q s3.13
                    div_valid     <= 1'b1;
                    state         <= S_DIV_WAIT;
                end

                S_DIV_WAIT: begin
                    if (div_result_valid) begin
                        candidate_AF <= div_quotient;
                        state <= S_UPDATE_BEST;
                    end
                end

                S_UPDATE_BEST: begin
                    if (candidate_AF > best_AF) begin
                        best_AF <= candidate_AF;
                        best_action_addr <= action_data_start_addr + {6'd0, candidate_counter};
                        best_feature_idx <= latched_feature_idx;
                    end
                    state <= S_NEXT_CAND;
                end

                S_NEXT_CAND: begin
                    candidate_counter <= candidate_counter + 1;
                    if (candidate_counter < n_candidates - 1) begin
                        state <= S_LOAD_ACT_W0;
                    end
                    else begin
                        state <= S_DONE;
                    end
                end

                S_DONE: begin
                    done <= 1'b1;
                end

            endcase
        end
    end

endmodule