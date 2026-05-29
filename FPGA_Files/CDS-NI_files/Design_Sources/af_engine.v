`timescale 1ns / 1ps

module af_engine(
    input wire clk,
    input wire reset,
    input wire start,
    input wire [7:0] node_idx,
    input wire signed [31:0] AF_init,

    // BRAM read data (from model_rom / sensor_interface)
    input wire [15:0] action_hdr_data,
    input wire [15:0] action_data_out,
    input wire signed [15:0] prob_phf_data,
    input wire signed [15:0] prob_pgt1_data,
    input wire signed [15:0] hr_bmin,
    input wire signed [15:0] hr_bmax,
    input wire signed [15:0] user_feature_value,

    // BRAM address outputs (to model_rom / sensor_interface)
    output reg [12:0] action_hdr_addr,
    output reg [13:0] action_data_addr,
    output reg [11:0] prob_phf_addr,
    output reg [7:0]  prob_pgt1_addr,
    output reg [16:0] hr_read_addr,
    output reg [8:0]  feature_read_addr,

    output reg [1:0]  decision,
    output reg signed [31:0] AF_out,
    output reg [3:0]  alarm_class,
    output reg done
);
    localparam [1:0] DEC_HEALTHY   = 2'b00,
                     DEC_UNHEALTHY = 2'b01,
                     DEC_SCREENING = 2'b10,
                     DEC_UNKNOWN   = 2'b11;

    localparam [3:0] N_DISEASES = 4'd12;

    localparam signed [31:0] THRESHOLD_FP = 32'sh0199999A; // 0.025 in Q s2.30

    // these are all the states that will be tracked based on the FSM diagram
    // BRAM takes 1 cycle to return data, therefore there is a wait state that lets them rest before being used

    localparam [4:0]
        S_IDLE           = 5'd0,
        S_LOAD_DISEASE   = 5'd1,
        S_LOAD_HDR_W0    = 5'd2,   // issue addr for header word 0
        S_LOAD_HDR_W1    = 5'd3,   // issue addr for header word 1, capture word 0
        S_CAPTURE_HDR    = 5'd4,   // capture start_addr from word 1
        S_LOAD_ACT_W0    = 5'd5,   // issue addr for action feature_idx
        S_LOAD_ACT_W1    = 5'd6,   // issue addr for r_j_h, capture feature_idx
        S_CAPTURE_ACT    = 5'd7,   // capture r_j_h
        S_LOAD_PROBS     = 5'd8,   // issue addr to prob_phf and prob_pgt1
        S_CAPTURE_PROBS  = 5'd9,   // capture P(h,f) and recip(P(h>1,f))
        S_LOAD_SENSOR    = 5'd10,  // issue feature_read_addr
        S_CAPTURE_SENSOR = 5'd11,  // capture user_feature_value
        S_LOAD_RANGE     = 5'd12,  // issue hr_read_addr
        S_CAPTURE_RANGE  = 5'd13,  // capture hr_bmin / hr_bmax
        S_COMPUTE_MUL    = 5'd14,  // drive fixedMultiply inputs
        S_WAIT_MUL       = 5'd15,  // wait 1 cycle for multiply result
        S_COMPUTE_DIV    = 5'd16,  // drive fixedDivide inputs
        S_WAIT_DIV       = 5'd17,  // wait 2 cycles (pipelined divider)
        S_ACCUMULATE     = 5'd18,  // pulse af_accumulator
        S_CHECK_RANGE    = 5'd19,  // drive rangeComparator inputs
        S_EVAL_RANGE     = 5'd20,  // read rangeComparator outputs
        S_NEXT_ACTION    = 5'd21,  // increment action pointer
        S_NEXT_DISEASE   = 5'd22,  // increment disease_offset
        S_THRESHOLD      = 5'd23,  // compare rw_real vs THRESHOLD_FP
        S_ALARM          = 5'd24,  // latch UNHEALTHY result
        S_DONE           = 5'd25;  // assert done

    reg [4:0] state;

    reg [3:0]  disease_offset;       // 0 to 11
    reg [5:0]  action_count;         // actions in current (node, disease) group
    reg [5:0]  action_idx;           // current action within group
    reg [13:0] action_base_addr;     // start_address of current group in data BRAM

    // BRAM data
    reg [15:0]        hdr_word0;           // action_count (full 16-bit)
    reg [15:0]        hdr_start_addr;      // start address in action data BRAM
    reg [8:0]         latched_feature_idx;  // from action data word 0 (9 bits used)
    reg signed [15:0] latched_r_j_h;       // from action data word 1 (Q s1.15)
    reg signed [15:0] latched_phf;         // P(h,f) (Q s1.15)
    reg signed [15:0] latched_pgt1_recip;  // reciprocal of P(h>1,f) (Q s3.13)
    reg signed [15:0] latched_sensor_val;  // user feature value
    reg signed [15:0] latched_bmin;        // healthy-range lower bound
    reg signed [15:0] latched_bmax;        // healthy-range upper bound


    reg signed [31:0] mul_result_reg;      // captured multiply product (Q s2.30)
    reg signed [31:0] div_result_reg;      // captured divide quotient  (Q s2.30)
    reg               div_wait_cnt;        // divider 2-stage pipeline counter


    wire [11:0] node_times_12 = {1'b0, node_idx, 3'b0} + {2'b0, node_idx, 2'b0};

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

    // af_accumulator
    reg  signed [31:0] accum_delta;
    reg                accum_delta_valid;
    reg                accum_clear;
    wire signed [31:0] accum_AF_real;
    wire signed [31:0] accum_rw_real;

    // rangeComparator
    reg  signed [15:0] rc_raw, rc_bmin, rc_bmax;
    reg                rc_valid;
    wire               rc_triggered;
    wire               rc_is_nan;
    wire               rc_result_valid;

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

    af_accumulator u_accum (
        .clk         (clk),
        .reset       (reset),
        .delta_AF    (accum_delta),
        .delta_valid (accum_delta_valid),
        .clear       (accum_clear),
        .AF_real     (accum_AF_real),
        .rw_real     (accum_rw_real)
    );

    rangeComparator u_range (
        .raw_value    (rc_raw),
        .b_min        (rc_bmin),
        .b_max        (rc_bmax),
        .valid        (rc_valid),
        .triggered    (rc_triggered),
        .is_nan       (rc_is_nan),
        .result_valid (rc_result_valid)
    );

    
    always @(posedge clk) begin
        if (reset) begin
            state             <= S_IDLE;
            done              <= 1'b0;
            decision          <= DEC_UNKNOWN;
            AF_out            <= 32'sd0;
            alarm_class       <= 4'd0;

            disease_offset    <= 4'd0;
            action_count      <= 6'd0;
            action_idx        <= 6'd0;
            action_base_addr  <= 14'd0;

            action_hdr_addr   <= 13'd0;
            action_data_addr  <= 14'd0;
            prob_phf_addr     <= 12'd0;
            prob_pgt1_addr    <= 8'd0;
            hr_read_addr      <= 17'd0;
            feature_read_addr <= 9'd0;

            mul_a             <= 16'sd0;
            mul_b             <= 16'sd0;
            mul_valid         <= 1'b0;
            div_numerator     <= 32'sd0;
            div_recip         <= 16'sd0;
            div_valid         <= 1'b0;
            accum_delta       <= 32'sd0;
            accum_delta_valid <= 1'b0;
            accum_clear       <= 1'b1;
            rc_raw            <= 16'sd0;
            rc_bmin           <= 16'sd0;
            rc_bmax           <= 16'sd0;
            rc_valid          <= 1'b0;

            div_wait_cnt      <= 1'b0;
        end
        else begin
            // Default: de-assert one-shot pulses each cycle
            mul_valid         <= 1'b0;
            div_valid         <= 1'b0;
            accum_delta_valid <= 1'b0;
            accum_clear       <= 1'b0;
            rc_valid          <= 1'b0;
            done              <= 1'b0;

            case (state)

                S_IDLE: begin
                    if (start) begin
                        accum_clear <= 1'b1;
                        state <= S_LOAD_DISEASE;
                    end
                end

                S_LOAD_DISEASE: begin
                    accum_delta <= AF_init;
                    accum_delta_valid <= 1'b1;
                    disease_offset <= 4'd0;
                    state <= S_LOAD_HDR_W0;
                end

                S_LOAD_HDR_W0: begin
                    action_hdr_addr <= {node_times_12 + {8'd0, disease_offset}, 1'b0};
                    state <= S_LOAD_HDR_W1;
                end

                S_LOAD_HDR_W1: begin
                    hdr_word0 <= action_hdr_data;
                    action_hdr_addr <= action_hdr_addr + 13'd1;
                    state <= S_CAPTURE_HDR;
                end

                S_CAPTURE_HDR: begin
                    hdr_start_addr <= action_hdr_data;
                    action_count <= hdr_word0[5:0];
                    if (hdr_word0[5:0] > 0) begin
                        action_idx <= 6'd0;
                        action_base_addr <= action_hdr_data[13:0];
                        state <= S_LOAD_ACT_W0;
                    end
                    else begin
                        state <= S_NEXT_DISEASE;
                    end
                end

                S_LOAD_ACT_W0: begin
                    action_data_addr <= {action_base_addr + {8'd0, action_idx}, 1'b0};
                    state <= S_LOAD_ACT_W1;
                end

                S_LOAD_ACT_W1: begin
                    latched_feature_idx <= action_data_out[8:0];
                    action_data_addr <= action_data_addr + 14'd1;
                    state <= S_CAPTURE_ACT;
                end

                S_CAPTURE_ACT: begin
                    latched_r_j_h <= action_data_out;
                    state <= S_LOAD_PROBS;
                end

                S_LOAD_PROBS: begin
                    prob_phf_addr <= node_times_12 + disease_offset;
                    prob_pgt1_addr <= node_idx;
                    state <= S_CAPTURE_PROBS;
                end

                S_CAPTURE_PROBS: begin
                    latched_phf <= prob_phf_data;
                    latched_pgt1_recip <= prob_pgt1_data;
                    state <= S_LOAD_SENSOR;
                end

                S_LOAD_SENSOR: begin
                    feature_read_addr <= latched_feature_idx;
                    state <= S_CAPTURE_SENSOR;
                end

                S_CAPTURE_SENSOR: begin
                    latched_sensor_val <= user_feature_value;
                    state <= S_LOAD_RANGE;
                end

                S_LOAD_RANGE: begin
                    hr_read_addr <= {node_idx, latched_feature_idx};
                    state <= S_CAPTURE_RANGE;
                end

                S_CAPTURE_RANGE: begin
                    latched_bmin <= hr_bmin;
                    latched_bmax <= hr_bmax;
                    state <= S_COMPUTE_MUL;
                end

                S_COMPUTE_MUL: begin
                    mul_a <= latched_phf;
                    mul_b <= latched_r_j_h;
                    mul_valid <= 1'b1;
                    state <= S_WAIT_MUL;
                end

                S_WAIT_MUL: begin
                    if (mul_result_valid) begin
                        mul_result_reg <= mul_product;
                        state <= S_COMPUTE_DIV;
                    end
                end

                S_COMPUTE_DIV: begin
                    div_numerator <= mul_result_reg;
                    div_recip <= latched_pgt1_recip;
                    div_valid <= 1'b1;
                    div_wait_cnt <= 1'b0;
                    state <= S_WAIT_DIV;
                end

                S_WAIT_DIV: begin
                    if (div_result_valid) begin
                        div_result_reg <= div_quotient;
                        state <= S_ACCUMULATE;
                    end
                end

                S_ACCUMULATE: begin
                    accum_delta <= div_result_reg;
                    accum_delta_valid <= 1'b1;
                    state <= S_CHECK_RANGE;
                end

                S_CHECK_RANGE: begin
                    rc_raw <= latched_sensor_val;
                    rc_bmin <= latched_bmin;
                    rc_bmax <= latched_bmax;
                    rc_valid <= 1'b1;
                    state <= S_EVAL_RANGE;
                end

                S_EVAL_RANGE: begin
                    rc_valid <= 1'b1;
                    if (rc_triggered) begin
                        state <= S_ALARM;
                    end
                    else begin
                        state <= S_NEXT_ACTION;
                    end
                end

                S_NEXT_ACTION: begin
                    if (action_idx < action_count - 1) begin
                        action_idx <= action_idx + 1;
                        state <= S_LOAD_ACT_W0;
                    end
                    else begin
                        state <= S_NEXT_DISEASE;
                    end
                end

                S_NEXT_DISEASE: begin
                    if (disease_offset < N_DISEASES - 1) begin
                        disease_offset <= disease_offset + 1;
                        state <= S_LOAD_HDR_W0;
                    end
                    else begin
                        state <= S_THRESHOLD;
                    end
                end

                S_THRESHOLD: begin
                    AF_out <= accum_AF_real;
                    if (accum_rw_real <= THRESHOLD_FP) begin
                        decision <= DEC_HEALTHY;
                    end
                    else begin
                        decision <= DEC_SCREENING;
                    end
                    state <= S_DONE;
                end

                S_ALARM: begin
                    decision <= DEC_UNHEALTHY;
                    alarm_class <= disease_offset;
                    AF_out <= accum_AF_real;
                    state <= S_DONE;
                end

                S_DONE: begin
                    done <= 1'b1;
                end

                default: state <= S_IDLE;

            endcase
        end
    end

endmodule
