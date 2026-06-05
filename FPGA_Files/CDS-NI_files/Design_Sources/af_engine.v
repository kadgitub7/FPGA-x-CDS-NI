`timescale 1ns / 1ps

// This module is a little special since it introduces parallelism
// we accumulare, flag, and make decision in parallel, but multiplication and division are shared


module af_engine(
    input wire clk,
    input wire reset,
    input wire start,
    input wire [7:0] node_idx,
    input wire [3:0] lane_mask,             // which lanes are active for this node

    // Per-lane AF carry-forward
    input wire signed [31:0] AF_init_0,
    input wire signed [31:0] AF_init_1,
    input wire signed [31:0] AF_init_2,
    input wire signed [31:0] AF_init_3,

    // BRAM read data (shared, from model_rom)
    input wire [15:0] action_hdr_data,
    input wire [15:0] action_data_out,
    input wire signed [15:0] prob_phf_data,
    input wire signed [15:0] prob_pgt1_data,
    input wire signed [15:0] hr_bmin,
    input wire signed [15:0] hr_bmax,

    // 4 sensor data inputs (broadcast address, per-lane data)
    input wire signed [15:0] user_feature_value_0,
    input wire signed [15:0] user_feature_value_1,
    input wire signed [15:0] user_feature_value_2,
    input wire signed [15:0] user_feature_value_3,

    // BRAM address outputs (shared, to model_rom)
    output reg [12:0] action_hdr_addr,
    output reg [13:0] action_data_addr,
    output reg [11:0] prob_phf_addr,
    output reg [7:0]  prob_pgt1_addr,
    output reg [15:0] hr_read_addr,
    output reg [8:0]  feature_read_addr,    // broadcast to all 4 sensors

    // Per-lane results
    output reg [1:0]  decision_0, decision_1, decision_2, decision_3,
    output reg signed [31:0] AF_out_0, AF_out_1, AF_out_2, AF_out_3,
    output reg [3:0]  alarm_class_0, alarm_class_1, alarm_class_2, alarm_class_3,
    output reg done
);
    localparam [1:0] DEC_HEALTHY   = 2'b00,
                     DEC_UNHEALTHY = 2'b01,
                     DEC_SCREENING = 2'b10,
                     DEC_UNKNOWN   = 2'b11;

    localparam [3:0] N_DISEASES = 4'd12;
    localparam signed [31:0] THRESHOLD_FP = 32'sh0199999A;

    // State encoding
    localparam [5:0]
        S_IDLE           = 6'd0,
        S_LOAD_DISEASE   = 6'd1,
        S_LOAD_HDR_W0    = 6'd2,
        S_WAIT_HDR_W0    = 6'd3,
        S_LOAD_HDR_W1    = 6'd4,
        S_WAIT_HDR_W1    = 6'd5,
        S_CAPTURE_HDR    = 6'd6,
        S_LOAD_ACT_W0    = 6'd7,
        S_WAIT_ACT_W0    = 6'd8,
        S_LOAD_ACT_W1    = 6'd9,
        S_WAIT_ACT_W1    = 6'd10,
        S_CAPTURE_ACT    = 6'd11,
        S_LOAD_PROBS     = 6'd12,
        S_WAIT_PROBS     = 6'd13,
        S_CAPTURE_PROBS  = 6'd14,
        S_LOAD_SENSOR    = 6'd15,
        S_WAIT_SENSOR    = 6'd16,
        S_CAPTURE_SENSOR = 6'd17,
        S_LOAD_RANGE     = 6'd18,
        S_WAIT_RANGE_L1  = 6'd19,
        S_WAIT_RANGE_L2  = 6'd20,
        S_CAPTURE_RANGE  = 6'd21,
        S_COMPUTE_MUL    = 6'd22,
        S_WAIT_MUL       = 6'd23,
        S_COMPUTE_DIV    = 6'd24,
        S_WAIT_DIV       = 6'd25,
        S_ACCUMULATE     = 6'd26,
        S_CHECK_RANGE    = 6'd27,
        S_EVAL_RANGE     = 6'd28,
        S_NEXT_ACTION    = 6'd29,
        S_NEXT_DISEASE   = 6'd30,
        S_THRESHOLD      = 6'd31,
        S_DONE           = 6'd33;

    reg [5:0] state;

    // Shared iteration state
    reg [3:0]  disease_offset;
    reg [5:0]  action_count;
    reg [5:0]  action_idx;
    reg [13:0] action_base_addr;

    // Shared BRAM data
    reg [15:0]        hdr_word0;
    reg [15:0]        hdr_start_addr;
    reg [8:0]         latched_feature_idx;
    reg signed [15:0] latched_r_j_h;
    reg signed [15:0] latched_phf;
    reg signed [15:0] latched_pgt1_recip;
    reg signed [15:0] latched_bmin;
    reg signed [15:0] latched_bmax;

    reg signed [31:0] mul_result_reg;
    reg signed [31:0] div_result_reg;

    // Per-lane sensor values and NaN flags
    reg signed [15:0] latched_sensor_0, latched_sensor_1,
                      latched_sensor_2, latched_sensor_3;
    reg [3:0] nan_mask;       // which lanes have NaN for current feature
    reg [3:0] lane_alarm;     // which lanes triggered UNHEALTHY

    // Shared address computation
    wire [11:0] node_times_12 = {1'b0, node_idx, 3'b0} + {2'b0, node_idx, 2'b0};
    wire [15:0] node_times_279 = {node_idx, 8'd0} + {4'd0, node_idx, 4'd0}
                                + {6'd0, node_idx, 2'd0} + {7'd0, node_idx, 1'd0}
                                + {8'd0, node_idx};

    // Shared multiply/divide
    reg  signed [15:0] mul_a, mul_b;
    reg                mul_valid;
    wire signed [31:0] mul_product;
    wire               mul_result_valid;

    reg  signed [31:0] div_numerator;
    reg  signed [15:0] div_recip;
    reg                div_valid;
    wire signed [31:0] div_quotient;
    wire               div_result_valid;

    fixedMultiply u_mul (
        .clk(clk), .reset(reset),
        .a(mul_a), .b(mul_b), .valid(mul_valid),
        .product(mul_product), .result_valid(mul_result_valid)
    );

    fixedDivide u_div (
        .clk(clk), .reset(reset),
        .numerator(div_numerator), .reciprocal_denominator(div_recip),
        .valid(div_valid), .quotient(div_quotient), .result_valid(div_result_valid)
    );


    reg  signed [31:0] accum_delta;         // shared delta value
    reg  [3:0]         accum_delta_valid;    // per-lane pulse
    reg  [3:0]         accum_clear;          // per-lane clear
    wire signed [31:0] accum_AF_real  [0:3]; // per-lane AF output
    wire signed [31:0] accum_rw_real  [0:3]; // per-lane rw output

    af_accumulator u_accum_0 (
        .clk(clk), .reset(reset), .delta_AF(accum_delta),
        .delta_valid(accum_delta_valid[0]), .clear(accum_clear[0]),
        .AF_real(accum_AF_real[0]), .rw_real(accum_rw_real[0])
    );
    af_accumulator u_accum_1 (
        .clk(clk), .reset(reset), .delta_AF(accum_delta),
        .delta_valid(accum_delta_valid[1]), .clear(accum_clear[1]),
        .AF_real(accum_AF_real[1]), .rw_real(accum_rw_real[1])
    );
    af_accumulator u_accum_2 (
        .clk(clk), .reset(reset), .delta_AF(accum_delta),
        .delta_valid(accum_delta_valid[2]), .clear(accum_clear[2]),
        .AF_real(accum_AF_real[2]), .rw_real(accum_rw_real[2])
    );
    af_accumulator u_accum_3 (
        .clk(clk), .reset(reset), .delta_AF(accum_delta),
        .delta_valid(accum_delta_valid[3]), .clear(accum_clear[3]),
        .AF_real(accum_AF_real[3]), .rw_real(accum_rw_real[3])
    );

    // 4 range comparators
    // All share the same bmin/bmax (from model BRAM), differ on raw_value
    reg                rc_valid;
    wire [3:0]         rc_triggered;
    wire [3:0]         rc_is_nan;

    rangeComparator u_range_0 (
        .raw_value(latched_sensor_0), .b_min(latched_bmin), .b_max(latched_bmax),
        .valid(rc_valid), .triggered(rc_triggered[0]), .is_nan(rc_is_nan[0]),
        .result_valid()
    );
    rangeComparator u_range_1 (
        .raw_value(latched_sensor_1), .b_min(latched_bmin), .b_max(latched_bmax),
        .valid(rc_valid), .triggered(rc_triggered[1]), .is_nan(rc_is_nan[1]),
        .result_valid()
    );
    rangeComparator u_range_2 (
        .raw_value(latched_sensor_2), .b_min(latched_bmin), .b_max(latched_bmax),
        .valid(rc_valid), .triggered(rc_triggered[2]), .is_nan(rc_is_nan[2]),
        .result_valid()
    );
    rangeComparator u_range_3 (
        .raw_value(latched_sensor_3), .b_min(latched_bmin), .b_max(latched_bmax),
        .valid(rc_valid), .triggered(rc_triggered[3]), .is_nan(rc_is_nan[3]),
        .result_valid()
    );

    // Active lanes that are not NaN and not already alarmed
    wire [3:0] live_lanes = lane_mask & ~lane_alarm;

    always @(posedge clk) begin
        if (reset) begin
            state             <= S_IDLE;
            done              <= 1'b0;
            decision_0 <= DEC_UNKNOWN; decision_1 <= DEC_UNKNOWN;
            decision_2 <= DEC_UNKNOWN; decision_3 <= DEC_UNKNOWN;
            AF_out_0 <= 32'sd0; AF_out_1 <= 32'sd0;
            AF_out_2 <= 32'sd0; AF_out_3 <= 32'sd0;
            alarm_class_0 <= 4'd0; alarm_class_1 <= 4'd0;
            alarm_class_2 <= 4'd0; alarm_class_3 <= 4'd0;

            disease_offset    <= 4'd0;
            action_count      <= 6'd0;
            action_idx        <= 6'd0;
            action_base_addr  <= 14'd0;
            lane_alarm        <= 4'd0;
            nan_mask          <= 4'd0;

            action_hdr_addr   <= 13'd0;
            action_data_addr  <= 14'd0;
            prob_phf_addr     <= 12'd0;
            prob_pgt1_addr    <= 8'd0;
            hr_read_addr      <= 16'd0;
            feature_read_addr <= 9'd0;

            mul_a <= 16'sd0; mul_b <= 16'sd0; mul_valid <= 1'b0;
            div_numerator <= 32'sd0; div_recip <= 16'sd0; div_valid <= 1'b0;
            accum_delta <= 32'sd0;
            accum_delta_valid <= 4'd0;
            accum_clear <= 4'b1111;
            rc_valid <= 1'b0;
        end
        else begin
            // Default de-assertions
            mul_valid         <= 1'b0;
            div_valid         <= 1'b0;
            accum_delta_valid <= 4'd0;
            accum_clear       <= 4'd0;
            rc_valid          <= 1'b0;
            done              <= 1'b0;

            case (state)

                S_IDLE: begin
                    if (start) begin
                        accum_clear <= 4'b1111;  // clear all 4 accumulators
                        lane_alarm  <= 4'd0;
                        state       <= S_LOAD_DISEASE;
                    end
                end

                // Load AF_init into lane 0's accumulator (1 lane per cycle)
                S_LOAD_DISEASE: begin
                    disease_offset <= 4'd0;
                    if (lane_mask[0]) begin
                        accum_delta <= AF_init_0;
                        accum_delta_valid[0] <= 1'b1;
                    end
                    state <= 6'd34; // S_LOAD_INIT_1
                end

                // ── Per-lane AF_init loading (3 more sub-cycles) ─────
                6'd34: begin // lane 1
                    if (lane_mask[1]) begin
                        accum_delta <= AF_init_1;
                        accum_delta_valid[1] <= 1'b1;
                    end
                    state <= 6'd35;
                end
                6'd35: begin // lane 2
                    if (lane_mask[2]) begin
                        accum_delta <= AF_init_2;
                        accum_delta_valid[2] <= 1'b1;
                    end
                    state <= 6'd36;
                end
                6'd36: begin // lane 3
                    if (lane_mask[3]) begin
                        accum_delta <= AF_init_3;
                        accum_delta_valid[3] <= 1'b1;
                    end
                    state <= S_LOAD_HDR_W0;
                end

                // ACTION HEADER READ
                S_LOAD_HDR_W0: begin
                    action_hdr_addr <= {node_times_12 + {8'd0, disease_offset}, 1'b0};
                    state <= S_WAIT_HDR_W0;
                end
                S_WAIT_HDR_W0: state <= S_LOAD_HDR_W1;
                S_LOAD_HDR_W1: begin
                    hdr_word0 <= action_hdr_data;
                    action_hdr_addr <= action_hdr_addr + 13'd1;
                    state <= S_WAIT_HDR_W1;
                end
                S_WAIT_HDR_W1: state <= S_CAPTURE_HDR;
                S_CAPTURE_HDR: begin
                    hdr_start_addr <= action_hdr_data;
                    action_count <= hdr_word0[5:0];
                    if (hdr_word0[5:0] > 0) begin
                        action_idx <= 6'd0;
                        action_base_addr <= action_hdr_data[13:0];
                        state <= S_LOAD_ACT_W0;
                    end
                    else state <= S_NEXT_DISEASE;
                end

                // ACTION DATA READ 
                S_LOAD_ACT_W0: begin
                    action_data_addr <= {action_base_addr + {8'd0, action_idx}, 1'b0};
                    state <= S_WAIT_ACT_W0;
                end
                S_WAIT_ACT_W0: state <= S_LOAD_ACT_W1;
                S_LOAD_ACT_W1: begin
                    latched_feature_idx <= action_data_out[8:0];
                    action_data_addr <= action_data_addr + 14'd1;
                    state <= S_WAIT_ACT_W1;
                end
                S_WAIT_ACT_W1: state <= S_CAPTURE_ACT;
                S_CAPTURE_ACT: begin
                    latched_r_j_h <= action_data_out;
                    state <= S_LOAD_PROBS;
                end

                // PROBABILITY READS
                S_LOAD_PROBS: begin
                    prob_phf_addr <= node_times_12 + disease_offset;
                    prob_pgt1_addr <= node_idx;
                    state <= S_WAIT_PROBS;
                end
                S_WAIT_PROBS: state <= S_CAPTURE_PROBS;
                S_CAPTURE_PROBS: begin
                    latched_phf <= prob_phf_data;
                    latched_pgt1_recip <= prob_pgt1_data;
                    state <= S_LOAD_SENSOR;
                end

                // SENSOR READ
                S_LOAD_SENSOR: begin
                    feature_read_addr <= latched_feature_idx;
                    state <= S_WAIT_SENSOR;
                end
                S_WAIT_SENSOR: state <= S_CAPTURE_SENSOR;

                // Capture all 4 sensor values, check NaN per lane
                S_CAPTURE_SENSOR: begin
                    latched_sensor_0 <= user_feature_value_0;
                    latched_sensor_1 <= user_feature_value_1;
                    latched_sensor_2 <= user_feature_value_2;
                    latched_sensor_3 <= user_feature_value_3;

                    nan_mask[0] <= (user_feature_value_0 == 16'sh7FFF);
                    nan_mask[1] <= (user_feature_value_1 == 16'sh7FFF);
                    nan_mask[2] <= (user_feature_value_2 == 16'sh7FFF);
                    nan_mask[3] <= (user_feature_value_3 == 16'sh7FFF);

                    // If ALL active non-alarmed lanes have NaN, skip
                    if ( (live_lanes & ~{
                            (user_feature_value_3 == 16'sh7FFF),
                            (user_feature_value_2 == 16'sh7FFF),
                            (user_feature_value_1 == 16'sh7FFF),
                            (user_feature_value_0 == 16'sh7FFF)
                         }) == 4'd0 ) begin
                        state <= S_NEXT_ACTION;
                    end
                    else begin
                        state <= S_LOAD_RANGE;
                    end
                end

                // HEALTHY RANGE LOOKUP
                S_LOAD_RANGE: begin
                    hr_read_addr <= node_times_279 + {7'd0, latched_feature_idx};
                    state <= S_WAIT_RANGE_L1;
                end
                S_WAIT_RANGE_L1: state <= S_WAIT_RANGE_L2;
                S_WAIT_RANGE_L2: state <= S_CAPTURE_RANGE;
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
                    accum_delta_valid <= live_lanes & ~nan_mask;
                    state <= S_CHECK_RANGE;
                end

                S_CHECK_RANGE: begin
                    // rangeComparators already have latched_sensor_i and
                    // latched_bmin/bmax wired. Just pulse valid.
                    rc_valid <= 1'b1;
                    state <= S_EVAL_RANGE;
                end

                // Evaluate 4 range results simultaneously
                S_EVAL_RANGE: begin
                    rc_valid <= 1'b1;
                    // For each active, non-NaN, non-alarmed lane: check trigger
                    if (rc_triggered[0] & live_lanes[0] & ~nan_mask[0]) begin
                        lane_alarm[0] <= 1'b1;
                        alarm_class_0 <= disease_offset;
                    end
                    if (rc_triggered[1] & live_lanes[1] & ~nan_mask[1]) begin
                        lane_alarm[1] <= 1'b1;
                        alarm_class_1 <= disease_offset;
                    end
                    if (rc_triggered[2] & live_lanes[2] & ~nan_mask[2]) begin
                        lane_alarm[2] <= 1'b1;
                        alarm_class_2 <= disease_offset;
                    end
                    if (rc_triggered[3] & live_lanes[3] & ~nan_mask[3]) begin
                        lane_alarm[3] <= 1'b1;
                        alarm_class_3 <= disease_offset;
                    end
                    state <= S_NEXT_ACTION;
                end

                S_NEXT_ACTION: begin
                    if (action_idx < action_count - 1) begin
                        action_idx <= action_idx + 1;
                        state <= S_LOAD_ACT_W0;
                    end
                    else state <= S_NEXT_DISEASE;
                end

                S_NEXT_DISEASE: begin
                    if (disease_offset < N_DISEASES - 1) begin
                        disease_offset <= disease_offset + 1;
                        state <= S_LOAD_HDR_W0;
                    end
                    else state <= S_THRESHOLD;
                end

                S_THRESHOLD: begin
                    // Lane 0
                    AF_out_0 <= accum_AF_real[0];
                    if (lane_alarm[0])           decision_0 <= DEC_UNHEALTHY;
                    else if (accum_rw_real[0] <= THRESHOLD_FP) decision_0 <= DEC_HEALTHY;
                    else                         decision_0 <= DEC_SCREENING;

                    // Lane 1
                    AF_out_1 <= accum_AF_real[1];
                    if (lane_alarm[1])           decision_1 <= DEC_UNHEALTHY;
                    else if (accum_rw_real[1] <= THRESHOLD_FP) decision_1 <= DEC_HEALTHY;
                    else                         decision_1 <= DEC_SCREENING;

                    // Lane 2
                    AF_out_2 <= accum_AF_real[2];
                    if (lane_alarm[2])           decision_2 <= DEC_UNHEALTHY;
                    else if (accum_rw_real[2] <= THRESHOLD_FP) decision_2 <= DEC_HEALTHY;
                    else                         decision_2 <= DEC_SCREENING;

                    // Lane 3
                    AF_out_3 <= accum_AF_real[3];
                    if (lane_alarm[3])           decision_3 <= DEC_UNHEALTHY;
                    else if (accum_rw_real[3] <= THRESHOLD_FP) decision_3 <= DEC_HEALTHY;
                    else                         decision_3 <= DEC_SCREENING;

                    state <= S_DONE;
                end

                S_DONE: begin
                    done <= 1'b1;
                    state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
