`timescale 1ns / 1ps

// ============================================================================
// af_engine.v - Anomaly Factor computation for one tree node
// ============================================================================
//
// BRAM TIMING FIX:
//   All BRAMs (model_rom, sensor_interface) have REGISTERED read ports.
//   After presenting an address, data needs 1 extra clock cycle to appear:
//     Cycle N:   FSM sets address (NBA)
//     Cycle N+1: BRAM clocks in address, data appears via NBA (end of N+1)
//     Cycle N+2: Data is stable - FSM can capture it
//
//   A WAIT state is inserted after every address setup.
//
//   For the healthy range two-level lookup (hr_index → hr_pairs):
//     Cycle N:   set compact address → hr_index BRAM
//     Cycle N+1: hr_index BRAM processes → pair_id appears (end of N+1)
//     Cycle N+2: pair_id feeds hr_pairs BRAM → {bmin,bmax} appears (end of N+2)
//     Cycle N+3: FSM captures bmin/bmax
//   This needs 3 WAIT states (S_WAIT_RANGE_L1, S_WAIT_RANGE_L2, S_CAPTURE_RANGE).
// ============================================================================

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
    output reg [15:0] hr_read_addr,    // compact: node_idx*279 + feat_idx
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

    // ── State encoding (6 bits, 34 states) ────────────────────────────
    localparam [5:0]
        S_IDLE           = 6'd0,
        S_LOAD_DISEASE   = 6'd1,
        S_LOAD_HDR_W0    = 6'd2,   // issue addr for header word 0
        S_WAIT_HDR_W0    = 6'd3,   // *** WAIT for BRAM ***
        S_LOAD_HDR_W1    = 6'd4,   // capture word 0, issue addr for word 1
        S_WAIT_HDR_W1    = 6'd5,   // *** WAIT for BRAM ***
        S_CAPTURE_HDR    = 6'd6,   // capture word 1 (start_addr)
        S_LOAD_ACT_W0    = 6'd7,   // issue addr for action feature_idx
        S_WAIT_ACT_W0    = 6'd8,   // *** WAIT for BRAM ***
        S_LOAD_ACT_W1    = 6'd9,   // capture feature_idx, issue addr for r_j_h
        S_WAIT_ACT_W1    = 6'd10,  // *** WAIT for BRAM ***
        S_CAPTURE_ACT    = 6'd11,  // capture r_j_h
        S_LOAD_PROBS     = 6'd12,  // issue addr to prob_phf and prob_pgt1
        S_WAIT_PROBS     = 6'd13,  // *** WAIT for BRAM ***
        S_CAPTURE_PROBS  = 6'd14,  // capture P(h,f) and recip(P(h>1,f))
        S_LOAD_SENSOR    = 6'd15,  // issue feature_read_addr
        S_WAIT_SENSOR    = 6'd16,  // *** WAIT for sensor BRAM ***
        S_CAPTURE_SENSOR = 6'd17,  // capture user_feature_value
        S_LOAD_RANGE     = 6'd18,  // issue compact addr to hr_index BRAM (level 1)
        S_WAIT_RANGE_L1  = 6'd19,  // wait for hr_index → pair_id (level 1)
        S_WAIT_RANGE_L2  = 6'd20,  // wait for hr_pairs → {bmin,bmax} (level 2)
        S_CAPTURE_RANGE  = 6'd21,  // capture hr_bmin / hr_bmax
        S_COMPUTE_MUL    = 6'd22,  // drive fixedMultiply inputs
        S_WAIT_MUL       = 6'd23,  // wait for multiply result
        S_COMPUTE_DIV    = 6'd24,  // drive fixedDivide inputs
        S_WAIT_DIV       = 6'd25,  // wait for divide result
        S_ACCUMULATE     = 6'd26,  // pulse af_accumulator
        S_CHECK_RANGE    = 6'd27,  // drive rangeComparator inputs
        S_EVAL_RANGE     = 6'd28,  // read rangeComparator outputs
        S_NEXT_ACTION    = 6'd29,  // increment action pointer
        S_NEXT_DISEASE   = 6'd30,  // increment disease_offset
        S_THRESHOLD      = 6'd31,  // compare rw_real vs THRESHOLD_FP
        S_ALARM          = 6'd32,  // latch UNHEALTHY result
        S_DONE           = 6'd33;  // assert done

    reg [5:0] state;

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

    // node_idx * 12 = (node_idx << 3) + (node_idx << 2)
    wire [11:0] node_times_12 = {1'b0, node_idx, 3'b0} + {2'b0, node_idx, 2'b0};

    // Compact healthy-range address: node_idx * 279 + feature_idx
    // 279 = 256 + 16 + 4 + 2 + 1
    wire [15:0] node_times_279 = {node_idx, 8'd0}        // node_idx * 256
                                + {4'd0, node_idx, 4'd0}  // node_idx * 16
                                + {6'd0, node_idx, 2'd0}  // node_idx * 4
                                + {7'd0, node_idx, 1'd0}  // node_idx * 2
                                + {8'd0, node_idx};        // node_idx * 1

    // ── Sub-module interfaces ─────────────────────────────────────────

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
            hr_read_addr      <= 16'd0;
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

                // ── Idle: wait for start pulse ──────────────────────
                S_IDLE: begin
                    if (start) begin
                        accum_clear <= 1'b1;
                        state <= S_LOAD_DISEASE;
                    end
                end

                // ── Initialize: load AF_init into accumulator ───────
                S_LOAD_DISEASE: begin
                    accum_delta <= AF_init;
                    accum_delta_valid <= 1'b1;
                    disease_offset <= 4'd0;
                    state <= S_LOAD_HDR_W0;
                end

                // ═══════════════════════════════════════════════════
                // ACTION HEADER READ (2 words per slot)
                //   Word 0: action_count
                //   Word 1: start_address in action_data BRAM
                // ═══════════════════════════════════════════════════

                // Issue address for header word 0
                S_LOAD_HDR_W0: begin
                    action_hdr_addr <= {node_times_12 + {8'd0, disease_offset}, 1'b0};
                    state <= S_WAIT_HDR_W0;
                end

                // *** WAIT for action_hdr BRAM ***
                S_WAIT_HDR_W0: begin
                    state <= S_LOAD_HDR_W1;
                end

                // Capture word 0 (action_count), issue address for word 1
                S_LOAD_HDR_W1: begin
                    hdr_word0 <= action_hdr_data;
                    action_hdr_addr <= action_hdr_addr + 13'd1;
                    state <= S_WAIT_HDR_W1;
                end

                // *** WAIT for action_hdr BRAM ***
                S_WAIT_HDR_W1: begin
                    state <= S_CAPTURE_HDR;
                end

                // Capture word 1 (start_address), decide what to do
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

                // ═══════════════════════════════════════════════════
                // ACTION DATA READ (2 words per action)
                //   Word 0: feature_idx (low 9 bits)
                //   Word 1: r_j_h (Q s1.15)
                // ═══════════════════════════════════════════════════

                // Issue address for action word 0
                S_LOAD_ACT_W0: begin
                    action_data_addr <= {action_base_addr + {8'd0, action_idx}, 1'b0};
                    state <= S_WAIT_ACT_W0;
                end

                // *** WAIT for action_data BRAM ***
                S_WAIT_ACT_W0: begin
                    state <= S_LOAD_ACT_W1;
                end

                // Capture word 0 (feature_idx), issue address for word 1
                S_LOAD_ACT_W1: begin
                    latched_feature_idx <= action_data_out[8:0];
                    action_data_addr <= action_data_addr + 14'd1;
                    state <= S_WAIT_ACT_W1;
                end

                // *** WAIT for action_data BRAM ***
                S_WAIT_ACT_W1: begin
                    state <= S_CAPTURE_ACT;
                end

                // Capture word 1 (r_j_h)
                S_CAPTURE_ACT: begin
                    latched_r_j_h <= action_data_out;
                    state <= S_LOAD_PROBS;
                end

                // ═══════════════════════════════════════════════════
                // PROBABILITY READS (both BRAMs simultaneously)
                //   prob_phf:  P(h,f) at addr = node_idx*12 + disease_offset
                //   prob_pgt1: 1/P(h>1,f) at addr = node_idx
                // ═══════════════════════════════════════════════════

                S_LOAD_PROBS: begin
                    prob_phf_addr <= node_times_12 + disease_offset;
                    prob_pgt1_addr <= node_idx;
                    state <= S_WAIT_PROBS;
                end

                // *** WAIT for both prob BRAMs ***
                S_WAIT_PROBS: begin
                    state <= S_CAPTURE_PROBS;
                end

                S_CAPTURE_PROBS: begin
                    latched_phf <= prob_phf_data;
                    latched_pgt1_recip <= prob_pgt1_data;
                    state <= S_LOAD_SENSOR;
                end

                // ═══════════════════════════════════════════════════
                // SENSOR READ (user feature value)
                // ═══════════════════════════════════════════════════

                S_LOAD_SENSOR: begin
                    feature_read_addr <= latched_feature_idx;
                    state <= S_WAIT_SENSOR;
                end

                // *** WAIT for sensor BRAM ***
                S_WAIT_SENSOR: begin
                    state <= S_CAPTURE_SENSOR;
                end

                S_CAPTURE_SENSOR: begin
                    latched_sensor_val <= user_feature_value;
                    // NaN sentinel check: if the feature value is 0x7FFF,
                    // skip this action entirely (no AF, no range check).
                    // This matches the Python golden model's "continue" on NaN.
                    if (user_feature_value == 16'sh7FFF) begin
                        state <= S_NEXT_ACTION;
                    end
                    else begin
                        state <= S_LOAD_RANGE;
                    end
                end

                // ═══════════════════════════════════════════════════
                // HEALTHY RANGE - Two-level BRAM lookup
                //   Level 1: hr_index BRAM  (compact addr → pair_id)
                //   Level 2: hr_pairs BRAM  (pair_id → {bmin, bmax})
                //   Total latency: 2 BRAM reads = 2 wait cycles
                // ═══════════════════════════════════════════════════

                S_LOAD_RANGE: begin
                    hr_read_addr <= node_times_279 + {7'd0, latched_feature_idx};
                    state <= S_WAIT_RANGE_L1;
                end

                // *** WAIT for level 1 (hr_index → pair_id) ***
                S_WAIT_RANGE_L1: begin
                    state <= S_WAIT_RANGE_L2;
                end

                // *** WAIT for level 2 (hr_pairs → {bmin, bmax}) ***
                // pair_id from level 1 feeds into level 2 address
                S_WAIT_RANGE_L2: begin
                    state <= S_CAPTURE_RANGE;
                end

                // Both levels done - capture healthy range bounds
                S_CAPTURE_RANGE: begin
                    latched_bmin <= hr_bmin;
                    latched_bmax <= hr_bmax;
                    state <= S_COMPUTE_MUL;
                end

                // ═══════════════════════════════════════════════════
                // COMPUTE: multiply, divide, accumulate, range-check
                // ═══════════════════════════════════════════════════

                // Multiply: P(h,f) * r_j_h → Q s2.30
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

                // Divide: (P(h,f) * r_j_h) * (1/P(h>1,f)) → Q s2.30
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

                // Accumulate: AF_real += delta_AF
                S_ACCUMULATE: begin
                    accum_delta <= div_result_reg;
                    accum_delta_valid <= 1'b1;
                    state <= S_CHECK_RANGE;
                end

                // Check healthy range: is sensor value outside [bmin, bmax]?
                S_CHECK_RANGE: begin
                    rc_raw <= latched_sensor_val;
                    rc_bmin <= latched_bmin;
                    rc_bmax <= latched_bmax;
                    rc_valid <= 1'b1;
                    state <= S_EVAL_RANGE;
                end

                // Evaluate range check result
                S_EVAL_RANGE: begin
                    rc_valid <= 1'b1;   // hold valid for combinational output
                    if (rc_triggered) begin
                        state <= S_ALARM;
                    end
                    else begin
                        state <= S_NEXT_ACTION;
                    end
                end

                // ═══════════════════════════════════════════════════
                // ITERATION: next action / next disease / done
                // ═══════════════════════════════════════════════════

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

                // ═══════════════════════════════════════════════════
                // FINAL DECISION
                // ═══════════════════════════════════════════════════

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
                    state <= S_IDLE;
                end

                default: state <= S_IDLE;

            endcase
        end
    end

endmodule
