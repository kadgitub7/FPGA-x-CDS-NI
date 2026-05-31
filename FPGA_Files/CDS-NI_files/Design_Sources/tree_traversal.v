`timescale 1ns / 1ps

// ============================================================================
// tree_traversal.v — Scans all tree nodes, reports matches
// ============================================================================
//
// BRAM TIMING FIX:
//   sensor_interface and model_rom both have REGISTERED read ports.
//   After presenting an address, data is available TWO posedge cycles later:
//     Cycle N:   FSM sets address (NBA, takes effect at end of cycle N)
//     Cycle N+1: BRAM clocks in address, computes data (NBA, end of N+1)
//     Cycle N+2: Data is stable — FSM can read it
//
//   We insert a WAIT state between each address-set and data-capture
//   so the BRAM has time to produce valid data.
//
// STATE FLOW (per node):
//   S_READ_W0  → S_WAIT_W0  → S_LATCH_W0  (capture word 0, set addr for word 1)
//              → S_WAIT_W1  → S_LATCH_W1  (capture word 1, set addr for word 2)
//              → S_WAIT_W2  → S_LATCH_W2  (capture word 2)
//              → S_READ_FEAT → S_WAIT_FEAT → S_CAPTURE_FEAT
//              → S_COMPARE  → S_NEXT
// ============================================================================

module tree_traversal (
    input wire        clk,
    input wire        reset,
    input wire        start,

    input wire signed [15:0]  user_feature_value,   // from sensor_interface
    input wire [15:0] tree_data,                     // from model_rom

    output reg [8:0]  feature_read_addr,    // to sensor_interface
    output reg [9:0]  tree_read_addr,       // to model_rom
    output reg [7:0]  active_node_idx,
    output reg        active_node_valid,
    output reg        node_done,
    output reg        all_done
);

    // ── State encoding ──────────────────────────────────────────────
    localparam [3:0]
        S_IDLE         = 4'd0,
        S_ROOT_EMIT    = 4'd1,     // emit root match unconditionally
        S_READ_W0      = 4'd2,     // put word-0 address on ROM bus
        S_WAIT_W0      = 4'd3,     // *** WAIT for BRAM latency ***
        S_LATCH_W0     = 4'd4,     // latch word 0, advance address to word 1
        S_WAIT_W1      = 4'd5,     // *** WAIT for BRAM latency ***
        S_LATCH_W1     = 4'd6,     // latch word 1 (low), advance to word 2
        S_WAIT_W2      = 4'd7,     // *** WAIT for BRAM latency ***
        S_LATCH_W2     = 4'd8,     // latch word 2 (high)
        S_READ_FEAT    = 4'd9,     // put feature address on sensor bus
        S_WAIT_FEAT    = 4'd10,    // *** WAIT for sensor BRAM latency ***
        S_CAPTURE_FEAT = 4'd11,    // capture user feature value
        S_COMPARE      = 4'd12,    // compare feature against [low, high)
        S_NEXT         = 4'd13,    // advance node_counter or finish
        S_DONE         = 4'd14;

    // ── Internal registers ──────────────────────────────────────────
    reg [3:0]  state;
    reg [7:0]  node_counter;               // linear iterator: 1 -> 214
    reg [8:0]  branch_feat_idx;            // which feature this node branches on
    reg signed [15:0] node_low;            // branch low bound  (ROM word 1)
    reg signed [15:0] node_high;           // branch high bound (ROM word 2)
    reg signed [15:0] feat_latched;        // user's feature value, held stable

    // node_base = node_counter * 3  (3 words per node in tree topology BRAM)
    wire [9:0] node_base = {1'b0, node_counter, 1'b0}
                         + {2'b00, node_counter};

    always @(posedge clk) begin
        active_node_valid <= 1'b0;
        node_done         <= 1'b0;
        all_done          <= 1'b0;

        if (reset) begin
            state             <= S_IDLE;
            node_counter      <= 8'd0;
            active_node_idx   <= 8'd0;
            feature_read_addr <= 9'd0;
            tree_read_addr    <= 10'd0;
        end
        else begin
            case (state)
                S_IDLE: begin
                    if (start) begin
                        node_counter <= 8'd1;       // first non-root node to scan
                        state        <= S_ROOT_EMIT;
                    end
                end

                // Root (node 0) always matches — emit it unconditionally.
                // Also set up tree_read_addr for node 1's word 0.
                S_ROOT_EMIT: begin
                    active_node_idx   <= 8'd0;      // root = node index 0
                    active_node_valid <= 1'b1;       // one-cycle pulse
                    tree_read_addr    <= node_base;   // addr for node 1, word 0
                    state             <= S_WAIT_W0;   // wait for BRAM
                end

                // For non-root nodes: set up address for word 0
                S_READ_W0: begin
                    tree_read_addr <= node_base;
                    state          <= S_WAIT_W0;
                end

                // *** WAIT: BRAM is processing the address ***
                S_WAIT_W0: begin
                    state <= S_LATCH_W0;
                end

                // Word 0 is now valid on tree_data.
                // Capture branch_feat_idx, advance address to word 1.
                S_LATCH_W0: begin
                    branch_feat_idx <= tree_data[8:0];
                    tree_read_addr  <= tree_read_addr + 10'd1;
                    state           <= S_WAIT_W1;
                end

                // *** WAIT: BRAM is processing word 1 address ***
                S_WAIT_W1: begin
                    state <= S_LATCH_W1;
                end

                // Word 1 is now valid. Capture node_low, advance to word 2.
                S_LATCH_W1: begin
                    node_low       <= tree_data;
                    tree_read_addr <= tree_read_addr + 10'd1;
                    state          <= S_WAIT_W2;
                end

                // *** WAIT: BRAM is processing word 2 address ***
                S_WAIT_W2: begin
                    state <= S_LATCH_W2;
                end

                // Word 2 is now valid. Capture node_high.
                S_LATCH_W2: begin
                    node_high <= tree_data;
                    state     <= S_READ_FEAT;
                end

                // Set sensor read address to the branch feature index
                S_READ_FEAT: begin
                    feature_read_addr <= branch_feat_idx;
                    state             <= S_WAIT_FEAT;
                end

                // *** WAIT: sensor BRAM is processing the address ***
                S_WAIT_FEAT: begin
                    state <= S_CAPTURE_FEAT;
                end

                // Sensor data is now valid. Capture the user's feature value.
                S_CAPTURE_FEAT: begin
                    feat_latched <= user_feature_value;
                    state        <= S_COMPARE;
                end

                // Compare: does the user's feature value fall in [low, high)?
                // All values are signed Q s11.4.
                S_COMPARE: begin
                    if (feat_latched >= node_low && feat_latched < node_high) begin
                        active_node_idx   <= node_counter;
                        active_node_valid <= 1'b1;
                    end
                    state <= S_NEXT;
                end

                // Advance to next node, or finish if all 214 non-root nodes checked.
                S_NEXT: begin
                    if (node_counter >= 8'd214) begin
                        node_done <= 1'b1;
                        state     <= S_DONE;
                    end
                    else begin
                        node_counter <= node_counter + 8'd1;
                        state        <= S_READ_W0;
                    end
                end

                S_DONE: begin
                    all_done <= 1'b1;
                    state    <= S_IDLE;
                end

                default: state <= S_IDLE;

            endcase
        end
    end

endmodule
