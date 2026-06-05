`timescale 1ns / 1ps

module tree_traversal (
    input wire        clk,
    input wire        reset,
    input wire        start,

    // 4 sensor data inputs (broadcast address, per-lane data)
    input wire signed [15:0] user_feature_value_0,
    input wire signed [15:0] user_feature_value_1,
    input wire signed [15:0] user_feature_value_2,
    input wire signed [15:0] user_feature_value_3,

    input wire [15:0] tree_data,                     // from model_rom

    output reg [8:0]  feature_read_addr,    // to all 4 sensor_interfaces (broadcast)
    output reg [9:0]  tree_read_addr,       // to model_rom
    output reg [7:0]  active_node_idx,      // which node (shared, same for all lanes)
    output reg        active_node_valid_0,  // lane 0 matched this node
    output reg        active_node_valid_1,  // lane 1 matched this node
    output reg        active_node_valid_2,  // lane 2 matched this node
    output reg        active_node_valid_3,  // lane 3 matched this node
    output reg        node_done,
    output reg        all_done
);

    localparam [3:0]
        S_IDLE         = 4'd0,
        S_ROOT_EMIT    = 4'd1,
        S_READ_W0      = 4'd2,
        S_WAIT_W0      = 4'd3,
        S_LATCH_W0     = 4'd4,
        S_WAIT_W1      = 4'd5,
        S_LATCH_W1     = 4'd6,
        S_WAIT_W2      = 4'd7,
        S_LATCH_W2     = 4'd8,
        S_READ_FEAT    = 4'd9,
        S_WAIT_FEAT    = 4'd10,
        S_CAPTURE_FEAT = 4'd11,
        S_COMPARE      = 4'd12,
        S_NEXT         = 4'd13,
        S_DONE         = 4'd14;

    reg [3:0]  state;
    reg [7:0]  node_counter;
    reg [8:0]  branch_feat_idx;
    reg signed [15:0] node_low;
    reg signed [15:0] node_high;

    // Per-lane captured sensor values
    reg signed [15:0] feat_latched_0;
    reg signed [15:0] feat_latched_1;
    reg signed [15:0] feat_latched_2;
    reg signed [15:0] feat_latched_3;

    wire [9:0] node_base = {1'b0, node_counter, 1'b0}
                         + {2'b00, node_counter};

    always @(posedge clk) begin
        // Default: de-assert all one-shot pulses
        active_node_valid_0 <= 1'b0;
        active_node_valid_1 <= 1'b0;
        active_node_valid_2 <= 1'b0;
        active_node_valid_3 <= 1'b0;
        node_done           <= 1'b0;
        all_done            <= 1'b0;

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
                        node_counter <= 8'd1;
                        state        <= S_ROOT_EMIT;
                    end
                end

                // Root (node 0) always matches ALL lanes
                S_ROOT_EMIT: begin
                    active_node_idx     <= 8'd0;
                    active_node_valid_0 <= 1'b1;
                    active_node_valid_1 <= 1'b1;
                    active_node_valid_2 <= 1'b1;
                    active_node_valid_3 <= 1'b1;
                    tree_read_addr      <= node_base;
                    state               <= S_WAIT_W0;
                end

                S_READ_W0: begin
                    tree_read_addr <= node_base;
                    state          <= S_WAIT_W0;
                end

                S_WAIT_W0: state <= S_LATCH_W0;

                S_LATCH_W0: begin
                    branch_feat_idx <= tree_data[8:0];
                    tree_read_addr  <= tree_read_addr + 10'd1;
                    state           <= S_WAIT_W1;
                end

                S_WAIT_W1: state <= S_LATCH_W1;

                S_LATCH_W1: begin
                    node_low       <= tree_data;
                    tree_read_addr <= tree_read_addr + 10'd1;
                    state          <= S_WAIT_W2;
                end

                S_WAIT_W2: state <= S_LATCH_W2;

                S_LATCH_W2: begin
                    node_high <= tree_data;
                    state     <= S_READ_FEAT;
                end

                // Broadcast sensor address to all 4 lanes
                S_READ_FEAT: begin
                    feature_read_addr <= branch_feat_idx;
                    state             <= S_WAIT_FEAT;
                end

                S_WAIT_FEAT: state <= S_CAPTURE_FEAT;

                // Capture all 4 sensor values simultaneously
                S_CAPTURE_FEAT: begin
                    feat_latched_0 <= user_feature_value_0;
                    feat_latched_1 <= user_feature_value_1;
                    feat_latched_2 <= user_feature_value_2;
                    feat_latched_3 <= user_feature_value_3;
                    state          <= S_COMPARE;
                end

                // 4 parallel comparisons (combinational, same cycle)
                S_COMPARE: begin
                    active_node_idx <= node_counter;

                    if (feat_latched_0 >= node_low && feat_latched_0 < node_high)
                        active_node_valid_0 <= 1'b1;
                    if (feat_latched_1 >= node_low && feat_latched_1 < node_high)
                        active_node_valid_1 <= 1'b1;
                    if (feat_latched_2 >= node_low && feat_latched_2 < node_high)
                        active_node_valid_2 <= 1'b1;
                    if (feat_latched_3 >= node_low && feat_latched_3 < node_high)
                        active_node_valid_3 <= 1'b1;

                    state <= S_NEXT;
                end

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
