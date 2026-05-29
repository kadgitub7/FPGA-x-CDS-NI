`timescale 1ns / 1ps

module tree_traversal (
    input wire        clk,
    input wire        reset,
    input wire        start,

    input wire signed [15:0]  user_feature_value,   // from sensor_interface
    input wire [15:0] tree_data,            // from model_rom

    output reg [8:0]  feature_read_addr,    // to sensor_interface
    output reg [9:0]  tree_read_addr,       // to model_rom
    output reg [7:0]  active_node_idx,
    output reg        active_node_valid,
    output reg        node_done,
    output reg        all_done
);

    // ── State encoding ──────────────────────────────────────────────
    localparam [3:0]
        S_IDLE      = 4'd0,
        S_ROOT_EMIT = 4'd1,    // emit root match unconditionally
        S_READ_W0   = 4'd2,    // put word-0 address on ROM bus
        S_LATCH_W0  = 4'd3,    // latch word 0, advance address to word 1
        S_LATCH_W1  = 4'd4,    // latch word 1 (low), advance to word 2
        S_LATCH_W2  = 4'd5,    // latch word 2 (high)
        S_READ_FEAT = 4'd6,    // put feature address on sensor bus
        S_WAIT_FEAT = 4'd7,    // wait one cycle, latch feature value
        S_COMPARE   = 4'd8,    // compare feature against [low, high)
        S_NEXT      = 4'd9,    // advance node_counter or finish
        S_DONE      = 4'd10;

    // ── Internal registers ──────────────────────────────────────────
    reg [3:0]  state;
    reg [7:0]  node_counter;        // linear iterator: 1 -> 214
    reg [8:0]  branch_feat_idx;     // which feature this node branches on
    reg signed [15:0] node_low;            // branch low bound  (ROM word 1)
    reg signed [15:0] node_high;           // branch high bound (ROM word 2)
    reg signed [15:0]  feat_latched;        // user's feature value, held stable

    wire [9:0] node_base = {1'b0, node_counter, 1'b0}  
                         + {2'b00, node_counter};   

    wire signed [15:0] feat_ext = {8'd0, feat_latched};

    always @(posedge clk) begin
        active_node_valid <= 1'b0;
        node_done         <= 1'b0;
        all_done          <= 1'b0;

        if (reset) begin
            state             <= S_IDLE;
            node_counter      <= 8'd0;
            active_node_idx   <= 8'd0;
            feature_read_addr <= 8'd0;
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

                S_ROOT_EMIT: begin
                    active_node_idx   <= 8'd0;
                    active_node_valid <= 1'b1;
                    tree_read_addr    <= node_base;  
                    state             <= S_LATCH_W0;
                end

                S_READ_W0: begin
                    tree_read_addr <= node_base;
                    state          <= S_LATCH_W0;
                end

                S_LATCH_W0: begin
                    branch_feat_idx <= tree_data[8:0]; 
                    tree_read_addr  <= tree_read_addr + 10'd1;  
                    state           <= S_LATCH_W1;
                end

                S_LATCH_W1: begin
                    node_low       <= tree_data;
                    tree_read_addr <= tree_read_addr + 10'd1;
                    state          <= S_LATCH_W2;
                end

                S_LATCH_W2: begin
                    node_high <= tree_data;
                    state     <= S_READ_FEAT;
                end

                S_READ_FEAT: begin
                    feature_read_addr <= branch_feat_idx;
                    state             <= S_WAIT_FEAT;
                end

                S_WAIT_FEAT: begin
                    feat_latched <= user_feature_value;
                    state        <= S_COMPARE;
                end

                S_COMPARE: begin
                    if (feat_ext >= node_low && feat_ext < node_high) begin
                        active_node_idx   <= node_counter;  // THIS node matched
                        active_node_valid <= 1'b1;          // one-cycle pulse
                    end
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
