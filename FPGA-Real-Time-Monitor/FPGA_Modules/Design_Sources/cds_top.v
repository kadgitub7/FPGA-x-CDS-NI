// ============================================================================
// cds_top.v - 4-Lane Parallel CDS Algorithm 4 FPGA Implementation
// ============================================================================
//
// 4-LANE PARALLEL DATA FLOW:
//   1. UART RX receives 4 patients sequentially (4 x [0xAA + 558 bytes])
//   2. Dispatcher routes bytes to sensor_interface_0..3
//   3. tree_traversal scans 215 nodes, comparing all 4 patients per node
//   4. af_engine processes matched nodes with 4 parallel datapaths
//   5. Results sent back as 4 x 5-byte packets over UART TX
//
// RESOURCE SHARING:
//   - model_rom (all BRAMs): 1 shared instance, read once per action
//   - fixedMultiply, fixedDivide: 1 shared instance inside af_engine
//   - sensor_interface: 4 instances (each stores 279 features for 1 patient)
//   - af_accumulator: 4 instances inside af_engine
//   - rangeComparator: 4 instances inside af_engine (combinational)
// ============================================================================

module cds_top(
    input  wire clk,
    input  wire reset,
    input  wire rx_pin,
    output wire tx_pin,
    output wire [1:0] led_decision,
    output wire led_done
);

    localparam CLKS_PER_BIT = 868;
    localparam N_LANES = 4;

    localparam [1:0] DEC_HEALTHY   = 2'b00,
                     DEC_UNHEALTHY = 2'b01,
                     DEC_SCREENING = 2'b10;

    // ========================================================================
    // INTER-MODULE WIRES
    // ========================================================================

    // UART RX
    wire [7:0] rx_byte;
    wire       rx_byte_valid;

    // UART TX
    wire        tx_active;
    wire        tx_serial;
    wire        tx_done;

    // 4 Sensor Interfaces — per-lane load_complete and data_out
    wire        sensor_load_complete_0, sensor_load_complete_1,
                sensor_load_complete_2, sensor_load_complete_3;
    wire [15:0] sensor_data_out_0, sensor_data_out_1,
                sensor_data_out_2, sensor_data_out_3;

    // Tree Traversal
    reg         tree_start;
    wire [8:0]  tree_feature_addr;
    wire [9:0]  tree_rom_addr;
    wire [7:0]  tree_active_node;
    wire        tree_node_valid_0, tree_node_valid_1,
                tree_node_valid_2, tree_node_valid_3;
    wire        tree_all_done;

    // Model ROM data (shared)
    wire [15:0]        rom_tree_data;
    wire               rom_tree_valid;
    wire [15:0]        rom_action_hdr_data;
    wire [15:0]        rom_action_data_out;
    wire signed [15:0] rom_prob_phf_data;
    wire signed [15:0] rom_prob_pgt1_data;
    wire signed [15:0] rom_hr_bmin;
    wire signed [15:0] rom_hr_bmax;

    // AF Engine
    reg          af_start;
    reg  [7:0]   af_node_idx;
    reg  [3:0]   af_lane_mask;
    wire [12:0]  af_action_hdr_addr;
    wire [13:0]  af_action_data_addr;
    wire [11:0]  af_prob_phf_addr;
    wire [7:0]   af_prob_pgt1_addr;
    wire [15:0]  af_hr_read_addr;
    wire [8:0]   af_feature_addr;
    wire [1:0]   af_decision_0, af_decision_1, af_decision_2, af_decision_3;
    wire signed [31:0] af_AF_out_0, af_AF_out_1, af_AF_out_2, af_AF_out_3;
    wire [3:0]   af_alarm_class_0, af_alarm_class_1,
                 af_alarm_class_2, af_alarm_class_3;
    wire         af_done;

    // Decision Logic + Result Sender (reused 4 times sequentially)
    reg          trigger_decision;
    reg  [1:0]   current_decision;
    reg  [3:0]   current_alarm_class;
    reg  signed [31:0] current_af;
    wire [1:0]   dl_final_decision;
    wire [3:0]   dl_final_alarm_class;
    wire         dl_prediction_complete;
    wire [7:0]   sender_tx_data;
    wire         sender_tx_start;
    wire         sender_done;

    // ========================================================================
    // SENSOR READ PORT MUX
    // ========================================================================
    reg sensor_mux_sel;  // 0 = tree, 1 = af_engine

    wire [8:0] sensor_addr_mux = sensor_mux_sel ? af_feature_addr
                                                  : tree_feature_addr;

    // ========================================================================
    // UART BYTE ROUTING — dispatch to active sensor lane
    // ========================================================================
    reg [1:0] load_lane;  // which sensor is currently being loaded (0..3)

    wire sensor_byte_valid_0 = rx_byte_valid & (load_lane == 2'd0);
    wire sensor_byte_valid_1 = rx_byte_valid & (load_lane == 2'd1);
    wire sensor_byte_valid_2 = rx_byte_valid & (load_lane == 2'd2);
    wire sensor_byte_valid_3 = rx_byte_valid & (load_lane == 2'd3);

    // ========================================================================
    // MODULE INSTANTIATIONS
    // ========================================================================

    uart_rx #(.CLKS_PER_BIT(CLKS_PER_BIT)) u_uart_rx (
        .i_Clock(clk), .i_Rx_Serial(rx_pin),
        .o_Rx_Byte(rx_byte), .o_Rx_DV(rx_byte_valid)
    );

    uart_tx #(.CLKS_PER_BIT(CLKS_PER_BIT)) u_uart_tx (
        .i_Clock(clk), .i_Tx_DV(sender_tx_start), .i_Tx_Byte(sender_tx_data),
        .o_Tx_Active(tx_active), .o_Tx_Serial(tx_serial), .o_Tx_Done(tx_done)
    );

    // --- 4 Sensor Interfaces ------------------------------------------------
    sensor_interface u_sensor_0 (
        .clk(clk), .reset(reset), .uart_byte(rx_byte),
        .uart_byte_valid(sensor_byte_valid_0),
        .enableB(1'b1), .addrb(sensor_addr_mux),
        .load_complete(sensor_load_complete_0), .dataOutB(sensor_data_out_0)
    );
    sensor_interface u_sensor_1 (
        .clk(clk), .reset(reset), .uart_byte(rx_byte),
        .uart_byte_valid(sensor_byte_valid_1),
        .enableB(1'b1), .addrb(sensor_addr_mux),
        .load_complete(sensor_load_complete_1), .dataOutB(sensor_data_out_1)
    );
    sensor_interface u_sensor_2 (
        .clk(clk), .reset(reset), .uart_byte(rx_byte),
        .uart_byte_valid(sensor_byte_valid_2),
        .enableB(1'b1), .addrb(sensor_addr_mux),
        .load_complete(sensor_load_complete_2), .dataOutB(sensor_data_out_2)
    );
    sensor_interface u_sensor_3 (
        .clk(clk), .reset(reset), .uart_byte(rx_byte),
        .uart_byte_valid(sensor_byte_valid_3),
        .enableB(1'b1), .addrb(sensor_addr_mux),
        .load_complete(sensor_load_complete_3), .dataOutB(sensor_data_out_3)
    );

    // --- Model ROM (1 shared instance) ----------------------------------------
    model_rom u_model_rom (
        .clk(clk),
        .tree_read_addr(tree_rom_addr), .tree_re(1'b1),
        .tree_data(rom_tree_data), .tree_valid(rom_tree_valid),
        .action_hdr_addr(af_action_hdr_addr),
        .action_data_addr(af_action_data_addr),
        .prob_phf_addr(af_prob_phf_addr),
        .prob_pgt1_addr(af_prob_pgt1_addr),
        .hr_read_addr(af_hr_read_addr),
        .action_hdr_data(rom_action_hdr_data),
        .action_data_out(rom_action_data_out),
        .prob_phf_data(rom_prob_phf_data),
        .prob_pgt1_data(rom_prob_pgt1_data),
        .hr_bmin(rom_hr_bmin), .hr_bmax(rom_hr_bmax)
    );

    // --- Tree Traversal (4-lane) ---------------------------------------------
    tree_traversal u_tree_trav (
        .clk(clk), .reset(reset), .start(tree_start),
        .user_feature_value_0(sensor_data_out_0),
        .user_feature_value_1(sensor_data_out_1),
        .user_feature_value_2(sensor_data_out_2),
        .user_feature_value_3(sensor_data_out_3),
        .tree_data(rom_tree_data),
        .feature_read_addr(tree_feature_addr),
        .tree_read_addr(tree_rom_addr),
        .active_node_idx(tree_active_node),
        .active_node_valid_0(tree_node_valid_0),
        .active_node_valid_1(tree_node_valid_1),
        .active_node_valid_2(tree_node_valid_2),
        .active_node_valid_3(tree_node_valid_3),
        .node_done(), .all_done(tree_all_done)
    );

    // --- AF Engine (4-lane) --------------------------------------------------
    af_engine u_af_engine (
        .clk(clk), .reset(reset), .start(af_start),
        .node_idx(af_node_idx), .lane_mask(af_lane_mask),
        .AF_init_0(running_AF_0), .AF_init_1(running_AF_1),
        .AF_init_2(running_AF_2), .AF_init_3(running_AF_3),
        .action_hdr_data(rom_action_hdr_data),
        .action_data_out(rom_action_data_out),
        .prob_phf_data(rom_prob_phf_data),
        .prob_pgt1_data(rom_prob_pgt1_data),
        .hr_bmin(rom_hr_bmin), .hr_bmax(rom_hr_bmax),
        .user_feature_value_0(sensor_data_out_0),
        .user_feature_value_1(sensor_data_out_1),
        .user_feature_value_2(sensor_data_out_2),
        .user_feature_value_3(sensor_data_out_3),
        .action_hdr_addr(af_action_hdr_addr),
        .action_data_addr(af_action_data_addr),
        .prob_phf_addr(af_prob_phf_addr),
        .prob_pgt1_addr(af_prob_pgt1_addr),
        .hr_read_addr(af_hr_read_addr),
        .feature_read_addr(af_feature_addr),
        .decision_0(af_decision_0), .decision_1(af_decision_1),
        .decision_2(af_decision_2), .decision_3(af_decision_3),
        .AF_out_0(af_AF_out_0), .AF_out_1(af_AF_out_1),
        .AF_out_2(af_AF_out_2), .AF_out_3(af_AF_out_3),
        .alarm_class_0(af_alarm_class_0), .alarm_class_1(af_alarm_class_1),
        .alarm_class_2(af_alarm_class_2), .alarm_class_3(af_alarm_class_3),
        .done(af_done)
    );

    // --- Decision Logic + Result Sender (shared, called 4 times) ----------
    decision_logic u_decision_logic (
        .clk(clk), .reset(reset),
        .decision_made(trigger_decision),
        .decision(current_decision),
        .alarm_class(current_alarm_class),
        .final_decision(dl_final_decision),
        .final_alarm_class(dl_final_alarm_class),
        .prediction_complete(dl_prediction_complete)
    );

    result_sender u_result_sender (
        .clk(clk), .reset(reset),
        .send_start(dl_prediction_complete),
        .final_decision(dl_final_decision),
        .final_alarm_class(dl_final_alarm_class),
        .af_final(current_af),
        .tx_busy(tx_active),
        .tx_data(sender_tx_data), .tx_start(sender_tx_start),
        .done(sender_done)
    );

    assign tx_pin       = tx_serial;
    assign led_decision = dl_final_decision;
    assign led_done     = (state == S_DONE);

    // ========================================================================
    // MASTER FSM
    // ========================================================================
    localparam [4:0]
        S_IDLE        = 5'd0,
        S_WAIT_LOAD   = 5'd1,   // wait for current lane to load
        S_START_TREE  = 5'd2,
        S_SCAN_TREE   = 5'd3,
        S_CHECK_NODES = 5'd4,
        S_FIND_NODE   = 5'd5,   // find next node any active lane needs
        S_START_AF    = 5'd6,
        S_WAIT_AF     = 5'd7,
        S_CHECK_AF    = 5'd8,
        S_NEXT_NODE   = 5'd9,
        S_OUTPUT      = 5'd10,  // send result for current send_lane
        S_WAIT_SEND   = 5'd11,
        S_NEXT_SEND   = 5'd12,  // advance to next lane's result
        S_DONE        = 5'd13;

    reg [4:0] state;

    // ── Per-lane node matching (215-bit bitmaps) ────────────────────
    reg [214:0] node_matched_0, node_matched_1,
                node_matched_2, node_matched_3;
    reg [7:0]   current_node;        // 0..214, iterating during AF phase
    reg [3:0]   lane_done;           // per-lane: decision reached
    reg [1:0]   lane_decision [0:3]; // per-lane final decision
    reg [3:0]   lane_alarm    [0:3]; // per-lane alarm class
    reg signed [31:0] running_AF_0, running_AF_1, running_AF_2, running_AF_3;
    reg [1:0]   send_lane;           // which lane's result we're sending (0..3)

    always @(posedge clk) begin
        if (reset) begin
            state              <= S_IDLE;
            tree_start         <= 1'b0;
            af_start           <= 1'b0;
            trigger_decision   <= 1'b0;
            sensor_mux_sel     <= 1'b0;
            load_lane          <= 2'd0;
            lane_done          <= 4'd0;
            current_node       <= 8'd0;
            send_lane          <= 2'd0;
            running_AF_0 <= 32'sd0; running_AF_1 <= 32'sd0;
            running_AF_2 <= 32'sd0; running_AF_3 <= 32'sd0;
            lane_decision[0] <= DEC_HEALTHY; lane_decision[1] <= DEC_HEALTHY;
            lane_decision[2] <= DEC_HEALTHY; lane_decision[3] <= DEC_HEALTHY;
            lane_alarm[0] <= 4'd0; lane_alarm[1] <= 4'd0;
            lane_alarm[2] <= 4'd0; lane_alarm[3] <= 4'd0;
            node_matched_0 <= 215'd0; node_matched_1 <= 215'd0;
            node_matched_2 <= 215'd0; node_matched_3 <= 215'd0;
            af_node_idx    <= 8'd0;
            af_lane_mask   <= 4'd0;
            current_decision    <= DEC_HEALTHY;
            current_alarm_class <= 4'd0;
            current_af          <= 32'sd0;
        end
        else begin
            tree_start       <= 1'b0;
            af_start         <= 1'b0;
            trigger_decision <= 1'b0;

            case (state)

                // ────────────────────────────────────────────────────
                // LOADING: wait for all 4 sensors to load sequentially
                // Python sends: [0xAA+558] [0xAA+558] [0xAA+558] [0xAA+558]
                // ────────────────────────────────────────────────────
                S_IDLE: begin
                    load_lane <= 2'd0;
                    state     <= S_WAIT_LOAD;
                end

                S_WAIT_LOAD: begin
                    // Check if current lane's sensor just finished loading
                    if ((load_lane == 2'd0 && sensor_load_complete_0) ||
                        (load_lane == 2'd1 && sensor_load_complete_1) ||
                        (load_lane == 2'd2 && sensor_load_complete_2) ||
                        (load_lane == 2'd3 && sensor_load_complete_3)) begin

                        if (load_lane == 2'd3) begin
                            // All 4 loaded — start processing
                            lane_done <= 4'd0;
                            lane_decision[0] <= DEC_HEALTHY;
                            lane_decision[1] <= DEC_HEALTHY;
                            lane_decision[2] <= DEC_HEALTHY;
                            lane_decision[3] <= DEC_HEALTHY;
                            lane_alarm[0] <= 4'd0; lane_alarm[1] <= 4'd0;
                            lane_alarm[2] <= 4'd0; lane_alarm[3] <= 4'd0;
                            running_AF_0 <= 32'sd0; running_AF_1 <= 32'sd0;
                            running_AF_2 <= 32'sd0; running_AF_3 <= 32'sd0;
                            node_matched_0 <= 215'd0; node_matched_1 <= 215'd0;
                            node_matched_2 <= 215'd0; node_matched_3 <= 215'd0;
                            sensor_mux_sel <= 1'b0;
                            state <= S_START_TREE;
                        end
                        else begin
                            load_lane <= load_lane + 2'd1;
                            // stay in S_WAIT_LOAD for next lane
                        end
                    end
                end

                // ────────────────────────────────────────────────────
                // TREE TRAVERSAL: scan 215 nodes, record per-lane matches
                // ────────────────────────────────────────────────────
                S_START_TREE: begin
                    tree_start <= 1'b1;
                    state      <= S_SCAN_TREE;
                end

                S_SCAN_TREE: begin
                    // Capture per-lane node matches into bitmaps
                    if (tree_node_valid_0)
                        node_matched_0[tree_active_node] <= 1'b1;
                    if (tree_node_valid_1)
                        node_matched_1[tree_active_node] <= 1'b1;
                    if (tree_node_valid_2)
                        node_matched_2[tree_active_node] <= 1'b1;
                    if (tree_node_valid_3)
                        node_matched_3[tree_active_node] <= 1'b1;

                    if (tree_all_done)
                        state <= S_CHECK_NODES;
                end

                // ────────────────────────────────────────────────────
                // AF PROCESSING: iterate nodes, process active lanes
                // ────────────────────────────────────────────────────
                S_CHECK_NODES: begin
                    current_node   <= 8'd0;
                    sensor_mux_sel <= 1'b1;  // af_engine controls sensors
                    state          <= S_FIND_NODE;
                end

                // Find the next node that any non-done lane matched
                S_FIND_NODE: begin
                    if (current_node > 8'd214 || lane_done == 4'b1111) begin
                        // All nodes scanned or all lanes decided
                        state <= S_OUTPUT;
                    end
                    else begin
                        // Compute lane_mask: lanes that matched this node AND aren't done
                        af_lane_mask <= {
                            node_matched_3[current_node] & ~lane_done[3],
                            node_matched_2[current_node] & ~lane_done[2],
                            node_matched_1[current_node] & ~lane_done[1],
                            node_matched_0[current_node] & ~lane_done[0]
                        };
                        // If any lane needs this node, process it
                        if ((node_matched_0[current_node] & ~lane_done[0]) |
                            (node_matched_1[current_node] & ~lane_done[1]) |
                            (node_matched_2[current_node] & ~lane_done[2]) |
                            (node_matched_3[current_node] & ~lane_done[3])) begin
                            state <= S_START_AF;
                        end
                        else begin
                            current_node <= current_node + 8'd1;
                            // stay in S_FIND_NODE
                        end
                    end
                end

                S_START_AF: begin
                    af_node_idx <= current_node;
                    af_start    <= 1'b1;
                    state       <= S_WAIT_AF;
                end

                S_WAIT_AF: begin
                    if (af_done) state <= S_CHECK_AF;
                end

                // Check per-lane results and mark done lanes
                S_CHECK_AF: begin
                    // Lane 0
                    if (af_lane_mask[0] && !lane_done[0]) begin
                        running_AF_0 <= af_AF_out_0;
                        if (af_decision_0 == DEC_UNHEALTHY) begin
                            lane_done[0] <= 1'b1;
                            lane_decision[0] <= DEC_UNHEALTHY;
                            lane_alarm[0] <= af_alarm_class_0;
                        end
                        else if (af_decision_0 == DEC_HEALTHY) begin
                            lane_done[0] <= 1'b1;
                            lane_decision[0] <= DEC_HEALTHY;
                        end
                        else begin
                            lane_decision[0] <= DEC_SCREENING;
                        end
                    end
                    // Lane 1
                    if (af_lane_mask[1] && !lane_done[1]) begin
                        running_AF_1 <= af_AF_out_1;
                        if (af_decision_1 == DEC_UNHEALTHY) begin
                            lane_done[1] <= 1'b1;
                            lane_decision[1] <= DEC_UNHEALTHY;
                            lane_alarm[1] <= af_alarm_class_1;
                        end
                        else if (af_decision_1 == DEC_HEALTHY) begin
                            lane_done[1] <= 1'b1;
                            lane_decision[1] <= DEC_HEALTHY;
                        end
                        else begin
                            lane_decision[1] <= DEC_SCREENING;
                        end
                    end
                    // Lane 2
                    if (af_lane_mask[2] && !lane_done[2]) begin
                        running_AF_2 <= af_AF_out_2;
                        if (af_decision_2 == DEC_UNHEALTHY) begin
                            lane_done[2] <= 1'b1;
                            lane_decision[2] <= DEC_UNHEALTHY;
                            lane_alarm[2] <= af_alarm_class_2;
                        end
                        else if (af_decision_2 == DEC_HEALTHY) begin
                            lane_done[2] <= 1'b1;
                            lane_decision[2] <= DEC_HEALTHY;
                        end
                        else begin
                            lane_decision[2] <= DEC_SCREENING;
                        end
                    end
                    // Lane 3
                    if (af_lane_mask[3] && !lane_done[3]) begin
                        running_AF_3 <= af_AF_out_3;
                        if (af_decision_3 == DEC_UNHEALTHY) begin
                            lane_done[3] <= 1'b1;
                            lane_decision[3] <= DEC_UNHEALTHY;
                            lane_alarm[3] <= af_alarm_class_3;
                        end
                        else if (af_decision_3 == DEC_HEALTHY) begin
                            lane_done[3] <= 1'b1;
                            lane_decision[3] <= DEC_HEALTHY;
                        end
                        else begin
                            lane_decision[3] <= DEC_SCREENING;
                        end
                    end

                    state <= S_NEXT_NODE;
                end

                S_NEXT_NODE: begin
                    if (lane_done == 4'b1111) begin
                        state <= S_OUTPUT;
                    end
                    else begin
                        current_node <= current_node + 8'd1;
                        state <= S_FIND_NODE;
                    end
                end

                // ────────────────────────────────────────────────────
                // OUTPUT: send 4 results sequentially via UART
                // ────────────────────────────────────────────────────
                S_OUTPUT: begin
                    send_lane <= 2'd0;
                    // Set up lane 0's result
                    current_decision    <= lane_decision[0];
                    current_alarm_class <= lane_alarm[0];
                    current_af          <= running_AF_0;
                    trigger_decision    <= 1'b1;
                    state               <= S_WAIT_SEND;
                end

                S_WAIT_SEND: begin
                    if (sender_done) begin
                        state <= S_NEXT_SEND;
                    end
                end

                S_NEXT_SEND: begin
                    if (send_lane == 2'd3) begin
                        // All 4 results sent
                        state <= S_DONE;
                    end
                    else begin
                        send_lane <= send_lane + 2'd1;
                        // Set up next lane's result
                        case (send_lane + 2'd1)
                            2'd1: begin
                                current_decision <= lane_decision[1];
                                current_alarm_class <= lane_alarm[1];
                                current_af <= running_AF_1;
                            end
                            2'd2: begin
                                current_decision <= lane_decision[2];
                                current_alarm_class <= lane_alarm[2];
                                current_af <= running_AF_2;
                            end
                            2'd3: begin
                                current_decision <= lane_decision[3];
                                current_alarm_class <= lane_alarm[3];
                                current_af <= running_AF_3;
                            end
                        endcase
                        trigger_decision <= 1'b1;
                        state <= S_WAIT_SEND;
                    end
                end

                S_DONE: begin
                    state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
