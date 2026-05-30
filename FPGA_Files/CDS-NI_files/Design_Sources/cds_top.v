// ============================================================================
// cds_top.v — Top-level module for the CDS Algorithm 4 FPGA implementation
// ============================================================================
//
// DATA FLOW OVERVIEW:
//   1. UART RX receives 558 bytes from the PC (one patient's sensor data)
//   2. sensor_interface assembles bytes into 279 x 16-bit features
//   3. tree_traversal scans all 215 tree nodes, buffering matches
//   4. af_engine processes each matched node (12 diseases each)
//   5. decision_logic latches the final aggregated decision
//   6. result_sender serializes the result into 5 UART bytes
//   7. UART TX sends those bytes back to the PC
//
// IMPORTANT VERILOG CONCEPT — Wiring Modules Together:
//   In Verilog, you do NOT use object.attribute syntax (that's Python/Java).
//   Instead you:
//     1. Declare a wire or reg with YOUR chosen name
//     2. Connect it to module ports using .port_name(your_wire_name)
//   Think of wires like copper traces on a PCB — you name the trace,
//   then solder module pins to it.
//
//   Rule of thumb:
//     - Signal is an OUTPUT of some module    → declare as "wire"
//     - YOUR top-level FSM drives it          → declare as "reg"
//     - It's a constant (like tree_re = 1)    → use a literal in the port map
// ============================================================================

module cds_top(
    input  wire clk,                 // board oscillator (100 MHz on Basys3)
    input  wire reset,               // physical reset button
    input  wire rx_pin,              // UART receive pin (from PC's TX)
    output wire tx_pin,              // UART transmit pin (to PC's RX)
    output wire [1:0] led_decision,  // LEDs: 00=healthy, 01=unhealthy, 10=screening
    output wire led_done             // LED: lights when prediction is complete
);

    // ========================================================================
    // PARAMETERS
    // ========================================================================

    // CLKS_PER_BIT: number of clock cycles per one UART bit.
    // Formula: clock_freq / baud_rate = 100,000,000 / 115,200 = 868
    localparam CLKS_PER_BIT = 868;

    // Decision codes — must match the values inside af_engine
    localparam [1:0] DEC_HEALTHY   = 2'b00,
                     DEC_UNHEALTHY = 2'b01,
                     DEC_SCREENING = 2'b10;


    // ========================================================================
    // INTER-MODULE WIRES AND REGS
    // ------------------------------------------------------------------------
    // Every signal that travels between two modules needs a wire or reg
    // declared here. The name is YOUR choice — pick something descriptive.
    //
    // KEY RULE:
    //   wire → for signals that are OUTPUTS of an instantiated module
    //          (the module drives them, you just name the trace)
    //   reg  → for signals that THIS module's always block drives
    //          (your master FSM controls them)
    //
    // Naming convention used here:
    //   rx_*       → from/to UART receiver
    //   sensor_*   → from/to sensor_interface
    //   tree_*     → from/to tree_traversal
    //   rom_*      → from/to model_rom
    //   af_*       → from/to af_engine
    //   dl_*       → from/to decision_logic
    //   sender_*   → from/to result_sender
    //   tx_*       → from/to UART transmitter
    // ========================================================================

    // --- UART RX outputs ---
    wire [7:0] rx_byte;            // the 8-bit byte just received
    wire       rx_byte_valid;      // pulses high for 1 clock when rx_byte is ready

    // --- Sensor Interface outputs ---
    wire        sensor_load_complete;  // pulses when all 279 features are loaded
    wire [15:0] sensor_data_out;       // read port: feature value at requested address

    // --- Tree Traversal ---
    // tree_start is a REG because our master FSM drives it (pulses it)
    reg         tree_start;
    // These are WIRES because they are outputs of the tree_traversal module
    wire [8:0]  tree_feature_addr;     // tree's address request to sensor
    wire [9:0]  tree_rom_addr;         // tree's address request to model_rom
    wire [7:0]  tree_active_node;      // index of the node that just matched
    wire        tree_node_valid;       // pulses for 1 cycle per matched node
    wire        tree_node_done;        // (available but unused in master FSM)
    wire        tree_all_done;         // pulses when traversal is finished

    // --- Model ROM data outputs (all wires — they are ROM outputs) ---
    wire [15:0]        rom_tree_data;        // tree topology word
    wire               rom_tree_valid;       // tree BRAM valid (unused, always 1)
    wire [15:0]        rom_action_hdr_data;  // action header word
    wire [15:0]        rom_action_data_out;  // action data word
    wire signed [15:0] rom_prob_phf_data;    // P(h,f)
    wire signed [15:0] rom_prob_pgt1_data;   // 1/P(h>1,f) reciprocal
    wire signed [15:0] rom_hr_bmin;          // healthy range lower bound
    wire signed [15:0] rom_hr_bmax;          // healthy range upper bound

    // --- AF Engine ---
    // af_start, af_node_idx, af_init_value are REGS — our FSM drives them
    reg          af_start;
    reg  [7:0]   af_node_idx;           // which tree node to process
    reg  signed [31:0] af_init_value;   // carry-forward AF from previous node
    // These are WIRES — af_engine outputs them
    wire [12:0]  af_action_hdr_addr;    // address → model_rom action header BRAM
    wire [13:0]  af_action_data_addr;   // address → model_rom action data BRAM
    wire [11:0]  af_prob_phf_addr;      // address → model_rom P(h,f) BRAM
    wire [7:0]   af_prob_pgt1_addr;     // address → model_rom P(h>1,f) BRAM
    wire [16:0]  af_hr_read_addr;       // address → model_rom healthy range BRAM
    wire [8:0]   af_feature_addr;       // address → sensor read port (through MUX)
    wire [1:0]   af_decision;           // result: HEALTHY/UNHEALTHY/SCREENING
    wire signed [31:0] af_AF_out;       // result: AF after processing this node
    wire [3:0]   af_alarm_class;        // result: which disease (0-11) triggered alarm
    wire         af_done;               // high when af_engine finishes

    // --- Decision Logic ---
    // trigger_decision, master_decision, master_alarm_class are REGS — our FSM drives them
    reg          trigger_decision;
    reg  [1:0]   master_decision;       // aggregated decision across ALL nodes
    reg  [3:0]   master_alarm_class;    // alarm class from the UNHEALTHY node
    // These are WIRES — decision_logic outputs them
    wire [1:0]   dl_final_decision;
    wire [3:0]   dl_final_alarm_class;
    wire         dl_prediction_complete; // pulses 1 cycle after decision is latched

    // --- Result Sender ---
    // All wires — result_sender outputs them
    wire [7:0]  sender_tx_data;
    wire        sender_tx_start;
    wire        sender_done;

    // --- UART TX ---
    // All wires — uart_tx outputs them
    wire        tx_active;   // high while transmitting (= "busy" for result_sender)
    wire        tx_serial;   // the serial bit stream to the pin
    wire        tx_done;     // pulses when one byte finishes (unused here)


    // ========================================================================
    // SENSOR READ PORT MUX
    // ------------------------------------------------------------------------
    // PROBLEM: both tree_traversal and af_engine need to read features from
    // sensor_interface, but it only has ONE read port (addrb/enableB/dataOutB).
    //
    // SOLUTION: a multiplexer (MUX) controlled by the master FSM.
    //   sensor_mux_sel = 0 → tree_traversal controls the sensor read port
    //   sensor_mux_sel = 1 → af_engine controls the sensor read port
    //
    // The data output (sensor_data_out) fans out to BOTH modules — they
    // only look at it when they're actually running.
    // ========================================================================
    reg sensor_mux_sel;

    wire [8:0] sensor_addr_mux = sensor_mux_sel ? af_feature_addr
                                                  : tree_feature_addr;


    // ========================================================================
    // MODULE INSTANTIATIONS
    // ------------------------------------------------------------------------
    // This is where we "place" each module and "solder" its pins to our wires.
    //
    // Syntax:  module_name #(.PARAM(value)) instance_name (
    //              .port_name(our_wire_name),
    //              ...
    //          );
    //
    // The .port_name is defined INSIDE the module file.
    // The (our_wire_name) is what WE declared above.
    // ========================================================================

    // --- 1. UART Receiver ------------------------------------------------
    // Converts serial bits on rx_pin into parallel bytes.
    // NOTE: The parameter #(.CLKS_PER_BIT(868)) is REQUIRED — without it
    //       the module doesn't know how fast our clock is relative to baud.
    // NOTE: The port is called "i_Clock" (not "i_Clk") — always check the
    //       module's actual port names, typos cause "unconnected port" errors.
    uart_rx #(.CLKS_PER_BIT(CLKS_PER_BIT)) u_uart_rx (
        .i_Clock    (clk),
        .i_Rx_Serial(rx_pin),
        .o_Rx_Byte  (rx_byte),          // output → wire rx_byte
        .o_Rx_DV    (rx_byte_valid)      // output → wire rx_byte_valid
    );

    // --- 2. UART Transmitter ---------------------------------------------
    // Converts parallel bytes into serial bits on tx_pin.
    // result_sender feeds it bytes one at a time via sender_tx_start/sender_tx_data.
    // tx_active = 1 while a byte is being sent (result_sender uses this as "busy").
    uart_tx #(.CLKS_PER_BIT(CLKS_PER_BIT)) u_uart_tx (
        .i_Clock    (clk),
        .i_Tx_DV    (sender_tx_start),   // input ← from result_sender
        .i_Tx_Byte  (sender_tx_data),    // input ← from result_sender
        .o_Tx_Active(tx_active),          // output → to result_sender as tx_busy
        .o_Tx_Serial(tx_serial),          // output → to tx_pin (see assign below)
        .o_Tx_Done  (tx_done)             // output → unused
    );

    // --- 3. Sensor Interface ---------------------------------------------
    // Receives bytes from UART, pairs them into 279 x 16-bit features.
    // Provides a synchronous read port: put address on addrb, get data on
    // dataOutB one cycle later (standard BRAM read pattern).
    //
    // enableB is tied to 1'b1 (always reading). This is fine — the data
    // output just follows whatever address is on the bus. The modules
    // only look at the data when they need it.
    sensor_interface u_sensor (
        .clk            (clk),
        .reset          (reset),
        .uart_byte      (rx_byte),            // ← from uart_rx
        .uart_byte_valid(rx_byte_valid),       // ← from uart_rx
        .enableB        (1'b1),                // always enabled
        .addrb          (sensor_addr_mux),     // ← MUXed: tree or af_engine
        .load_complete  (sensor_load_complete), // output → wire
        .dataOutB       (sensor_data_out)       // output → wire
    );

    // --- 4. Model ROM ----------------------------------------------------
    // All 6 BRAMs live here. tree_re is tied to 1'b1 so the tree BRAM
    // reads every cycle (tree_traversal expects data 1 cycle after address).
    // The other 5 BRAMs have re tied to 1'b1 internally in model_rom.v.
    model_rom u_model_rom (
        .clk             (clk),
        // Tree BRAM — driven by tree_traversal
        .tree_read_addr  (tree_rom_addr),         // ← from tree_traversal
        .tree_re         (1'b1),                   // always reading
        .tree_data       (rom_tree_data),           // → to tree_traversal
        .tree_valid      (rom_tree_valid),           // → unused
        // AF-related BRAMs — driven by af_engine
        .action_hdr_addr (af_action_hdr_addr),     // ← from af_engine
        .action_data_addr(af_action_data_addr),     // ← from af_engine
        .prob_phf_addr   (af_prob_phf_addr),        // ← from af_engine
        .prob_pgt1_addr  (af_prob_pgt1_addr),       // ← from af_engine
        .hr_read_addr    (af_hr_read_addr),         // ← from af_engine
        .action_hdr_data (rom_action_hdr_data),     // → to af_engine
        .action_data_out (rom_action_data_out),     // → to af_engine
        .prob_phf_data   (rom_prob_phf_data),       // → to af_engine
        .prob_pgt1_data  (rom_prob_pgt1_data),      // → to af_engine
        .hr_bmin         (rom_hr_bmin),             // → to af_engine
        .hr_bmax         (rom_hr_bmax)              // → to af_engine
    );

    // --- 5. Tree Traversal -----------------------------------------------
    // Scans all 215 nodes. For each node: read branch feature+bounds from
    // model_rom, read the patient's value from sensor, compare.
    // Pulses tree_node_valid for each match, then pulses tree_all_done.
    tree_traversal u_tree_trav (
        .clk              (clk),
        .reset            (reset),
        .start            (tree_start),             // ← pulsed by master FSM
        .user_feature_value(sensor_data_out),       // ← from sensor read port
        .tree_data        (rom_tree_data),           // ← from model_rom
        .feature_read_addr(tree_feature_addr),       // → to sensor MUX
        .tree_read_addr   (tree_rom_addr),           // → to model_rom
        .active_node_idx  (tree_active_node),        // → buffered by master FSM
        .active_node_valid(tree_node_valid),          // → checked by master FSM
        .node_done        (tree_node_done),           // → unused
        .all_done         (tree_all_done)             // → checked by master FSM
    );

    // --- 6. AF Engine ----------------------------------------------------
    // Processes ONE node: iterates 12 diseases, computes AF using multiply,
    // divide, accumulate, range-check. Outputs decision + alarm_class + AF_out.
    //
    // NOTE: This is af_engine, NOT rl_action_selector!
    //   - af_engine: the main 26-state FSM, has node_idx/AF_init/decision outputs
    //   - rl_action_selector: a helper used INSIDE af_engine (already wired there)
    //   These are completely different modules with different ports.
    //
    // TODO: af_engine.v S_DONE currently stays stuck (done=1 forever, no
    //       transition). Add "state <= S_IDLE;" in S_DONE so it can be
    //       restarted for the next matched node.
    af_engine u_af_engine (
        .clk              (clk),
        .reset            (reset),
        .start            (af_start),               // ← pulsed by master FSM
        .node_idx         (af_node_idx),             // ← set by master FSM from buffer
        .AF_init          (af_init_value),            // ← carry-forward from previous node
        // Data from model_rom BRAMs
        .action_hdr_data  (rom_action_hdr_data),
        .action_data_out  (rom_action_data_out),
        .prob_phf_data    (rom_prob_phf_data),
        .prob_pgt1_data   (rom_prob_pgt1_data),
        .hr_bmin          (rom_hr_bmin),
        .hr_bmax          (rom_hr_bmax),
        // Data from sensor (through the MUX)
        .user_feature_value(sensor_data_out),
        // Address outputs to model_rom
        .action_hdr_addr  (af_action_hdr_addr),
        .action_data_addr (af_action_data_addr),
        .prob_phf_addr    (af_prob_phf_addr),
        .prob_pgt1_addr   (af_prob_pgt1_addr),
        .hr_read_addr     (af_hr_read_addr),
        // Address output to sensor (through the MUX)
        .feature_read_addr(af_feature_addr),
        // Result outputs
        .decision         (af_decision),
        .AF_out           (af_AF_out),
        .alarm_class      (af_alarm_class),
        .done             (af_done)
    );

    // --- 7. Decision Logic -----------------------------------------------
    // Latches master_decision and master_alarm_class on the rising edge of
    // trigger_decision. One cycle later, pulses dl_prediction_complete,
    // which auto-triggers result_sender (see send_start wiring below).
    //
    // WHY we don't wire af_engine.done directly to decision_made:
    //   The master FSM needs to aggregate decisions across multiple nodes
    //   (UNHEALTHY from ANY node = alarm). So the FSM collects results,
    //   decides the final answer, THEN pulses trigger_decision once.
    decision_logic u_decision_logic (
        .clk                (clk),
        .reset              (reset),
        .decision_made      (trigger_decision),       // ← pulsed by master FSM
        .decision           (master_decision),         // ← aggregated by master FSM
        .alarm_class        (master_alarm_class),      // ← set by master FSM
        .final_decision     (dl_final_decision),       // → to result_sender + LEDs
        .final_alarm_class  (dl_final_alarm_class),    // → to result_sender
        .prediction_complete(dl_prediction_complete)   // → triggers result_sender
    );

    // --- 8. Result Sender ------------------------------------------------
    // Sends a 5-byte packet over UART: [decision_byte, AF3, AF2, AF1, AF0].
    // Triggered by dl_prediction_complete (auto-chain from decision_logic).
    //
    // TIMING CHAIN:
    //   Master FSM pulses trigger_decision
    //     → decision_logic latches (1 cycle)
    //     → decision_logic pulses prediction_complete (1 more cycle)
    //     → result_sender starts sending (many cycles for 5 UART bytes)
    //     → result_sender pulses sender_done when finished
    result_sender u_result_sender (
        .clk            (clk),
        .reset          (reset),
        .send_start     (dl_prediction_complete),   // ← auto-triggered
        .final_decision (dl_final_decision),        // ← from decision_logic
        .final_alarm_class(dl_final_alarm_class),   // ← from decision_logic
        .af_final       (running_AF),               // ← cumulative AF across all nodes
        .tx_busy        (tx_active),                // ← from uart_tx (1=busy)
        .tx_data        (sender_tx_data),           // → to uart_tx
        .tx_start       (sender_tx_start),          // → to uart_tx
        .done           (sender_done)               // → checked by master FSM
    );


    // ========================================================================
    // OUTPUT PIN ASSIGNMENTS
    // ------------------------------------------------------------------------
    // "assign" creates a permanent hardwired connection (like a solder joint).
    // These connect internal signals to the physical FPGA output pins.
    // ========================================================================
    assign tx_pin       = tx_serial;           // uart_tx serial output → FPGA pin
    assign led_decision = dl_final_decision;   // show decision on 2 LEDs
    assign led_done     = (state == S_DONE);   // light up when prediction complete


    // ========================================================================
    // MASTER FSM — State Encoding
    // ========================================================================
    localparam [3:0]
        S_IDLE        = 4'd0,    // wait for sensor data to arrive
        S_START_TREE  = 4'd1,    // pulse tree_traversal to begin
        S_SCAN_TREE   = 4'd2,    // buffer matched nodes as tree runs
        S_CHECK_NODES = 4'd3,    // any matches? → start AF or skip to output
        S_START_AF    = 4'd4,    // set up af_engine inputs, pulse start
        S_WAIT_AF     = 4'd5,    // wait for af_engine to finish
        S_CHECK_AF    = 4'd6,    // read af_engine result, aggregate decision
        S_NEXT_NODE   = 4'd7,    // advance to next buffered node or finish
        S_OUTPUT      = 4'd8,    // trigger decision_logic
        S_WAIT_SEND   = 4'd9,    // wait for result_sender to finish
        S_DONE        = 4'd10;   // prediction complete, LED on

    reg [3:0] state;

    // ========================================================================
    // NODE BUFFER
    // ------------------------------------------------------------------------
    // During tree traversal, tree_traversal pulses tree_node_valid for each
    // matched node. We capture those node indices here so we can process them
    // one-by-one through af_engine AFTER traversal finishes.
    //
    // WHY BUFFER? tree_traversal and af_engine both need the sensor read port.
    // They can't run simultaneously. So: traverse first (buffer matches),
    // then process each match through af_engine.
    // ========================================================================
    reg [7:0] node_buffer [0:31];  // stores up to 32 matched node indices
    reg [4:0] node_count;          // how many nodes matched
    reg [4:0] node_process_idx;    // which buffered node we're processing now

    // ========================================================================
    // AF CARRY-FORWARD
    // ------------------------------------------------------------------------
    // When processing multiple matched nodes, the AF accumulates across nodes.
    // Node 0 (root) starts with AF = 0. After it finishes, AF_out becomes the
    // starting point (AF_init) for node 1, and so on. The final running_AF
    // is the overall AF sent to the PC.
    // ========================================================================
    reg signed [31:0] running_AF;


    // ========================================================================
    // MASTER FSM — Sequential Logic
    // ========================================================================
    always @(posedge clk) begin
        if (reset) begin
            state              <= S_IDLE;
            tree_start         <= 1'b0;
            af_start           <= 1'b0;
            trigger_decision   <= 1'b0;
            sensor_mux_sel     <= 1'b0;
            node_count         <= 5'd0;
            node_process_idx   <= 5'd0;
            running_AF         <= 32'sd0;
            af_node_idx        <= 8'd0;
            af_init_value      <= 32'sd0;
            master_decision    <= DEC_HEALTHY;
            master_alarm_class <= 4'd0;
        end
        else begin
            // ================================================================
            // DEFAULT DE-ASSERTION
            // ----------------------------------------------------------------
            // These one-shot pulses are cleared to 0 every cycle by default.
            // They only become 1 in the specific state that needs them.
            // This is the "one-shot pulse pattern" — it guarantees the pulse
            // is exactly 1 clock cycle wide without needing an explicit
            // "turn it off" state.
            // ================================================================
            tree_start       <= 1'b0;
            af_start         <= 1'b0;
            trigger_decision <= 1'b0;

            case (state)

                // ============================================================
                // S_IDLE: Wait for all sensor data to arrive via UART.
                // sensor_load_complete pulses for 1 cycle when all 558 bytes
                // (279 features x 2 bytes each) have been received and stored.
                // ============================================================
                S_IDLE: begin
                    if (sensor_load_complete) begin
                        // Reset everything for a fresh prediction
                        node_count         <= 5'd0;           // empty the buffer
                        master_decision    <= DEC_HEALTHY;     // innocent until proven guilty
                        master_alarm_class <= 4'd0;
                        running_AF         <= 32'sd0;          // AF starts at zero
                        sensor_mux_sel     <= 1'b0;            // tree controls sensor port
                        state              <= S_START_TREE;
                    end
                end

                // ============================================================
                // S_START_TREE: Pulse tree_start for exactly 1 cycle.
                // Next cycle, the default de-assertion turns it off.
                // tree_traversal sees the pulse and begins scanning.
                // ============================================================
                S_START_TREE: begin
                    tree_start <= 1'b1;
                    state      <= S_SCAN_TREE;
                end

                // ============================================================
                // S_SCAN_TREE: Tree traversal is running in the background.
                // Each time it finds a matching node, tree_node_valid pulses
                // for 1 cycle and tree_active_node holds the node index.
                // We capture each match into our buffer.
                //
                // When tree_all_done pulses, all 215 nodes have been checked.
                //
                // NOTE: tree_node_valid and tree_all_done can pulse on the
                // SAME cycle (if the last node matches). Both if-blocks are
                // checked independently — this is correct because Verilog
                // evaluates both conditions, and non-blocking assignments
                // don't conflict (they write different registers).
                // ============================================================
                S_SCAN_TREE: begin
                    if (tree_node_valid) begin
                        node_buffer[node_count] <= tree_active_node;
                        node_count              <= node_count + 5'd1;
                    end
                    if (tree_all_done) begin
                        state <= S_CHECK_NODES;
                    end
                end

                // ============================================================
                // S_CHECK_NODES: Traversal is done. If we have matched nodes
                // (root always matches, so node_count >= 1), start processing.
                // Switch sensor_mux_sel so af_engine can read features.
                // ============================================================
                S_CHECK_NODES: begin
                    if (node_count > 5'd0) begin
                        node_process_idx <= 5'd0;    // start with first match
                        sensor_mux_sel   <= 1'b1;    // af_engine controls sensor
                        state            <= S_START_AF;
                    end
                    else begin
                        // No matches (shouldn't happen — root always matches)
                        master_decision <= DEC_SCREENING;
                        state           <= S_OUTPUT;
                    end
                end

                // ============================================================
                // S_START_AF: Set up af_engine's inputs and pulse start.
                //
                //   af_node_idx:    which tree node to analyze (from buffer)
                //   af_init_value:  carry-forward AF from previous node
                //                   (0 for the very first node)
                //
                // af_engine will then internally iterate all 12 diseases
                // for this node, doing BRAM reads, multiplies, divides, etc.
                // ============================================================
                S_START_AF: begin
                    af_node_idx   <= node_buffer[node_process_idx];
                    af_init_value <= running_AF;
                    af_start      <= 1'b1;   // one-cycle pulse
                    state         <= S_WAIT_AF;
                end

                // ============================================================
                // S_WAIT_AF: Spin here while af_engine processes all 12
                // diseases. This takes many clock cycles (hundreds or more).
                // ============================================================
                S_WAIT_AF: begin
                    if (af_done) begin
                        state <= S_CHECK_AF;
                    end
                end

                // ============================================================
                // S_CHECK_AF: af_engine is done. Read its result.
                //
                // DECISION AGGREGATION LOGIC:
                //   - UNHEALTHY from ANY node → immediate alarm, skip the rest
                //   - SCREENING from any node → upgrade decision if currently HEALTHY
                //   - HEALTHY → keep whatever decision we already have
                //
                // Always capture AF_out into running_AF for carry-forward.
                // The non-blocking assignment (<=) means running_AF updates
                // at the END of this cycle, so it's ready for the next node.
                // ============================================================
                S_CHECK_AF: begin
                    running_AF <= af_AF_out;    // always capture latest AF

                    if (af_decision == DEC_UNHEALTHY) begin
                        // ALARM: this patient has an unhealthy condition.
                        // Capture which disease class triggered it.
                        // Skip remaining nodes — go straight to output.
                        master_decision    <= DEC_UNHEALTHY;
                        master_alarm_class <= af_alarm_class;
                        state              <= S_OUTPUT;
                    end
                    else begin
                        // Not unhealthy — but maybe screening?
                        if (af_decision == DEC_SCREENING)
                            master_decision <= DEC_SCREENING;
                        // Either way, check if there are more nodes to process
                        state <= S_NEXT_NODE;
                    end
                end

                // ============================================================
                // S_NEXT_NODE: More buffered nodes to process?
                //   Yes → increment index, loop back to S_START_AF
                //   No  → all nodes done, go to output
                //
                // NOTE: "node_count - 1" is safe here because we only reach
                // this state if node_count >= 1 (checked in S_CHECK_NODES).
                // ============================================================
                S_NEXT_NODE: begin
                    if (node_process_idx < node_count - 5'd1) begin
                        node_process_idx <= node_process_idx + 5'd1;
                        state            <= S_START_AF;
                    end
                    else begin
                        state <= S_OUTPUT;
                    end
                end

                // ============================================================
                // S_OUTPUT: Pulse trigger_decision to tell decision_logic
                // to latch master_decision and master_alarm_class.
                //
                // TIMING CHAIN (automatic, no extra states needed):
                //   This cycle:      trigger_decision = 1 (pulse)
                //   +1 cycle:        decision_logic detects rising edge, latches
                //   +2 cycles:       decision_logic pulses prediction_complete
                //   +2 cycles:       result_sender sees send_start, begins sending
                //   +many cycles:    result_sender finishes, pulses sender_done
                //
                // We go straight to S_WAIT_SEND because the chain happens
                // automatically through the wiring. No manual "send" state needed.
                // ============================================================
                S_OUTPUT: begin
                    trigger_decision <= 1'b1;   // one-cycle pulse
                    state            <= S_WAIT_SEND;
                end

                // ============================================================
                // S_WAIT_SEND: Result sender is transmitting 5 bytes over UART.
                // At 115200 baud, each byte takes ~87 us, so total ~434 us.
                // We just wait here until sender_done pulses.
                // ============================================================
                S_WAIT_SEND: begin
                    if (sender_done) begin
                        state <= S_DONE;
                    end
                end

                // ============================================================
                // S_DONE: Prediction complete! led_done lights up (see assign
                // above). Next cycle we return to S_IDLE to wait for the next
                // patient's sensor data.
                // ============================================================
                S_DONE: begin
                    state <= S_IDLE;
                end

                default: state <= S_IDLE;

            endcase
        end
    end

endmodule
