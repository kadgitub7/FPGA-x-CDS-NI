// ============================================================================
// result_sender.v - Sends 5-byte result packet over UART
// ============================================================================
//
// Packet format (5 bytes, MSB first):
//   Byte 0: {2'b00, alarm_class[3:0], decision[1:0]}
//   Byte 1: AF[31:24]  (most significant byte)
//   Byte 2: AF[23:16]
//   Byte 3: AF[15:8]
//   Byte 4: AF[7:0]    (least significant byte)
//
// IMPORTANT PATTERN - "wait for busy, then wait for not-busy":
//   After we pulse tx_start to begin a byte, we must NOT immediately
//   check !tx_busy - uart_tx takes 1 clock cycle to assert tx_busy
//   (r_Tx_Active) via non-blocking assignment. If we check too early,
//   tx_busy is still 0 and we'd RACE to the next byte.
//
//   FIX: Each WAIT state uses a "tx_started" flag.
//     Phase 1: Wait for tx_busy to go HIGH  (byte actually started)
//     Phase 2: Wait for tx_busy to go LOW   (byte finished)
//   This guarantees we never skip ahead.
// ============================================================================

module result_sender(
    input clk, reset, send_start,
    input [1:0] final_decision,
    input [3:0] final_alarm_class,
    input [31:0] af_final,
    input tx_busy,

    output reg [7:0] tx_data,
    output reg tx_start,
    output reg done
);
    localparam [3:0]
        S_IDLE = 4'd0,
        S_SEND_DECISION = 4'd1,
        S_WAIT_1 = 4'd2,
        S_SEND_AF3 = 4'd3,
        S_WAIT_2 = 4'd4,
        S_SEND_AF2 = 4'd5,
        S_WAIT_3 = 4'd6,
        S_SEND_AF1 = 4'd7,
        S_WAIT_4 = 4'd8,
        S_SEND_AF0 = 4'd9,
        S_WAIT_5 = 4'd10,
        S_DONE = 4'd11;

    reg [3:0] state;

    reg [7:0] latched_decision_byte;
    reg [31:0] latched_af;

    // ── tx_started flag ──────────────────────────────────────────────
    // Set to 0 by each SEND state (right before pulsing tx_start).
    // Set to 1 by each WAIT state once it sees tx_busy go HIGH.
    // The WAIT state only checks !tx_busy AFTER tx_started is 1.
    // This prevents the race condition where we check !tx_busy before
    // uart_tx has had time to assert it.
    // ─────────────────────────────────────────────────────────────────
    reg tx_started;

    always @(posedge clk) begin
        if (reset) begin
            state <= S_IDLE;
            tx_start <= 1'b0;
            tx_data <= 8'd0;
            done <= 1'b0;
            tx_started <= 1'b0;
        end
        else begin
            tx_start <= 1'b0;   // one-shot pulse: default OFF
            done     <= 1'b0;   // one-shot pulse: default OFF

            case(state)

                // ── Idle: wait for decision_logic to trigger us ──────
                S_IDLE: begin
                    if (send_start) begin
                        latched_decision_byte <= {2'b00, final_alarm_class, final_decision};
                        latched_af <= af_final;
                        state <= S_SEND_DECISION;
                    end
                end

                // ── Byte 0: decision byte ────────────────────────────
                S_SEND_DECISION: begin
                    tx_data <= latched_decision_byte;
                    tx_start <= 1'b1;       // pulse for 1 cycle
                    tx_started <= 1'b0;     // reset flag before waiting
                    state <= S_WAIT_1;
                end
                S_WAIT_1: begin
                    if (!tx_started) begin
                        // Phase 1: wait for uart_tx to START (tx_busy=1)
                        if (tx_busy) tx_started <= 1'b1;
                    end
                    else if (!tx_busy) begin
                        // Phase 2: byte finished transmitting
                        state <= S_SEND_AF3;
                    end
                end

                // ── Byte 1: AF[31:24] ────────────────────────────────
                S_SEND_AF3: begin
                    tx_data <= latched_af[31:24];
                    tx_start <= 1'b1;
                    tx_started <= 1'b0;
                    state <= S_WAIT_2;
                end
                S_WAIT_2: begin
                    if (!tx_started) begin
                        if (tx_busy) tx_started <= 1'b1;
                    end
                    else if (!tx_busy) begin
                        state <= S_SEND_AF2;
                    end
                end

                // ── Byte 2: AF[23:16] ────────────────────────────────
                S_SEND_AF2: begin
                    tx_data <= latched_af[23:16];
                    tx_start <= 1'b1;
                    tx_started <= 1'b0;
                    state <= S_WAIT_3;
                end
                S_WAIT_3: begin
                    if (!tx_started) begin
                        if (tx_busy) tx_started <= 1'b1;
                    end
                    else if (!tx_busy) begin
                        state <= S_SEND_AF1;
                    end
                end

                // ── Byte 3: AF[15:8] ─────────────────────────────────
                S_SEND_AF1: begin
                    tx_data <= latched_af[15:8];
                    tx_start <= 1'b1;
                    tx_started <= 1'b0;
                    state <= S_WAIT_4;
                end
                S_WAIT_4: begin
                    if (!tx_started) begin
                        if (tx_busy) tx_started <= 1'b1;
                    end
                    else if (!tx_busy) begin
                        state <= S_SEND_AF0;
                    end
                end

                // ── Byte 4: AF[7:0] ──────────────────────────────────
                S_SEND_AF0: begin
                    tx_data <= latched_af[7:0];
                    tx_start <= 1'b1;
                    tx_started <= 1'b0;
                    state <= S_WAIT_5;
                end
                S_WAIT_5: begin
                    if (!tx_started) begin
                        if (tx_busy) tx_started <= 1'b1;
                    end
                    else if (!tx_busy) begin
                        state <= S_DONE;
                    end
                end

                // ── Done: pulse done for 1 cycle, return to idle ─────
                S_DONE: begin
                    done <= 1'b1;
                    state <= S_IDLE;
                end

            endcase
        end
    end
endmodule
