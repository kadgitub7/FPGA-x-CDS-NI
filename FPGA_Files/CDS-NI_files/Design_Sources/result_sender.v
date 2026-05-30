
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

    always @(posedge clk) begin
        if (reset) begin
            state <= S_IDLE;
            tx_start <= 1'b0;
            tx_data <= 8'd0;
            done <= 1'b0;
        end
        else begin
            tx_start <= 1'b0;
            done     <= 1'b0;
            case(state)
                S_IDLE: begin
                    if (send_start) begin
                        latched_decision_byte <= {2'b00, final_alarm_class, final_decision};
                        latched_af <= af_final;
                        state <= S_SEND_DECISION;
                    end
                end
                S_SEND_DECISION: begin
                    tx_data <= latched_decision_byte;
                    tx_start <= 1'b1;
                    state <= S_WAIT_1;
                end
                S_WAIT_1: begin
                    if (!tx_busy) begin
                        state <= S_SEND_AF3;
                    end
                end
                S_SEND_AF3: begin
                    tx_data <= latched_af[31:24];
                    tx_start <= 1'b1;
                    state <= S_WAIT_2;
                end
                S_WAIT_2: begin
                    if (!tx_busy) begin
                        state <= S_SEND_AF2;
                    end
                end
                S_SEND_AF2: begin
                    tx_data <= latched_af[23:16];
                    tx_start <= 1'b1;
                    state <= S_WAIT_3;
                end
                S_WAIT_3: begin
                    if (!tx_busy) begin
                        state <= S_SEND_AF1;
                    end
                end
                S_SEND_AF1: begin
                    tx_data <= latched_af[15:8];
                    tx_start <= 1'b1;
                    state <= S_WAIT_4;
                end
                S_WAIT_4: begin
                    if (!tx_busy) begin
                        state <= S_SEND_AF0;
                    end
                end
                S_SEND_AF0: begin
                    tx_data <= latched_af[7:0];
                    tx_start <= 1'b1;
                    state <= S_WAIT_5;
                end
                S_WAIT_5: begin
                    if (!tx_busy) begin
                        state <= S_DONE;
                    end
                end
                S_DONE: begin
                    done <= 1'b1;
                    state <= S_IDLE;
                end
            endcase
        end
    end
endmodule