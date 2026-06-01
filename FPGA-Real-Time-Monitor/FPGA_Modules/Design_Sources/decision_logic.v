module decision_logic(
    input clk,reset,
    input decision_made,
    input [3:0] alarm_class,
    input [1:0] decision,

    output reg [1:0] final_decision,
    output reg [3:0] final_alarm_class,
    output reg prediction_complete
);

    localparam [1:0]
        S_IDLE = 2'd0,
        S_OUTPUT = 2'd1;
    
    reg [1:0] state;

    reg decision_made_prev;

    always @(posedge clk) begin
        if (reset)
            decision_made_prev <= 1'b0;
        else
            decision_made_prev <= decision_made;
    end

    wire decision_made_pulse = decision_made & ~decision_made_prev;

    always @(posedge clk) begin
        if(reset) begin
            state <= S_IDLE;
            final_decision <= 2'b00;
            final_alarm_class <= 4'b0000;
            prediction_complete <= 1'b0;
        end
        else begin
            prediction_complete <= 1'b0;
            case(state)
                S_IDLE: begin
                    if(decision_made_pulse) begin
                        final_decision <= decision;
                        final_alarm_class <= alarm_class;
                        state <= S_OUTPUT;
                    end
                end

                S_OUTPUT: begin
                    prediction_complete <= 1'b1;
                    state <= S_IDLE; // wait for next decision
                end
            endcase
        end
    end
endmodule