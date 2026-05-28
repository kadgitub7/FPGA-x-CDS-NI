module sensor_interface(
    input clk,enableB,
    input [8:0] addrb,
    input [7:0] uart_byte,
    input uart_byte_valid,
    input reset,
    output reg load_complete,
    output reg [15:0] dataOutB
);
    
    reg [15:0] feature_mem [0:278]; // 279 features 16 bits wide
    reg [9:0] counter;
    reg [7:0] inputHigh;
    
    localparam [1:0]
        S_IDLE = 2'd0,
        S_LOAD = 2'd1,
        S_DONE = 2'd2;
    
    reg [1:0] state;

    always @(posedge clk) begin
        if (enableB)
            dataOutB <= feature_mem[addrb];
    end
    
    always @(posedge clk) begin
        if(reset) begin
            counter <= 0;
            load_complete <= 0;
            inputHigh <= 0;
            state <= S_IDLE;  // go straight to waiting for bytes
        end
        else begin
            case(state)
                S_IDLE: begin
                    load_complete <= 0;
                    if(uart_byte == 8'hAA && uart_byte_valid == 1) begin
                        state <= S_LOAD;
                        counter <= 0;
                    end
                    else begin
                        state <= S_IDLE;
                    end
                end
                S_LOAD: begin
                    if (uart_byte_valid) begin
                        if(counter[0] == 0) begin
                            inputHigh <= uart_byte;
                        end
                        else begin
                            feature_mem[counter >> 1] <= {inputHigh, uart_byte};
                        end
                        counter <= counter + 1;
    
                        // check if this was the last byte (557)
                        if(counter == 10'd557) begin
                            state <= S_DONE;
                        end
                    end
                end
    
                S_DONE: begin
                    load_complete <= 1;
                    state <= S_IDLE;
                end
                
                default: state <= S_IDLE;
                
            endcase
        end
    end
endmodule
    