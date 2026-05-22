module blickLED(
    input wire clk,
    input wire reset,
    output led
);
    reg [23:0] counter; // counter to add delay to LED blinking

    always @(posedge clk or posedge reset) begin
        if (reset) begin
            counter <= 0; // This is to rest the counter to 0
        else begin
            counter <= counter + 1; // increment the counter if not reset
        end
        end
    end

    assign led = counter[23]; // The LED will blick at a rate determine by the 24th bit of the counter

endmodule