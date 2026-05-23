`timescale 1ns / 1ps
module blinkLED(
    input wire clk,
    input wire reset,
    output reg led
);
    reg [25:0] counter; // counter to add delay to LED blinking

    always @(posedge clk or posedge reset) begin
        if (reset) begin
            counter <= 0; // This is to rest the counter to 0
            led <= 0; // This is to turn off the LED when reset is asserted
        end else if (counter == 26'd49999999) begin // This is to set the counter to 0 after it reaches 50 million, which will make the LED blink at a rate of 1Hz
            counter <= 0;
            led <= ~led; // toggle the LED state
        end else begin
            counter <= counter + 1; // increment the counter if not reset
        end
    end

endmodule