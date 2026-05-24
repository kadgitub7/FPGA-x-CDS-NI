`timescale 1ns / 1ps
module rangeComparator(
    input wire [15:0] raw_value,
    input wire [15:0] b_min,
    input wire [15:0] b_max,
    output wire triggered
);
    always @(*) begin
        if (raw_value == 16'hFFFF) // This is to check if the raw value is invalid, which is represented by 0xFFFF in our case.
            triggered = 0; // If the raw value is invalid, we do not want to trigger an alert.
        else
            triggered = (raw_value < b_min) || (raw_value > b_max);
    end
endmodule