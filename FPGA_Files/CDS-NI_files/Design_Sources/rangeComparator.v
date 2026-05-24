`timescale 1ns / 1ps
module rangeComparator(
    input wire [7:0] raw_value,
    input wire [7:0] b_min,
    input wire [7:0] b_max,
    output wire triggered
);
    if (raw_value == 0xFFFF)
        assign triggered = 0; // If the raw value is invalid, we do not want to trigger an alert.
    else
        assign triggered = (raw_value < b_min) || (raw_value > b_max);

endmodule