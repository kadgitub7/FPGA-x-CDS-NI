`timescale 1ns / 1ps
module rangeComparator(
    input wire signed [15:0] raw_value,
    input wire signed [15:0] b_min,
    input wire signed [15:0] b_max,
    output wire triggered
);
    assign triggered = (raw_value == 16'hFFFF) ? 1'b0 : ((raw_value <= b_min) || (raw_value >= b_max));

endmodule