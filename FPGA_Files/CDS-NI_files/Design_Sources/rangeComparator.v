`timescale 1ns / 1ps
module rangeComparator(
    input wire signed [15:0] raw_value,
    input wire signed [15:0] b_min,
    input wire signed [15:0] b_max,
    input wire valid,
    output wire triggered
);
    // Here we check if the input is valid and if it is then we compare boundaries
    // If input is not valid then triggered is 0
    // If input is valid then triggered is 1 if raw_value is less than b_min or greater than b_max
    assign triggered = valid ? ((raw_value < b_min) || (raw_value > b_max)) : 1'b0;

endmodule