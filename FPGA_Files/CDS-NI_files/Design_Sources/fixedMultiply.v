`timescale 1ns / 1ps

module fixedMultiply(
    input wire signed [15:0] a,
    input wire signed [15:0] b,
    input wire valid,
    output wire signed [31:0] product,
    output wire result_valid
);
    //inputs are in the form Q s 1.15
    //output is in the form Q s 2.30
    // * sign takes care of all multiplication as long as signed bits are used and noted
    assign product = valid ? (a * b) : 32'sh00000000; // If valid is 0 then product is 0
    assign result_valid = valid; // Output valid is the same as input valid

endmodule