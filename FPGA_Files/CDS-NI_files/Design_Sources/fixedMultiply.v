`timescale 1ns / 1ps

module fixedMultiply(
    input clk,
    input reset,

    input wire signed [15:0] a,
    input wire signed [15:0] b,
    input wire valid,

    output reg signed [31:0] product,
    output reg result_valid
);
    //inputs are in the form Q s 1.15
    //output is in the form Q s 2.30
    // * sign takes care of all multiplication as long as signed bits are used and noted
    always @(posedge clk) begin
        if (reset) begin
            product <= 32'sd0;
            result_valid <= 1'b0;
        end 
        else begin
            product <= valid ? (a * b) : 32'sd0; // If valid is 0 then product is 0
            result_valid <= valid; // Output valid is the same as input valid
        end
    end
    
endmodule