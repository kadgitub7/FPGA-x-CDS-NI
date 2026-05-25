`timescale 1ns / 1ps

module fixedDivide(
    input clk,
    input reset,

    input wire signed [31:0] numerator,
    input wire signed [15:0] reciprocal_denominator, // This is the precomputed reciprocal of the denominator in Q2.14 format
    input wire valid,

    output reg signed [31:0] quotient,
    output reg result_valid
);
    reg signed [47:0] full_product; // Intermediate product with enough bits to hold the result of multiplication before shifting
    reg valid_stage1; // Register to hold the valid signal for the next stage

    // the division performed by this module utilizes a precomputed reciprocal to do fixed point multiplication
    // numerator is in Q2.30 format, reciprocal_denominator is in Q2.14 format, so the product is in Q4.44 format
    // we will then use a right shift to bring it back to Q2.30 format by 14 bits (since we have 14 fractional bits in the reciprocal)

    always @(posedge clk) begin
        if (reset) begin
            full_product <= 48'sd0;
            quotient <= 32'sd0;
            result_valid <= 1'b0;
            valid_stage1 <= 1'b0;
        end 
        else begin
            // stage 1, capture multiplication result
            full_product <= (numerator * reciprocal_denominator); // Multiply numerator by reciprocal of denominator without worrying about losing information
            valid_stage1 <= valid; // Register the valid signal for the next stage

            // stage 2, use previous cycle's multiplication result
            quotient <= valid_stage1 ? (full_product >>> 14) : 32'sd0; // If valid is 0 then quotient is 0
            result_valid <= valid_stage1; // Output valid is the same as input valid
        end
    end

endmodule