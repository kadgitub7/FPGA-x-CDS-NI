`timescale 1ns / 1ps

module fixedDivide(
    input clk,
    input reset,

    input wire signed [31:0] numerator,
    input wire signed [15:0] reciprocal_denominator, // This is the precomputed reciprocal of the denominator in Q3.13 format
    input wire valid,

    output reg signed [31:0] quotient,
    output reg result_valid
);
    reg signed [47:0] full_product; // Intermediate product with enough bits to hold the result of multiplication before shifting
    reg valid_stage1; // Register to hold the valid signal for the next stage

    // the division performed by this module utilizes a precomputed reciprocal to do fixed point multiplication
    // numerator is in Q2.30 format, reciprocal_denominator is in Q3.13 format, so the product is in Q5.43 format
    // we will then use a right shift to bring it back to Q2.30 format by 13 bits (since we have 13 fractional bits in the reciprocal)

    // Shifted result (combinational, uses previous cycle's full_product due to NBA)
    wire signed [47:0] shifted = full_product >>> 13;

    // Overflow detection: the 48-bit shifted result doesn't fit in 32-bit signed
    // if bits [47:31] are NOT all copies of the sign bit (bit 31).
    wire fits_in_32 = (shifted[47:31] == {17{shifted[31]}});

    always @(posedge clk) begin
        if (reset) begin
            full_product <= 48'sd0;
            quotient <= 32'sd0;
            result_valid <= 1'b0;
            valid_stage1 <= 1'b0;
        end
        else begin
            // stage 1, capture multiplication result
            full_product <= (numerator * reciprocal_denominator);
            valid_stage1 <= valid;

            // stage 2, with saturation to prevent 32-bit overflow
            if (valid_stage1) begin
                if (fits_in_32)
                    quotient <= shifted[31:0];
                else if (shifted[47])          // negative overflow
                    quotient <= 32'sh80000000;  // saturate to min
                else                           // positive overflow
                    quotient <= 32'sh7FFFFFFF;  // saturate to max
            end
            else begin
                quotient <= 32'sd0;
            end
            result_valid <= valid_stage1;
        end
    end

endmodule