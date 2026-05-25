`timescale 1ns / 1ps

module af_accumulator(
    input clk,
    input reset,
    input wire signed [31:0] delta_AF, // Q s2.30
    input wire delta_valid,
    input wire clear,

    output reg signed [31:0] AF_real, // Q s2.30
    output wire signed [31:0] rw_real // Q s2.30
);

    // define ONE as a constant in Q s2.30 format
    localparam signed [31:0] ONE = 32'sh40000000; // 1.0 in Q s2.30

    reg signed [32:0] sum; // Use an extra bit to detect overflow

    // Update rw_real combinationally based on the new AF_real
    assign rw_real = ONE - AF_real; // rw_real is always 1.0 - AF_real

    always @(posedge clk) begin
        if (reset || clear) begin
            AF_real <= 32'sd0;
        end
        else if (delta_valid) begin
            // Perform accumulation
            sum = {{1{AF_real[31]}}, AF_real} + {{1{delta_AF[31]}}, delta_AF};

            if (sum > 33'sh40000000) begin
                AF_real <= ONE; // Clamp to 1.0
            end
            else if (sum[32] == 1'b1) begin
                AF_real <= 32'sd0; // Clamp to 0.0
            end
            else begin
                AF_real <= sum[31:0]; // No overflow, assign the sum
            end
        end    
    end
endmodule
