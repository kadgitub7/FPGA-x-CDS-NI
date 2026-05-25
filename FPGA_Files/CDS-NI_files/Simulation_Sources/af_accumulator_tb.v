`timescale 1ns / 1ps

module af_accumulator_tb();

    reg clk;
    reg reset;

    reg signed [31:0] delta_AF;
    reg delta_valid;
    reg clear;

    wire signed [31:0] AF_real;
    wire signed [31:0] rw_real;

    af_accumulator dut(
        .clk(clk),
        .reset(reset),
        .delta_AF(delta_AF),
        .delta_valid(delta_valid),
        .clear(clear),
        .AF_real(AF_real),
        .rw_real(rw_real)
    );

    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    initial begin
        reset = 1;
        delta_AF = 0;
        delta_valid = 0;
        clear = 0;

        repeat (2) @(posedge clk);
        reset = 0;

        // -------------------
        // Test 1: +0.5
        // -------------------
        delta_AF = 32'sh20000000;
        delta_valid = 1;

        @(posedge clk);
        delta_valid = 0;

        $display("AF=%h rw=%h", AF_real, rw_real);

        // -------------------
        // Test 2: -0.25
        // -------------------
        delta_AF = -32'sh10000000;
        delta_valid = 1;

        @(posedge clk);
        delta_valid = 0;

        $display("AF=%h rw=%h", AF_real, rw_real);

        // -------------------
        // Test 3: clear
        // -------------------
        clear = 1;

        @(posedge clk);
        clear = 0;

        $display("AF=%h rw=%h", AF_real, rw_real);

        $finish;
    end

endmodule