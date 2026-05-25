`timescale 1ns / 1ps

module fixedDivide_tb();
    reg clk;
    reg reset;

    reg signed [31:0] numerator;
    reg signed [15:0] reciprocal_denominator;
    reg valid;

    wire signed [31:0] quotient;
    wire result_valid;

    //dut instantiation
    fixedDivide dut(
        .clk(clk),
        .reset(reset),
        .numerator(numerator),
        .reciprocal_denominator(reciprocal_denominator),
        .valid(valid),
        .quotient(quotient),
        .result_valid(result_valid)
    );

    // Clock generation
    initial  begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    // Reset
    initial begin
        reset = 1;
        #12;
        reset = 0;
    end

    initial begin

        numerator = 0;
        reciprocal_denominator = 0;
        valid = 0;

        @(negedge reset);
        @(posedge clk);


        //----------------------------------------------------
        // TEST 1: 0.5 / 1 = 0.5
        //----------------------------------------------------
        numerator = 32'sh40000000;     // 0.5
        reciprocal_denominator = 16'sh4000; // 1.0
        valid = 1;
        
        repeat(2) @(posedge clk);
        #1;

        $display(
        "quotient=%h valid=%b",
        quotient,
        result_valid);

        //----------------------------------------------------
        // TEST 2: -0.5 / 1 = -0.5
        //----------------------------------------------------
        numerator = -32'sh40000000;
        reciprocal_denominator = 16'sh4000;
        valid = 1;

        repeat(2) @(posedge clk);
        #1;

        $display(
        "quotient=%h valid=%b",
        quotient,
        result_valid);

        //----------------------------------------------------
        // TEST 3: 0.5 / -0.5 = -1.0
        //----------------------------------------------------
        numerator = 32'sh40000000;
        reciprocal_denominator = 16'shC000; // -1.0 in Q2.14
        valid = 1;

        repeat(2) @(posedge clk);
        #1;

        $display(
        "quotient=%h valid=%b",
        quotient,
        result_valid);

        //----------------------------------------------------
        // TEST 4: -0.5 / -0.5 = 1.0
        //----------------------------------------------------
        numerator = -32'sh40000000;
        reciprocal_denominator = 16'shC000;
        valid = 1;

        repeat(2) @(posedge clk);
        #1;

        $display(
        "quotient=%h valid=%b",
         quotient,
         result_valid);
    end

endmodule