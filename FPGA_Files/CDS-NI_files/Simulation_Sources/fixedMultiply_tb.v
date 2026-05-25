`timescale 1ns / 1ps

module fixedMultiply_tb();

    reg clk;
    reg reset;

    reg signed [15:0] a;
    reg signed [15:0] b;
    reg valid;

    wire signed [31:0] product;
    wire result_valid;


    fixedMultiply dut(
        .clk(clk),
        .reset(reset),
        .a(a),
        .b(b),
        .valid(valid),
        .product(product),
        .result_valid(result_valid)
    );


    // Clock generation
    initial begin
        clk=0;
        forever #5 clk=~clk;
    end


    // Reset
    initial begin
        reset=1;

        #12;
        reset=0;
    end


    initial begin
        a = 0;
        b = 0;
        valid = 0;

        @(negedge reset);

        // 0.5 × 0.5

        a=16'sh4000;
        b=16'sh4000;
        valid=1;

        @(posedge clk);
        #3;

        $display(
        "product=%h valid=%b",
        product,
        result_valid
        );


        // -0.5 × -0.5

        a=16'shC000;
        b=16'shC000;

        @(posedge clk);
        #3;

        $display(
        "product=%h valid=%b",
        product,
        result_valid
        );


        // -0.5 × 0.5

        a=16'shC000;
        b=16'sh4000;

        @(posedge clk);
        #3;

        $display(
        "product=%h valid=%b",
        product,
        result_valid
        );

        $finish;

    end

endmodule