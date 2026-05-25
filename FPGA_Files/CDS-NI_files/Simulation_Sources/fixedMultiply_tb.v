`timescale 1ns / 1ps

module fixedMultiply_tb();

    reg signed [15:0] a;
    reg signed [15:0] b;
    reg valid;
    wire signed [31:0] product;
    wire result_valid;

    // DUT(Design Under Test) implementation
    fixedMultiply dut (.a(a), .b(b), .valid(valid), .product(product), .result_valid(result_valid));

    // testing parameters
    initial begin
        // Test case 1: valid multiplication
        a = 16'sh0001; // 1
        b = 16'sh0002; // 2
        valid = 1'b1;
        #10;
        $display("a=%0d b=%0d valid=%b product=%0d result_valid=%b",
          a,
          b,
          valid,
          product,
          result_valid);
          
        // Test case 2: invalid multiplication
        a = 16'sh0003; // 3
        b = 16'sh0004; // 4
        valid = 1'b0;
        #10;
        $display("a=%0d b=%0d valid=%b product=%0d result_valid=%b",
          a,
          b,
          valid,
          product,
          result_valid);

        // Test case 3: negative numbers multiplication
        a = 16'shFFFE; // -2
        b = 16'shFFFD; // -3
        valid = 1'b1;
        #10;
        $display("a=%0d b=%0d valid=%b product=%0d result_valid=%b",
          a,
          b,
          valid,
          product,
          result_valid);

        // Test case 4: mixed sign multiplication
        a = 16'shFFFE; // -2
        b = 16'sh0002; // 2
        valid = 1'b1;
        #10;
        $display("a=%0d b=%0d valid=%b product=%0d result_valid=%b",
          a,
          b,
          valid,
          product,
          result_valid);
        
        // Test case 5: zero multiplication
        a = 16'sh0000; // 0
        b = 16'sh0005; // 5
        valid = 1'b1;
        #10;
        $display("a=%0d b=%0d valid=%b product=%0d result_valid=%b",
          a,
          b,
          valid,
          product,
          result_valid);
        
        $finish;

    end

endmodule