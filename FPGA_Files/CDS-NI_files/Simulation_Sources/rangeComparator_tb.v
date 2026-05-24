`timescale 1ns / 1ps

module rangeComparator_tb();
    reg signed [15:0] raw_value;
    reg signed [15:0] b_min;
    reg signed [15:0] b_max;
    reg valid;
    wire triggered;

    // DUT instantiation
    rangeComparator dut (.raw_value(raw_value), .b_min(b_min), .b_max(b_max), .valid(valid), .triggered(triggered) );

    //start testing
    initial begin
        // Test case 1: raw_value within the range
        raw_value = 16'sh0000; // 0
        b_min = 16'shFFFE; // -2
        b_max = 16'sh0002; // 2
        valid = 1'b1;
        #10;
        // %0d printed signed decimal values
        $display("raw=%0d min=%0d max=%0d valid=%b triggered=%b",
          raw_value,
          b_min,
          b_max,
          valid,
          triggered);
          
        // Test case 2: raw_value below the range
        raw_value = 16'shFFFD; // -3
        b_min = 16'shFFFE; // -2
        b_max = 16'sh0002; // 2
        valid = 1'b1;
        #10;
        $display("raw=%0d min=%0d max=%0d valid=%b triggered=%b",
          raw_value,
          b_min,
          b_max,
          valid,
          triggered);
          
        // Test case 3: raw_value above the range
        raw_value = 16'sh0003; // 3
        b_min = 16'shFFFE; // -2
        b_max = 16'sh0002; // 2
        valid = 1'b1;
        #10;
        $display("raw=%0d min=%0d max=%0d valid=%b triggered=%b",
          raw_value,
          b_min,
          b_max,
          valid,
          triggered);
        
        // Test case 4: raw_value is invalid
        raw_value = 16'sh0001; // 1
        b_min = 16'shFFFE; // -2
        b_max = 16'sh0002; // 2
        valid = 1'b0;
        #10;
        $display("raw=%0d min=%0d max=%0d valid=%b triggered=%b",
          raw_value,
          b_min,
          b_max,
          valid,
          triggered);
          
    end
endmodule
