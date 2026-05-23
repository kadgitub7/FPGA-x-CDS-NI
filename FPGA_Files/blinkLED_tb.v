`timescale 1ns / 1ps

module blinkLED_tb();
    reg clk;
    reg reset;
    wire led;

    // DUT instantiation
    blinkLED dut (.clk(clk), .reset(reset), .led(led) );

    //clock generation
    initial begin
        clk = 0;
        forever #10 clk = ~clk; // 50MHz clock speed
    end

    // Testing

    initial begin
    reset = 1;
    #20;

    reset = 0;
    #2000000000; //2 seconds

    reset = 1;
    #20;

    reset = 0;
    #2000000000;

    $finish;
    end
endmodule