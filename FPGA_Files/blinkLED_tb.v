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
        //initial reset
        reset = 1; # 10
        reset = 0; # 1000000000 // Run the simulation for a long time to observe multiple LED blinks

        // Assert reset again to see if the LED turns off
        reset = 1; # 10
        reset = 0; # 1000000000 // Run the simulation for a long time to observe multiple LED blinks
        $finish;
    end
endmodule