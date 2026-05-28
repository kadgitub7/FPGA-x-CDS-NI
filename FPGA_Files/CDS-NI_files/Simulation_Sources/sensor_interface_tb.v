`timescale 1ns / 1ps

module sensor_interface_tb();
    reg clk,enableA,enableB,writeEnable;
    reg [8:0] addra,addrb;
    reg [15:0] dataInA;
    wire [15:0] dataOutB;
    
    sensor_interface dut(.clk(clk), .enableA(enableA),.enableB(enableB), .writeEnable(writeEnable), .addra(addra), .addrb(addrb), .dataInA(dataInA), .dataOutB(dataOutB));
    
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end
    
    initial begin
        enableA = 0;
        enableB = 0;
        writeEnable = 0;
        
        addra = 0;
        addrb = 0;
        dataInA = 0;
        
        // write data to position 0
        @(posedge clk);
        enableA = 1;
        writeEnable = 1;
        
        addra = 0;
        dataInA = 16'd1;
        
        // write data to position 1
        @(posedge clk);
        addra = 1;
        dataInA = 16'd2;
        
        //write data to position 2
        @(posedge clk);
        addra = 2;
        dataInA = 16'd3;
        
        // stop writing
        @(posedge clk);
        enableA = 0;
        writeEnable = 0;
        
        enableB = 1;
        addrb = 0;
        
        @(posedge clk);
        $display("in position 0 %b", dataOutB);
        
        addrb = 2;

        @(posedge clk);

        $display("Read addr 2 = %d", dataOutB);

        $finish;
    end    
endmodule
