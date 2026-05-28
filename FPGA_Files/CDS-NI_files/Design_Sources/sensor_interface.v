module sensor_interface(
    input clk,enableA,enableB,writeEnable,
    input [8:0] addra,addrb,
    input [15:0] dataInA,
    output reg [15:0] dataOutB
);
    
    reg [15:0] feature_mem [0:278]; // 279 features 16 bits wide

    always @(posedge clk) begin
        if (enableA && writeEnable) begin
            feature_mem[addra] <= dataInA;
        end
    end

    always @(posedge clk) begin
        if (enableB)
            dataOutB <= feature_mem[addrb];
    end

endmodule
    