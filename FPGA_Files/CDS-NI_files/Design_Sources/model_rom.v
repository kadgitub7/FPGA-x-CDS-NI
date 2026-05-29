`timescale 1ns / 1ps

module model_rom(
    input wire clk,

    // tree_traversal interface
    input wire [9:0]   tree_read_addr,
    input wire         tree_re,
    output wire [15:0] tree_data,
    output wire        tree_valid,

    // af_engine address inputs
    input wire [12:0]  action_hdr_addr,
    input wire [13:0]  action_data_addr,
    input wire [11:0]  prob_phf_addr,
    input wire [7:0]   prob_pgt1_addr,
    input wire [16:0]  hr_read_addr,

    // af_engine data outputs
    output wire [15:0]        action_hdr_data,
    output wire [15:0]        action_data_out,
    output wire signed [15:0] prob_phf_data,
    output wire signed [15:0] prob_pgt1_data,
    output wire signed [15:0] hr_bmin,
    output wire signed [15:0] hr_bmax
);

    wire [31:0] hr_data_wide;
    assign hr_bmin = hr_data_wide[31:16];
    assign hr_bmax = hr_data_wide[15:0];

    bram_rom #(.ADDR_W(10), .DATA_W(16), .DEPTH(645),
               .MEM_FILE("tree_topology.mem"))
    u_tree (.clk(clk), .addr(tree_read_addr), .re(tree_re),
            .data(tree_data), .valid(tree_valid));

    bram_rom #(.ADDR_W(13), .DATA_W(16), .DEPTH(5160),
               .MEM_FILE("action_hdr.mem"))
    u_ahdr (.clk(clk), .addr(action_hdr_addr), .re(1'b1),
            .data(action_hdr_data), .valid());

    bram_rom #(.ADDR_W(14), .DATA_W(16), .DEPTH(14688),
               .MEM_FILE("action_data.mem"))
    u_adata (.clk(clk), .addr(action_data_addr), .re(1'b1),
             .data(action_data_out), .valid());

    bram_rom #(.ADDR_W(12), .DATA_W(16), .DEPTH(2580),
               .MEM_FILE("prob_phf.mem"))
    u_phf (.clk(clk), .addr(prob_phf_addr), .re(1'b1),
           .data(prob_phf_data), .valid());

    bram_rom #(.ADDR_W(8), .DATA_W(16), .DEPTH(215),
               .MEM_FILE("prob_pgt1.mem"))
    u_pgt1 (.clk(clk), .addr(prob_pgt1_addr), .re(1'b1),
            .data(prob_pgt1_data), .valid());

    bram_rom #(.ADDR_W(17), .DATA_W(32), .DEPTH(131072),
               .MEM_FILE("healthy_ranges.mem"))
    u_hr (.clk(clk), .addr(hr_read_addr), .re(1'b1),
          .data(hr_data_wide), .valid());

endmodule


module bram_rom #(
    parameter ADDR_W   = 10,
    parameter DATA_W   = 16,
    parameter DEPTH    = 645,
    parameter MEM_FILE = "mem.hex"
)(
    input  wire                clk,
    input  wire [ADDR_W-1:0]  addr,
    input  wire                re,
    output reg  [DATA_W-1:0]  data,
    output reg                 valid
);
    reg [DATA_W-1:0] mem [0:DEPTH-1];

    initial $readmemh(MEM_FILE, mem);

    always @(posedge clk) begin
        valid <= re;
        if (re)
            data <= mem[addr];
    end
endmodule
