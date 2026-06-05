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
    input wire [15:0]  hr_read_addr,     // compact: node_idx*279 + feat_idx (16 bits)

    // af_engine data outputs
    output wire [15:0]        action_hdr_data,
    output wire [15:0]        action_data_out,
    output wire signed [15:0] prob_phf_data,
    output wire signed [15:0] prob_pgt1_data,
    output wire signed [15:0] hr_bmin,
    output wire signed [15:0] hr_bmax
);

    // healthy range lookup uses two BRAM's. This is done to save space
    // mulitple healthy ranges are redundant so this system is used

    wire [15:0] hr_pair_id;       // output of level-1 index BRAM
    wire [31:0] hr_data_wide;     // output of level-2 pair BRAM

    assign hr_bmin = hr_data_wide[31:16];
    assign hr_bmax = hr_data_wide[15:0];

    // Level 1: index table - maps compact address to pair_id
    bram_rom #(.ADDR_W(16), .DATA_W(16), .DEPTH(59985),
               .MEM_FILE("hr_index.mem"))
    u_hr_idx (.clk(clk), .addr(hr_read_addr), .re(1'b1),
              .data(hr_pair_id), .valid());

    // Level 2: pair table - maps pair_id to {bmin, bmax}
    // pair_id from level 1 feeds directly into level 2 address.
    // Since bram_rom registers its output, this creates a 2-cycle pipeline:
    //   cycle N:   hr_read_addr presented
    //   cycle N+1: hr_pair_id available (level 1 output)
    //   cycle N+2: hr_data_wide available (level 2 output)
    bram_rom #(.ADDR_W(12), .DATA_W(32), .DEPTH(4096),
               .MEM_FILE("hr_pairs.mem"))
    u_hr_pair (.clk(clk), .addr(hr_pair_id[11:0]), .re(1'b1),
               .data(hr_data_wide), .valid());

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
