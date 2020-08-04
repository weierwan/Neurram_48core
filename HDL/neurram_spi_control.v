`timescale 1ns / 1ps
//------------------------------------------------------------------------
// neurram_spi_control.v
// Author: Weier Wan
//
//
//
//------------------------------------------------------------------------

module neurram_spi_control #(parameter spi_length = 256)
	(
	input wire clk,
	input wire ok_clk,
	input wire rst,

	input wire spi_trigger,
	input wire [1:0] spi_config, // [write read]
	input wire [3:0] shift_multiplier,
	input wire [3:0] pipe_in_steps,
	input wire [3:0] pipe_out_steps,
	output reg spi_idle,

	input wire [31:0] pipe_in,
	input wire in_fifo_wr_en,
	output wire in_fifo_full,
	
	output wire [31:0] pipe_out,
	input wire out_fifo_rd_en,
	output wire out_fifo_empty,
	output wire out_fifo_valid,

	output reg spi_clk,
	output wire [1:0] shift_out,
	input wire [1:0] shift_in,

	input wire record_spi,
	output wire [spi_length*3/2-1:0] spi_from_neurram
	);

wire in_fifo_empty, in_fifo_valid;
reg in_fifo_rd_en;
wire [31:0] in_fifo_dout;

FIFO32x1024FWFT FIFO_pipe_in(

	// General input
	.wr_clk(ok_clk), // input wr_clk
	.rd_clk(clk), // input rd_clk
	.rst(rst), // input rst

	// End-Point Pipe interface
	.full(in_fifo_full), // output full
	.wr_en(in_fifo_wr_en), // input wr_en
	.wr_ack(), // output wr_ack
	.din(pipe_in), // input [31 : 0] din
	
	// FSM interface
	.empty(in_fifo_empty), // output empty
	.rd_en(in_fifo_rd_en), // input rd_en
	.dout(in_fifo_dout), // output [31 : 0] dout	
	.valid(in_fifo_valid) // output valid
);

wire out_fifo_full;
reg out_fifo_wr_en;
reg [31:0] out_fifo_din;

FIFO32x256 FIFO_pipe_out(

	// General input
	.wr_clk(clk), // input wr_clk
	.rd_clk(ok_clk), // input rd_clk
	.rst(rst), // input rst

	// FSM interface
	.full(out_fifo_full), // output full
	.wr_en(out_fifo_wr_en), // input wr_en
	.wr_ack(), // output wr_ack
	.din(out_fifo_din), // input [31 : 0] din
	
	// End-Point Pipe interface
	.empty(out_fifo_empty), // output empty
	.rd_en(out_fifo_rd_en), // input rd_en
	.dout(pipe_out), // output [31 : 0] dout	
	.valid(out_fifo_valid) // output valid
);


reg [2:0] state, next_state;
reg [11:0] clk_counter, next_clk_counter;
reg [4:0] pip_counter, next_pip_counter;
reg [3:0] pipe_in_counter, next_pipe_in_counter;
reg [3:0] pipe_out_counter, next_pipe_out_counter;
reg [spi_length-1:0] shift_pip2spi[1:0], next_shift_pip2spi[1:0];
reg [spi_length-1:0] shift_spi2pip[1:0], next_shift_spi2pip[1:0];
reg [spi_length*5/4-1:0] from_spi2pip, next_from_spi2pip;
wire [31:0] spi2pip_options [15:0];
reg next_spi_idle;


genvar step, i;
for (step=0; step<16; step=step+1) begin: spi_outer
	for (i=0; i<16; i=i+1) begin: spi_inner
		assign spi2pip_options[step][2*i] = shift_spi2pip[0][step*16+i];
		assign spi2pip_options[step][2*i+1] = shift_spi2pip[1][step*16+i];
	end
end


assign shift_out[0] = shift_pip2spi[0][0];
assign shift_out[1] = shift_pip2spi[1][0];
assign spi_from_neurram[spi_length*3/2-1 -: spi_length/4] = shift_spi2pip[0][spi_length/4-1 : 0];
assign spi_from_neurram[0 +: 5*spi_length/4] = from_spi2pip;

parameter [2:0] STATE_IDLE = 3'b000;
parameter [2:0] STATE_PIPE_IN = 3'b001;
parameter [2:0] STATE_SHIFT = 3'b011;
parameter [2:0] STATE_SHIFT2 = 3'b010;
parameter [2:0] STATE_PIPE_OUT = 3'b110;
// parameter [2:0] STATE_DONE = 3'b100;

integer index;

always @(posedge clk, posedge rst) begin
	if (rst) begin
		state <= STATE_IDLE;
		clk_counter <= 0;
		pip_counter <= 0;
		pipe_in_counter <= 0;
		pipe_out_counter <= 0;
		shift_pip2spi[0] <= 0;
		shift_pip2spi[1] <= 0;
		shift_spi2pip[0] <= 0;
		shift_spi2pip[1] <= 0;
		from_spi2pip <= 0;
		spi_idle <= 0;
	end else begin
		state <= next_state;
		clk_counter <= next_clk_counter;
		pip_counter <= next_pip_counter;
		pipe_in_counter <= next_pipe_in_counter;
		pipe_out_counter <= next_pipe_out_counter;
		shift_pip2spi[0] <= next_shift_pip2spi[0];
		shift_pip2spi[1] <= next_shift_pip2spi[1];
		shift_spi2pip[0] <= next_shift_spi2pip[0];
		shift_spi2pip[1] <= next_shift_spi2pip[1];
		from_spi2pip <= next_from_spi2pip;
		spi_idle <= next_spi_idle;
	end
end

always @(*) begin
	case (state)
		STATE_IDLE: begin
			spi_clk = 0;
			next_spi_idle = 1;
			in_fifo_rd_en = 0;
			out_fifo_wr_en = 0;
			out_fifo_din = 0;
			next_clk_counter = 0;
			next_pip_counter = 0;
			next_pipe_in_counter = 0;
			next_pipe_out_counter = 0;
			next_shift_pip2spi[0] = shift_pip2spi[0];
			next_shift_pip2spi[1] = shift_pip2spi[1];
			next_shift_spi2pip[0] = shift_spi2pip[0];
			next_shift_spi2pip[1] = shift_spi2pip[1];
			next_from_spi2pip = from_spi2pip;

			if (spi_trigger) begin
				if (spi_config[1]) next_state = STATE_PIPE_IN;
				else next_state = STATE_SHIFT;
			end
			else next_state = STATE_IDLE;
		end
		STATE_PIPE_IN: begin
			spi_clk = 0;
			next_spi_idle = 0;
			in_fifo_rd_en = 1;
			out_fifo_wr_en = 0;
			out_fifo_din = 0;
			next_clk_counter = 0;
			next_pipe_in_counter = pipe_in_counter;
			next_pipe_out_counter = pipe_out_counter;
			next_shift_pip2spi[0] = shift_pip2spi[0];
			next_shift_pip2spi[1] = shift_pip2spi[1];
			next_shift_spi2pip[0] = shift_spi2pip[0];
			next_shift_spi2pip[1] = shift_spi2pip[1];
			next_from_spi2pip = from_spi2pip;

			if (in_fifo_valid) begin
				next_pip_counter = pip_counter + 1;
				for (index=0; index<16; index=index+1) begin
					next_shift_pip2spi[0][pip_counter*16+index] = in_fifo_dout[2*index];
					next_shift_pip2spi[1][pip_counter*16+index] = in_fifo_dout[2*index+1];
				end
				if (pip_counter == spi_length/16 - 1) begin 
					next_state = STATE_SHIFT;
					next_pipe_in_counter = pipe_in_counter + 1;
				end else begin
					next_state = STATE_PIPE_IN;
				end
			end else begin
				next_state = STATE_PIPE_IN;
				next_pip_counter = pip_counter;
			end
		end
		STATE_SHIFT: begin
			spi_clk = 0;
			next_spi_idle = 0;
			in_fifo_rd_en = 0;
			out_fifo_wr_en = 0;
			out_fifo_din = 0;
			next_clk_counter = clk_counter;
			next_pip_counter = 0;
			next_pipe_in_counter = pipe_in_counter;
			next_pipe_out_counter = pipe_out_counter;
			next_shift_pip2spi[0] = shift_pip2spi[0];
			next_shift_pip2spi[1] = shift_pip2spi[1];
			next_shift_spi2pip[0][spi_length-1] = shift_in[0];
			next_shift_spi2pip[0][spi_length-2:0] = shift_spi2pip[0][spi_length-1:1];
			next_shift_spi2pip[1][spi_length-1] = shift_in[1];
			next_shift_spi2pip[1][spi_length-2:0] = shift_spi2pip[1][spi_length-1:1];
			if (record_spi && (clk_counter[8:6] == 3'b100)) begin
				next_from_spi2pip[spi_length*5/4-1] = shift_spi2pip[0][0];
				next_from_spi2pip[spi_length*5/4-2 : 0] = from_spi2pip[spi_length*5/4-1 : 1];
			end else next_from_spi2pip = from_spi2pip;
			next_state = STATE_SHIFT2;
		end
		STATE_SHIFT2: begin
			spi_clk = 1;
			next_spi_idle = 0;
			in_fifo_rd_en = 0;
			out_fifo_wr_en = 0;
			out_fifo_din = 0;
			next_clk_counter = clk_counter + 1;
			next_pip_counter = 0;
			next_pipe_in_counter = pipe_in_counter;
			next_pipe_out_counter = pipe_out_counter;
			next_shift_pip2spi[0] = shift_pip2spi[0] >> 1;
			next_shift_pip2spi[1] = shift_pip2spi[1] >> 1;
			next_shift_spi2pip[0] = shift_spi2pip[0];
			next_shift_spi2pip[1] = shift_spi2pip[1];
			next_from_spi2pip = from_spi2pip;

			if (clk_counter == spi_length * shift_multiplier -1) begin
				if (spi_config[1] && (pipe_in_counter < pipe_in_steps)) next_state = STATE_PIPE_IN;
				else if (spi_config[0]) next_state = STATE_PIPE_OUT;
				else next_state = STATE_IDLE;
			end
			else next_state = STATE_SHIFT;
		end
		STATE_PIPE_OUT: begin
			spi_clk = 0;
			next_spi_idle = 0;
			in_fifo_rd_en = 0;
			out_fifo_wr_en = 1;
			next_clk_counter = 0;
			next_pip_counter = pip_counter + 1;
			next_pipe_in_counter = pipe_in_counter;
			next_pipe_out_counter = pipe_out_counter;
			next_shift_pip2spi[0] = shift_pip2spi[0];
			next_shift_pip2spi[1] = shift_pip2spi[1];
			next_shift_spi2pip[0] = shift_spi2pip[0];
			next_shift_spi2pip[1] = shift_spi2pip[1];
			next_from_spi2pip = from_spi2pip;
			out_fifo_din = spi2pip_options[pip_counter];
			if (pip_counter == spi_length/16 - 1) begin
				next_pipe_out_counter = pipe_out_counter + 1;
				if (pipe_out_counter == pipe_out_steps - 1) next_state = STATE_IDLE;
				else next_state = STATE_SHIFT;
			end else begin
				next_state = STATE_PIPE_OUT;
			end
		end
		default: begin
			spi_clk = 0;
			next_spi_idle = 0;
			in_fifo_rd_en = 0;
			out_fifo_wr_en = 0;
			out_fifo_din = 0;
			next_clk_counter = 0;
			next_pip_counter = 0;
			next_pipe_in_counter = 0;
			next_pipe_out_counter = 0;
			next_shift_pip2spi[0] = 0;
			next_shift_pip2spi[1] = 0;
			next_shift_spi2pip[0] = 0;
			next_shift_spi2pip[1] = 0;
			next_from_spi2pip = 0;
			next_state = STATE_IDLE;
		end
	endcase
end

endmodule

