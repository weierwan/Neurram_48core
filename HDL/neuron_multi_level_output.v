`timescale 1ns / 1ps
//------------------------------------------------------------------------
// neurram_multi_level_output.v
// Author: Weier Wan
//
//
//
//------------------------------------------------------------------------
(* fsm_style = "bram" *)
module neuron_multi_level_output #(parameter spi_length = 768)
	(
	input wire clk,
	input wire ok_clk,
	input wire rst,

	// host interface
	input wire output_trigger,
	input wire y_addr_trigger,
	output reg idle,

	input wire [31:0] pipe_in,
	input wire in_fifo_wr_en,
	output wire in_fifo_full,
	
	output wire [31:0] pipe_out,
	input wire out_fifo_rd_en,
	output wire out_fifo_empty,
	output wire out_fifo_valid,

	// Neurram control module interface
	input wire neuron_idle,
	input wire spi_valid,
	input wire [spi_length-1:0] spi_input,
	output reg [1:0] reg_config,
	output reg spi_read_trigger,
	output reg neuron_reset_trigger
	);


wire in_fifo_empty, in_fifo_valid;
reg in_fifo_rd_en;
wire [31:0] in_fifo_dout;

FIFO32x32 FIFO_pipe_in (

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
reg [255:0] out_fifo_din, next_out_fifo_din;

FIFO256x32_32x256 FIFO_pipe_out(

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


reg [3:0] state, next_state;
reg [3:0] clk_counter, next_clk_counter;
reg [4:0] pip_counter, next_pip_counter;
reg [spi_length-1:0] init_value, next_init_value;
reg [spi_length-1:0] y_addr, next_y_addr;
reg [spi_length-1:0] fnd_idx, next_fnd_idx;
reg [spi_length-1:0] step_found, next_step_found;
reg [6:0] iter_number, next_iter_number;
reg [7*spi_length-1:0] num_step, next_num_step;
wire [255:0] out_fifo_din_options [23:0];

genvar k, i, j;
for (k=0; k<spi_length*8/256; k=k+1) begin: nmlo_outer
	for (i=0; i<8; i=i+1) begin: nmlo_middle
		for (j=0; j<4; j=j+1) begin: nmlo_inner
			assign out_fifo_din_options[k][(7-i)*32 + j*8 +: 7] = num_step[(k*32 + i*4 + j)*7 +: 7];
			assign out_fifo_din_options[k][(7-i)*32 + j*8 + 7] = init_value[k*32 + i*4 + j];
		end
	end
end


parameter [3:0] STATE_IDLE = 4'd0;
parameter [3:0] STATE_PIPE_IN = 4'd1;
parameter [3:0] STATE_NEURON_TRIG = 4'd2;
parameter [3:0] STATE_NEURON_WAIT = 4'd3;
parameter [3:0] STATE_SPI_TRIG = 4'd4;
parameter [3:0] STATE_SPI_WAIT = 4'd5;
parameter [3:0] STATE_UPDATE_FND_IDX = 4'd6;
parameter [3:0] STATE_UPDATE_NUM_STEP = 4'd7;
parameter [3:0] STATE_CHECK_FOUND = 4'd8;
parameter [3:0] STATE_PIPE_OUT = 4'd9;



always @(posedge clk) begin
	if (rst) begin
		state <= STATE_IDLE;
		idle <= 0;
		reg_config <= 2'b01;
		spi_read_trigger <= 0;
		neuron_reset_trigger <= 0;
		in_fifo_rd_en <= 0;
		out_fifo_wr_en <= 0;
		clk_counter <= 0;
		pip_counter <= 0;
		init_value <= 0;
		y_addr <= 0;
		fnd_idx <= 0;
		step_found <= 0;
		iter_number <= 0;
		out_fifo_din <= 0;
		num_step <= 0;
	end else begin
		state <= next_state;
		clk_counter <= next_clk_counter;
		pip_counter <= next_pip_counter;
		init_value <= next_init_value;
		y_addr <= next_y_addr;
		fnd_idx <= next_fnd_idx;
		step_found <= next_step_found;
		iter_number <= next_iter_number;
		out_fifo_din <= next_out_fifo_din;
		num_step <= next_num_step;
		case (state)
			STATE_IDLE: begin
				idle <= 1;
				reg_config <= 2'b00;
				spi_read_trigger <= 0;
				neuron_reset_trigger <= 0;
				in_fifo_rd_en <= 0;
				out_fifo_wr_en <= 0;
			end
			STATE_PIPE_IN: begin
				idle <= 0;
				reg_config <= 2'b00;
				spi_read_trigger <= 0;
				neuron_reset_trigger <= 0;
				in_fifo_rd_en <= 1;
				out_fifo_wr_en <= 0;
			end
			STATE_NEURON_TRIG: begin
				idle <= 0;
				reg_config <= 2'b00;
				spi_read_trigger <= 0;
				neuron_reset_trigger <= 1;
				in_fifo_rd_en <= 0;
				out_fifo_wr_en <= 0;
			end
			STATE_NEURON_WAIT: begin
				idle <= 0;
				reg_config <= 2'b00;
				spi_read_trigger <= 0;
				neuron_reset_trigger <= 0;
				in_fifo_rd_en <= 0;
				out_fifo_wr_en <= 0;
			end
			STATE_SPI_TRIG: begin
				idle <= 0;
				reg_config <= 2'b01;
				spi_read_trigger <= 1;
				neuron_reset_trigger <= 0;
				in_fifo_rd_en <= 0;
				out_fifo_wr_en <= 0;
			end
			STATE_SPI_WAIT: begin
				idle <= 0;
				reg_config <= 2'b01;
				spi_read_trigger <= 0;
				neuron_reset_trigger <= 0;
				in_fifo_rd_en <= 0;
				out_fifo_wr_en <= 0;
			end
			STATE_UPDATE_FND_IDX: begin
				idle <= 0;
				reg_config <= 2'b01;
				spi_read_trigger <= 0;
				neuron_reset_trigger <= 0;
				in_fifo_rd_en <= 0;
				out_fifo_wr_en <= 0;
			end
			STATE_UPDATE_NUM_STEP: begin
				idle <= 0;
				reg_config <= 2'b01;
				spi_read_trigger <= 0;
				neuron_reset_trigger <= 0;
				in_fifo_rd_en <= 0;
				out_fifo_wr_en <= 0;
			end
			STATE_CHECK_FOUND: begin
				idle <= 0;
				reg_config <= 2'b00;
				spi_read_trigger <= 0;
				neuron_reset_trigger <= 0;
				in_fifo_rd_en <= 0;
				out_fifo_wr_en <= 0;
			end
			STATE_PIPE_OUT: begin
				idle <= 0;
				reg_config <= 2'b00;
				spi_read_trigger <= 0;
				neuron_reset_trigger <= 0;
				in_fifo_rd_en <= 0;
				out_fifo_wr_en <= 1;
			end
			default: begin
				idle <= 0;
				reg_config <= 2'b00;
				spi_read_trigger <= 0;
				neuron_reset_trigger <= 0;
				in_fifo_rd_en <= 0;
				out_fifo_wr_en <= 0;
			end
		endcase
	end
end

integer index;
always @(*) begin
	case (state)
		STATE_IDLE: begin
			next_clk_counter = 0;
			next_pip_counter = 0;
			next_init_value = init_value;
			next_y_addr = y_addr;
			next_fnd_idx = 0;
			next_step_found = 0;
			next_iter_number = 0;
			next_out_fifo_din = 0;
			next_num_step = num_step;

			if (y_addr_trigger) next_state = STATE_PIPE_IN;
			else if (output_trigger) next_state = STATE_NEURON_TRIG;
			else next_state = STATE_IDLE;
		end
		STATE_PIPE_IN: begin
			next_clk_counter = 0;
			next_init_value = init_value;
			next_y_addr = y_addr;
			next_fnd_idx = 0;
			next_step_found = 0;
			next_iter_number = 0;
			next_out_fifo_din = 0;
			next_num_step = 0;

			if (in_fifo_valid) begin
				next_pip_counter = pip_counter + 1;
				next_y_addr[pip_counter*32 +: 32] = in_fifo_dout;
				if (pip_counter == spi_length / 32 - 1) begin 
					next_state = STATE_IDLE;
				end else begin
					next_state = STATE_PIPE_IN;
				end
			end else begin
				next_state = STATE_PIPE_IN;
				next_pip_counter = pip_counter;
			end
		end
		STATE_NEURON_TRIG: begin
			next_clk_counter = clk_counter + 1;
			next_pip_counter = 0;
			next_init_value = init_value;
			next_y_addr = y_addr;
			next_fnd_idx = fnd_idx;
			next_step_found = step_found;
			next_iter_number = iter_number;
			next_out_fifo_din = 0;
			next_num_step = num_step;

			if (clk_counter == 3) next_state = STATE_NEURON_WAIT;
			else next_state = STATE_NEURON_TRIG;
		end
		STATE_NEURON_WAIT: begin
			next_clk_counter = 0;
			next_pip_counter = 0;
			next_init_value = init_value;
			next_y_addr = y_addr;
			next_fnd_idx = fnd_idx;
			next_step_found = step_found;
			next_iter_number = iter_number;
			next_out_fifo_din = 0;
			next_num_step = num_step;

			if (neuron_idle) next_state = STATE_SPI_TRIG;
			else next_state = STATE_NEURON_WAIT;
		end
		STATE_SPI_TRIG: begin
			next_clk_counter = clk_counter + 1;
			next_pip_counter = 0;
			next_init_value = init_value;
			next_y_addr = y_addr;
			next_fnd_idx = fnd_idx;
			next_step_found = step_found;
			next_iter_number = iter_number;
			next_out_fifo_din = 0;
			next_num_step = num_step;

			if (clk_counter == 3) next_state = STATE_SPI_WAIT;
			else next_state = STATE_SPI_TRIG;
		end
		STATE_SPI_WAIT: begin
			next_clk_counter = 0;
			next_pip_counter = 0;
			next_init_value = init_value;
			next_y_addr = y_addr;
			next_fnd_idx = fnd_idx;
			next_step_found = step_found;
			next_iter_number = iter_number;
			next_out_fifo_din = 0;
			next_num_step = num_step;

			if (spi_valid) next_state = STATE_UPDATE_FND_IDX;
			else next_state = STATE_SPI_WAIT;
		end
		STATE_UPDATE_FND_IDX: begin
			next_clk_counter = 0;
			next_pip_counter = 0;
			next_init_value = (iter_number == 0) ? spi_input : init_value;
			next_y_addr = y_addr;
			next_fnd_idx = (iter_number == 0) ? 0 : ((spi_input ^ init_value) & (~step_found));
			next_step_found = (iter_number == 0)? (~y_addr) : step_found;
			next_iter_number = iter_number;
			next_out_fifo_din = 0;
			next_num_step = num_step;

			next_state = STATE_UPDATE_NUM_STEP;
		end
		STATE_UPDATE_NUM_STEP: begin
			next_clk_counter = 0;
			next_pip_counter = 0;
			next_init_value = init_value;
			next_y_addr = y_addr;
			next_fnd_idx = fnd_idx;
			next_step_found = step_found | fnd_idx;
			next_iter_number = iter_number;
			next_out_fifo_din = 0;
			for (index=0; index<spi_length; index=index+1) begin
				if (fnd_idx[index] == 1) next_num_step[index*7 +: 7] = iter_number;
				else next_num_step[index*7 +: 7] = num_step[index*7 +: 7];
			end

			next_state = STATE_CHECK_FOUND;
		end
		STATE_CHECK_FOUND: begin
			next_clk_counter = 0;
			next_pip_counter = 0;
			next_init_value = init_value;
			next_y_addr = y_addr;
			next_fnd_idx = fnd_idx;
			next_step_found = step_found;
			next_iter_number = iter_number + 1;
			next_out_fifo_din = 0;
			next_num_step = num_step;

			if ((& step_found) || (iter_number == 200)) next_state = STATE_PIPE_OUT;
			else next_state = STATE_NEURON_TRIG;
		end
		STATE_PIPE_OUT: begin
			next_clk_counter = 0;
			next_pip_counter = pip_counter + 1;
			next_init_value = init_value;
			next_y_addr = y_addr;
			next_fnd_idx = fnd_idx;
			next_step_found = step_found;
			next_iter_number = iter_number;
			next_num_step = num_step;
			next_out_fifo_din = out_fifo_din_options[pip_counter];

			if (pip_counter == spi_length*8/256-1) begin 
				next_state = STATE_IDLE;
			end else begin
				next_state = STATE_PIPE_OUT;
			end
		end
		default: begin
			next_clk_counter = 0;
			next_pip_counter = 0;
			next_init_value = 0;
			next_y_addr = 0;
			next_fnd_idx = 0;
			next_step_found = 0;
			next_iter_number = 0;
			next_out_fifo_din = 0;
			next_num_step = 0;

			next_state = STATE_IDLE;
		end
	endcase
end

endmodule