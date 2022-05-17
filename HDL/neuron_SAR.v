`timescale 1ns / 1ps
//------------------------------------------------------------------------
// neurram_multi_level_output.v
// Author: Weier Wan
//
//
//
//------------------------------------------------------------------------
// (* fsm_style = "bram" *)
module neuron_SAR
	(
	input wire clk,
	input wire ok_clk,
	input wire dac_clk, // 100MHz
	input wire rst,

	// host interface
	input wire output_trigger,
	input wire [2:0] num_bits,
	input wire [15:0] vreset_init,
	input wire [15:0] vref,
	input wire [15:0] t_dac_settle,
	input wire ext_dac_trigger,
	input wire ext_dac_mode,

	output reg idle,
	
	output wire [31:0] pipe_out,
	input wire out_fifo_rd_en,
	output wire out_fifo_empty,
	output wire out_fifo_valid,

	// Neurram control module interface
	input wire neuron_idle,
	input wire spi_valid,
	input wire [255:0] spi_input,
	input wire dac_fifo_wr_done,
	input wire dac_idle,
	output reg spi_read_trigger,
	output reg neuron_reset_trigger,
	output reg turn_off_inference,
	output reg ext_inference_enable,
	output reg reg_reset,
	output reg [31:0] dac_word,
	output reg dac_fifo_wr_en,
	output reg dac_update,
	output reg [3:0] state,
	output reg [3:0] state_dac

	);

localparam DAC_CMD_NO_OP = 4'b1111;
localparam DAC_CMD_UPDATE = 4'b0011;
localparam DAC_CHANNEL_PLUS = 4'b0100;
localparam DAC_CHANNEL_MINUS = 4'b0010;

wire out_fifo_full;
reg out_fifo_wr_en;
reg [255:0] out_fifo_din, next_out_fifo_din;

FIFO256x64_32x512 FIFO_pipe_out(

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


reg [3:0] next_state;
reg [15:0] clk_counter, next_clk_counter;
reg [3:0] pip_counter, next_pip_counter;
reg [255:0] sign_bit, next_sign_bit;
reg [2:0] iter_number, next_iter_number;
reg [7*256-1:0] results, next_results;
reg [15:0] vreset, next_vreset;
wire [15:0] vreset_plus, vreset_minus;
wire [255:0] out_fifo_din_single_core [7:0];
reg dac_update_trigger;

assign vreset_plus = ext_dac_mode? vref + vreset_init : vref + vreset;
assign vreset_minus = ext_dac_mode? vref - vreset_init : vref - vreset;

genvar k, i, j, b;
for (k=0; k<8; k=k+1) begin: options_sc
	for (i=0; i<8; i=i+1) begin: words_sc
		for (j=0; j<4; j=j+1) begin: bytes_sc
			for (b=0; b<7; b=b+1) begin: bit_sc
				assign out_fifo_din_single_core[k][(7-i)*32 + j*8 + b] = results[b*256 + k*32 + i*4 + j];
			end
			assign out_fifo_din_single_core[k][(7-i)*32 + j*8 + 7] = sign_bit[k*32 + i*4 + j];
		end
	end
end


parameter [3:0] STATE_IDLE = 4'd0;
parameter [3:0] STATE_EXT_INF_ON_0 = 4'd1;
parameter [3:0] STATE_INF_MODE_OFF = 4'd2;
parameter [3:0] STATE_EXT_INF_OFF_0 = 4'd3;
parameter [3:0] STATE_DAC = 4'd4;
parameter [3:0] STATE_WAIT_DAC_SETTLE = 4'd6;
parameter [3:0] STATE_NEURON_TRIG = 4'd7;
parameter [3:0] STATE_NEURON_WAIT = 4'd8;
parameter [3:0] STATE_SPI_TRIG = 4'd9;
parameter [3:0] STATE_SPI_WAIT = 4'd10;
parameter [3:0] STATE_UPDATE_RESULTS = 4'd11;
parameter [3:0] STATE_PIPE_OUT = 4'd12;
parameter [3:0] STATE_EXT_INF_ON_1 = 4'd13;
parameter [3:0] STATE_INF_MODE_ON = 4'd14;
parameter [3:0] STATE_EXT_INF_OFF_1 = 4'd15;



always @(posedge clk) begin
	if (rst) begin
		state <= STATE_IDLE;
		idle <= 0;
		spi_read_trigger <= 0;
		neuron_reset_trigger <= 0;
		out_fifo_wr_en <= 0;
		turn_off_inference <= 0;
		ext_inference_enable <= 0;
		reg_reset <= 0;
		clk_counter <= 0;
		pip_counter <= 0;
		sign_bit <= 0;
		iter_number <= 0;
		out_fifo_din <= 0;
		results <= 0;
		dac_update_trigger <= 0;
		vreset <= 0;
	end else begin
		state <= next_state;
		clk_counter <= next_clk_counter;
		pip_counter <= next_pip_counter;
		sign_bit <= next_sign_bit;
		iter_number <= next_iter_number;
		out_fifo_din <= next_out_fifo_din;
		results <= next_results;
		vreset <= next_vreset;
		case (state)
			STATE_IDLE: begin
				idle <= 1;
				spi_read_trigger <= 0;
				neuron_reset_trigger <= 0;
				out_fifo_wr_en <= 0;
				turn_off_inference <= 0;
				ext_inference_enable <= 0;
				reg_reset <= 0;
				dac_update_trigger <= 0;
			end
			STATE_EXT_INF_ON_0: begin
				idle <= 0;
				spi_read_trigger <= 0;
				neuron_reset_trigger <= 0;
				out_fifo_wr_en <= 0;
				turn_off_inference <= 0;
				ext_inference_enable <= 1;
				reg_reset <= 0;
				dac_update_trigger <= 0;
			end
			STATE_INF_MODE_OFF: begin
				idle <= 0;
				spi_read_trigger <= 0;
				neuron_reset_trigger <= 0;
				out_fifo_wr_en <= 0;
				turn_off_inference <= 1;
				ext_inference_enable <= 1;
				reg_reset <= 0;
				dac_update_trigger <= 0;
			end
			STATE_EXT_INF_OFF_0: begin
				idle <= 0;
				spi_read_trigger <= 0;
				neuron_reset_trigger <= 0;
				out_fifo_wr_en <= 0;
				turn_off_inference <= 1;
				ext_inference_enable <= 0;
				reg_reset <= 0;
				dac_update_trigger <= 0;
			end
			STATE_DAC: begin
				idle <= 0;
				spi_read_trigger <= 0;
				neuron_reset_trigger <= 0;
				out_fifo_wr_en <= 0;
				turn_off_inference <= 1;
				ext_inference_enable <= 0;
				reg_reset <= 0;
				dac_update_trigger <= 1;
			end
			STATE_WAIT_DAC_SETTLE: begin
				idle <= 0;
				spi_read_trigger <= 0;
				neuron_reset_trigger <= 0;
				out_fifo_wr_en <= 0;
				turn_off_inference <= 1;
				ext_inference_enable <= 0;
				reg_reset <= 0;
				dac_update_trigger <= 0;
			end
			STATE_NEURON_TRIG: begin
				idle <= 0;
				spi_read_trigger <= 0;
				neuron_reset_trigger <= 1;
				out_fifo_wr_en <= 0;
				turn_off_inference <= 1;
				ext_inference_enable <= 0;
				reg_reset <= 0;
				dac_update_trigger <= 0;
			end
			STATE_NEURON_WAIT: begin
				idle <= 0;
				spi_read_trigger <= 0;
				neuron_reset_trigger <= 0;
				out_fifo_wr_en <= 0;
				turn_off_inference <= 1;
				ext_inference_enable <= 0;
				reg_reset <= 0;
				dac_update_trigger <= 0;
			end
			STATE_SPI_TRIG: begin
				idle <= 0;
				spi_read_trigger <= 1;
				neuron_reset_trigger <= 0;
				out_fifo_wr_en <= 0;
				turn_off_inference <= 1;
				ext_inference_enable <= 0;
				reg_reset <= 0;
				dac_update_trigger <= 0;
			end
			STATE_SPI_WAIT: begin
				idle <= 0;
				spi_read_trigger <= 0;
				neuron_reset_trigger <= 0;
				out_fifo_wr_en <= 0;
				turn_off_inference <= 1;
				ext_inference_enable <= 0;
				reg_reset <= 0;
				dac_update_trigger <= 0;
			end
			STATE_UPDATE_RESULTS: begin
				idle <= 0;
				spi_read_trigger <= 0;
				neuron_reset_trigger <= 0;
				out_fifo_wr_en <= 0;
				turn_off_inference <= 1;
				ext_inference_enable <= 0;
				reg_reset <= 1;
				dac_update_trigger <= 0;
			end
			STATE_PIPE_OUT: begin
				idle <= 0;
				spi_read_trigger <= 0;
				neuron_reset_trigger <= 0;
				out_fifo_wr_en <= 1;
				turn_off_inference <= 1;
				ext_inference_enable <= 0;
				reg_reset <= 0;
				dac_update_trigger <= 0;
			end
			STATE_EXT_INF_ON_1: begin
				idle <= 0;
				spi_read_trigger <= 0;
				neuron_reset_trigger <= 0;
				out_fifo_wr_en <= 0;
				turn_off_inference <= 1;
				ext_inference_enable <= 1;
				reg_reset <= 0;
				dac_update_trigger <= 0;
			end
			STATE_INF_MODE_ON: begin
				idle <= 0;
				spi_read_trigger <= 0;
				neuron_reset_trigger <= 0;
				out_fifo_wr_en <= 0;
				turn_off_inference <= 0;
				ext_inference_enable <= 1;
				reg_reset <= 0;
				dac_update_trigger <= 0;
			end
			STATE_EXT_INF_OFF_1: begin
				idle <= 0;
				spi_read_trigger <= 0;
				neuron_reset_trigger <= 0;
				out_fifo_wr_en <= 0;
				turn_off_inference <= 0;
				ext_inference_enable <= 0;
				reg_reset <= 0;
				dac_update_trigger <= 0;
			end
			default: begin
				idle <= 0;
				spi_read_trigger <= 0;
				neuron_reset_trigger <= 0;
				out_fifo_wr_en <= 0;
				turn_off_inference <= 0;
				ext_inference_enable <= 0;
				reg_reset <= 0;
				dac_update_trigger <= 0;
			end
		endcase
	end
end

always @(*) begin
	case (state)
		STATE_IDLE: begin
			next_clk_counter = 0;
			next_pip_counter = 0;
			next_sign_bit = 0;
			next_iter_number = 0;
			next_out_fifo_din = 0;
			next_results = 0;
			next_vreset = 0;

			if (output_trigger) next_state = STATE_EXT_INF_ON_0;
			else next_state = STATE_IDLE;
		end
		STATE_EXT_INF_ON_0: begin
			next_clk_counter = 0;
			next_pip_counter = 0;
			next_sign_bit = 0;
			next_iter_number = 0;
			next_out_fifo_din = 0;
			next_results = 0;
			next_vreset = vreset_init;

			next_state = STATE_INF_MODE_OFF;
		end
		STATE_INF_MODE_OFF: begin
			next_clk_counter = 0;
			next_pip_counter = 0;
			next_sign_bit = 0;
			next_iter_number = 0;
			next_out_fifo_din = 0;
			next_results = 0;
			next_vreset = vreset;

			next_state = STATE_EXT_INF_OFF_0;
		end
		STATE_EXT_INF_OFF_0: begin
			next_clk_counter = 0;
			next_pip_counter = 0;
			next_sign_bit = 0;
			next_iter_number = 0;
			next_out_fifo_din = 0;
			next_results = 0;
			next_vreset = vreset;

			next_state = STATE_DAC;
		end
		STATE_DAC: begin
			next_clk_counter = 0;
			next_pip_counter = 0;
			next_sign_bit = sign_bit;
			next_iter_number = iter_number;
			next_out_fifo_din = 0;
			next_results = results;
			next_vreset = vreset;
			next_state = STATE_WAIT_DAC_SETTLE;
		end
		STATE_WAIT_DAC_SETTLE: begin
			next_clk_counter = clk_counter + 1;
			next_pip_counter = 0;
			next_sign_bit = sign_bit;
			next_iter_number = iter_number;
			next_out_fifo_din = 0;
			next_results = results;
			next_vreset = vreset;
			if (clk_counter == t_dac_settle - 1) begin
				next_clk_counter = 0;
				next_state = STATE_NEURON_TRIG;
			end else begin
				next_state = STATE_WAIT_DAC_SETTLE;
			end
		end
		STATE_NEURON_TRIG: begin
			next_clk_counter = clk_counter + 1;
			next_pip_counter = 0;
			next_sign_bit = sign_bit;
			next_iter_number = iter_number;
			next_out_fifo_din = 0;
			next_results = results;
			next_vreset = vreset;

			if (clk_counter == 3) next_state = STATE_NEURON_WAIT;
			else next_state = STATE_NEURON_TRIG;
		end
		STATE_NEURON_WAIT: begin
			next_clk_counter = 0;
			next_pip_counter = 0;
			next_sign_bit = sign_bit;
			next_iter_number = iter_number;
			next_out_fifo_din = 0;
			next_results = results;
			next_vreset = vreset;

			if (neuron_idle) next_state = STATE_SPI_TRIG;
			else next_state = STATE_NEURON_WAIT;
		end
		STATE_SPI_TRIG: begin
			next_clk_counter = clk_counter + 1;
			next_pip_counter = 0;
			next_sign_bit = sign_bit;
			next_iter_number = iter_number;
			next_out_fifo_din = 0;
			next_results = results;
			next_vreset = vreset;

			if (clk_counter == 3) next_state = STATE_SPI_WAIT;
			else next_state = STATE_SPI_TRIG;
		end
		STATE_SPI_WAIT: begin
			next_clk_counter = 0;
			next_pip_counter = 0;
			next_sign_bit = sign_bit;
			next_iter_number = iter_number;
			next_out_fifo_din = 0;
			next_results = results;
			next_vreset = vreset;

			if (spi_valid) next_state = STATE_UPDATE_RESULTS;
			else next_state = STATE_SPI_WAIT;
		end
		STATE_UPDATE_RESULTS: begin
			next_clk_counter = 0;
			next_pip_counter = 0;
			next_sign_bit = sign_bit;
			next_iter_number = iter_number + 1;
			next_out_fifo_din = 0;
			next_results = results;
			next_vreset = vreset >> 1;
			if (iter_number == 0) begin
				next_sign_bit = spi_input;
				next_results = 0;
			end else begin
				next_results[(num_bits - iter_number - 1) * 256 +: 256] = ~(sign_bit ^ spi_input);
			end
			if (iter_number == num_bits - 1) begin
				next_state = STATE_PIPE_OUT;
			end else begin
				next_state = STATE_DAC;
			end
		end
		STATE_PIPE_OUT: begin
			next_clk_counter = 0;
			next_pip_counter = pip_counter + 1;
			next_sign_bit = sign_bit;
			next_iter_number = iter_number;
			next_results = results;
			next_vreset = vreset;

			next_out_fifo_din = out_fifo_din_single_core[pip_counter];
			if (pip_counter == 7) begin 
				next_state = STATE_EXT_INF_ON_1;
			end else begin
				next_state = STATE_PIPE_OUT;
			end
		end
		STATE_EXT_INF_ON_1: begin
			next_clk_counter = 0;
			next_pip_counter = 0;
			next_sign_bit = sign_bit;
			next_iter_number = iter_number;
			next_out_fifo_din = 0;
			next_results = results;
			next_vreset = vreset;

			next_state = STATE_INF_MODE_ON;
		end
		STATE_INF_MODE_ON: begin
			next_clk_counter = 0;
			next_pip_counter = 0;
			next_sign_bit = sign_bit;
			next_iter_number = iter_number;
			next_out_fifo_din = 0;
			next_results = results;
			next_vreset = vreset;

			next_state = STATE_EXT_INF_OFF_1;
		end
		STATE_EXT_INF_OFF_1: begin
			next_clk_counter = 0;
			next_pip_counter = 0;
			next_sign_bit = sign_bit;
			next_iter_number = iter_number;
			next_out_fifo_din = 0;
			next_results = results;
			next_vreset = vreset;

			next_state = STATE_IDLE;
		end
		default: begin
			next_clk_counter = 0;
			next_pip_counter = 0;
			next_sign_bit = 0;
			next_iter_number = 0;
			next_out_fifo_din = 0;
			next_results = 0;
			next_vreset = 0;

			next_state = STATE_IDLE;
		end
	endcase
end

parameter [3:0] STATE_DAC_IDLE = 4'd0;
parameter [3:0] STATE_PROGRAM_DAC_FIFO_0 = 4'd1;
parameter [3:0] STATE_PROGRAM_DAC_FIFO_0_DONE = 4'd2;
parameter [3:0] STATE_PROGRAM_DAC_FIFO_1 = 4'd3;
parameter [3:0] STATE_PROGRAM_DAC_FIFO_1_DONE = 4'd4;
parameter [3:0] STATE_PROGRAM_DAC_FIFO_2 = 4'd5;
parameter [3:0] STATE_PROGRAM_DAC_FIFO_2_DONE = 4'd6;
parameter [3:0] STATE_PROGRAM_DAC_FIFO_3 = 4'd7;
parameter [3:0] STATE_PROGRAM_DAC_FIFO_3_DONE = 4'd8;
parameter [3:0] STATE_WAIT_DAC_RD = 4'd9;
parameter [3:0] STATE_UPDATE_DAC = 4'd10;
parameter [3:0] STATE_DAC_DONE = 4'd11;

reg [3:0] next_state_dac;
reg [31:0] next_dac_word;

always @(posedge dac_clk) begin
	if (rst) begin
		state_dac <= STATE_DAC_IDLE;
		dac_word <= 0;
		dac_fifo_wr_en <= 0;
		dac_update <= 0;
	end else begin
		state_dac <= next_state_dac;
		dac_word <= next_dac_word;
		case (state_dac)
			STATE_DAC_IDLE: begin
				dac_fifo_wr_en <= 0;
				dac_update <= 0;
			end
			STATE_PROGRAM_DAC_FIFO_0: begin
				dac_fifo_wr_en <= 1;
				dac_update <= 0;
			end
			STATE_PROGRAM_DAC_FIFO_0_DONE: begin
				dac_fifo_wr_en <= 0;
				dac_update <= 0;
			end
			STATE_PROGRAM_DAC_FIFO_1: begin
				dac_fifo_wr_en <= 1;
				dac_update <= 0;
			end
			STATE_PROGRAM_DAC_FIFO_1_DONE: begin
				dac_fifo_wr_en <= 0;
				dac_update <= 0;
			end
			STATE_PROGRAM_DAC_FIFO_2: begin
				dac_fifo_wr_en <= 1;
				dac_update <= 0;
			end
			STATE_PROGRAM_DAC_FIFO_2_DONE: begin
				dac_fifo_wr_en <= 0;
				dac_update <= 0;
			end
			STATE_PROGRAM_DAC_FIFO_3: begin
				dac_fifo_wr_en <= 1;
				dac_update <= 0;
			end
			STATE_PROGRAM_DAC_FIFO_3_DONE: begin
				dac_fifo_wr_en <= 0;
				dac_update <= 0;
			end
			STATE_WAIT_DAC_RD: begin
				dac_fifo_wr_en <= 0;
				dac_update <= 0;
			end
			STATE_UPDATE_DAC: begin
				dac_fifo_wr_en <= 0;
				dac_update <= 1;
			end
			STATE_DAC_DONE: begin
				dac_fifo_wr_en <= 0;
				dac_update <= 0;
			end
			default: begin
				dac_fifo_wr_en <= 0;
				dac_update <= 0;
			end
		endcase
	end
end


always @(*) begin
	case (state_dac)
		STATE_DAC_IDLE: begin
			next_dac_word = 0;
			if (dac_update_trigger | (ext_dac_trigger & ext_dac_mode)) next_state_dac = STATE_PROGRAM_DAC_FIFO_0;
			else next_state_dac = STATE_DAC_IDLE;
		end
		STATE_PROGRAM_DAC_FIFO_0: begin
			next_dac_word = 0;
			next_dac_word[23:20] = DAC_CMD_NO_OP;
			next_state_dac = STATE_PROGRAM_DAC_FIFO_0_DONE;
		end
		STATE_PROGRAM_DAC_FIFO_0_DONE: begin
			next_dac_word = dac_word;
			if (dac_fifo_wr_done) next_state_dac = STATE_PROGRAM_DAC_FIFO_1;
			else next_state_dac = STATE_PROGRAM_DAC_FIFO_0_DONE;
		end
		STATE_PROGRAM_DAC_FIFO_1: begin
			next_dac_word = 0;
			next_dac_word[23:20] = DAC_CMD_UPDATE;
			next_dac_word[19:16] = DAC_CHANNEL_MINUS;
			next_dac_word[15:0] = vreset_minus;
			next_state_dac = STATE_PROGRAM_DAC_FIFO_1_DONE;
		end
		STATE_PROGRAM_DAC_FIFO_1_DONE: begin
			next_dac_word = dac_word;
			if (dac_fifo_wr_done) next_state_dac = STATE_PROGRAM_DAC_FIFO_2;
			else next_state_dac = STATE_PROGRAM_DAC_FIFO_1_DONE;
		end
		STATE_PROGRAM_DAC_FIFO_2: begin
			next_dac_word = 0;
			next_dac_word[23:20] = DAC_CMD_NO_OP;
			next_state_dac = STATE_PROGRAM_DAC_FIFO_2_DONE;
		end
		STATE_PROGRAM_DAC_FIFO_2_DONE: begin
			next_dac_word = dac_word;
			if (dac_fifo_wr_done) next_state_dac = STATE_PROGRAM_DAC_FIFO_3;
			else next_state_dac = STATE_PROGRAM_DAC_FIFO_2_DONE;
		end
		STATE_PROGRAM_DAC_FIFO_3: begin
			next_dac_word = 0;
			next_dac_word[23:20] = DAC_CMD_UPDATE;
			next_dac_word[19:16] = DAC_CHANNEL_PLUS;
			next_dac_word[15:0] = vreset_plus;
			next_state_dac = STATE_PROGRAM_DAC_FIFO_3_DONE;
		end
		STATE_PROGRAM_DAC_FIFO_3_DONE: begin
			next_dac_word = dac_word;
			if (dac_fifo_wr_done) next_state_dac = STATE_WAIT_DAC_RD;
			else next_state_dac = STATE_PROGRAM_DAC_FIFO_3_DONE;
		end
		STATE_WAIT_DAC_RD: begin
			next_dac_word = 0;
			if (dac_idle) next_state_dac = STATE_WAIT_DAC_RD;
			else next_state_dac = STATE_UPDATE_DAC;
		end
		STATE_UPDATE_DAC: begin
			next_dac_word = 0;
			if (dac_idle) next_state_dac = STATE_DAC_DONE;
			else next_state_dac = STATE_UPDATE_DAC;
		end
		STATE_DAC_DONE: begin
			next_dac_word = 0;
			if (dac_update_trigger) next_state_dac = STATE_DAC_DONE;
			else next_state_dac = STATE_DAC_IDLE;
		end
		default: begin
			next_dac_word = 0;
			next_state_dac = STATE_DAC_IDLE;
		end
	endcase
end


endmodule