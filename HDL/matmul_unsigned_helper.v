`timescale 1ns / 1ps
//------------------------------------------------------------------------
// matmul_unsigned_helper.v
// Author: Weier Wan
//
//
//
//------------------------------------------------------------------------

module matmul_unsigned_helper(
	input wire clk,
	input wire rst,

	// host interface
	input wire trigger,
	input wire cds,
	input wire reset,
	input wire [2:0] num_bits,
	input wire [4:0] pulse_multiplier,
	output reg idle,

	// Neurram control module interface
	input wire neuron_idle,
	input wire spi_idle,
	output reg neuron_sample_trigger,
	output reg neuron_cds_trigger,
	output reg spi_write_trigger,
	output reg turn_off_inference,
	output reg ext_inference_enable,
	output reg [7:0] num_pulses,
	output reg reg_reset
	);


reg [3:0] state, next_state;
reg [7:0] next_num_pulses;
reg [3:0] cycle_counter, next_cycle_counter;
reg [3:0] clk_counter, next_clk_counter;


parameter [3:0] STATE_IDLE = 4'd0;
parameter [3:0] STATE_EXT_INF_ON = 4'd1;
parameter [3:0] STATE_INF_MODE_OFF = 4'd2;
parameter [3:0] STATE_SPI_TRIG = 4'd3;
parameter [3:0] STATE_SPI_WAIT = 4'd4;
parameter [3:0] STATE_INF_MODE_ON = 4'd5;
parameter [3:0] STATE_EXT_INF_OFF = 4'd6;
parameter [3:0] STATE_CDS_TRIG = 4'd7;
parameter [3:0] STATE_CDS_WAIT = 4'd8;
parameter [3:0] STATE_SAMPLE_TRIG = 4'd9;
parameter [3:0] STATE_SAMPLE_WAIT = 4'd10;
parameter [3:0] STATE_CYCLE_DONE = 4'd11;


always @(posedge clk, posedge rst) begin
	if (rst) begin
		state <= STATE_IDLE;
		idle <= 0;
		neuron_sample_trigger <= 0;
		neuron_cds_trigger <= 0;
		spi_write_trigger <= 0;
		turn_off_inference <= 0;
		ext_inference_enable <= 0;
		reg_reset <= 0;
		num_pulses <= 0;
		cycle_counter <= 0;
		clk_counter <= 0;
	end else begin
		state <= next_state;
		num_pulses <= next_num_pulses;
		cycle_counter <= next_cycle_counter;
		clk_counter <= next_clk_counter;
		case (state)
			STATE_IDLE: begin
				idle <= 1;
				neuron_sample_trigger <= 0;
				neuron_cds_trigger <= 0;
				spi_write_trigger <= 0;
				turn_off_inference <= 0;
				ext_inference_enable <= 0;
				reg_reset <= 0;
			end
			STATE_EXT_INF_ON: begin
				idle <= 0;
				neuron_sample_trigger <= 0;
				neuron_cds_trigger <= 0;
				spi_write_trigger <= 0;
				turn_off_inference <= 0;
				ext_inference_enable <= 1;
				reg_reset <= 0;
			end
			STATE_INF_MODE_OFF: begin
				idle <= 0;
				neuron_sample_trigger <= 0;
				neuron_cds_trigger <= 0;
				spi_write_trigger <= 0;
				turn_off_inference <= 1;
				ext_inference_enable <= 1;
				reg_reset <= 0;
			end
			STATE_SPI_TRIG: begin
				idle <= 0;
				neuron_sample_trigger <= 0;
				neuron_cds_trigger <= 0;
				spi_write_trigger <= 1;
				turn_off_inference <= 1;
				ext_inference_enable <= 1;
				reg_reset <= 0;
			end
			STATE_SPI_WAIT: begin
				idle <= 0;
				neuron_sample_trigger <= 0;
				neuron_cds_trigger <= 0;
				spi_write_trigger <= 0;
				turn_off_inference <= 1;
				ext_inference_enable <= 1;
				reg_reset <= 0;
			end
			STATE_INF_MODE_ON: begin
				idle <= 0;
				neuron_sample_trigger <= 0;
				neuron_cds_trigger <= 0;
				spi_write_trigger <= 0;
				turn_off_inference <= 0;
				ext_inference_enable <= 1;
				reg_reset <= 0;
			end
			STATE_EXT_INF_OFF: begin
				idle <= 0;
				neuron_sample_trigger <= 0;
				neuron_cds_trigger <= 0;
				spi_write_trigger <= 0;
				turn_off_inference <= 0;
				ext_inference_enable <= 0;
				reg_reset <= 0;
			end
			STATE_CDS_TRIG: begin
				idle <= 0;
				neuron_sample_trigger <= 0;
				neuron_cds_trigger <= 1;
				spi_write_trigger <= 0;
				turn_off_inference <= 0;
				ext_inference_enable <= 0;
				reg_reset <= 0;
			end
			STATE_CDS_WAIT: begin
				idle <= 0;
				neuron_sample_trigger <= 0;
				neuron_cds_trigger <= 0;
				spi_write_trigger <= 0;
				turn_off_inference <= 0;
				ext_inference_enable <= 0;
				reg_reset <= 0;
			end
			STATE_SAMPLE_TRIG: begin
				idle <= 0;
				neuron_sample_trigger <= 1;
				neuron_cds_trigger <= 0;
				spi_write_trigger <= 0;
				turn_off_inference <= 0;
				ext_inference_enable <= 0;
				reg_reset <= 0;
			end
			STATE_SAMPLE_WAIT: begin
				idle <= 0;
				neuron_sample_trigger <= 0;
				neuron_cds_trigger <= 0;
				spi_write_trigger <= 0;
				turn_off_inference <= 0;
				ext_inference_enable <= 0;
				reg_reset <= 0;
			end
			STATE_CYCLE_DONE: begin
				idle <= 0;
				neuron_sample_trigger <= 0;
				neuron_cds_trigger <= 0;
				spi_write_trigger <= 0;
				turn_off_inference <= 0;
				ext_inference_enable <= 0;
				reg_reset <= reset;
			end
			default: begin
				idle <= 0;
				neuron_sample_trigger <= 0;
				neuron_cds_trigger <= 0;
				spi_write_trigger <= 0;
				turn_off_inference <= 0;
				ext_inference_enable <= 0;
				reg_reset <= 0;
			end
		endcase
	end
end


always @(*) begin
	case (state)
		STATE_IDLE: begin
			next_num_pulses = pulse_multiplier;
			next_cycle_counter = 0;
			next_clk_counter = 0;
			if (trigger) next_state = STATE_EXT_INF_ON;
			else next_state = STATE_IDLE;
		end
		STATE_EXT_INF_ON: begin
			next_num_pulses = num_pulses;
			next_cycle_counter = cycle_counter;
			next_clk_counter = 0;
			next_state = STATE_INF_MODE_OFF;
		end
		STATE_INF_MODE_OFF: begin
			next_num_pulses = num_pulses;
			next_cycle_counter = cycle_counter;
			next_clk_counter = 0;
			if (spi_idle) next_state = STATE_SPI_TRIG;
			else next_state = STATE_INF_MODE_OFF;
		end
		STATE_SPI_TRIG: begin
			next_num_pulses = num_pulses;
			next_cycle_counter = cycle_counter;
			next_clk_counter = clk_counter + 1;
			if (clk_counter == 3) next_state = STATE_SPI_WAIT;
			else next_state = STATE_SPI_TRIG;
		end
		STATE_SPI_WAIT: begin
			next_num_pulses = num_pulses;
			next_cycle_counter = cycle_counter;
			next_clk_counter = 0;
			if (spi_idle) next_state = STATE_INF_MODE_ON;
			else next_state = STATE_SPI_WAIT;
		end
		STATE_INF_MODE_ON: begin
			next_num_pulses = num_pulses;
			next_cycle_counter = cycle_counter;
			next_clk_counter = 0;
			next_state = STATE_EXT_INF_OFF;
		end
		STATE_EXT_INF_OFF: begin
			next_num_pulses = num_pulses;
			next_cycle_counter = cycle_counter;
			next_clk_counter = 0;
			if (neuron_idle) begin
				if (cds && (cycle_counter == 0)) next_state = STATE_CDS_TRIG;
				else next_state = STATE_SAMPLE_TRIG;
			end else begin
				next_state = STATE_EXT_INF_OFF;
			end
		end
		STATE_CDS_TRIG: begin
			next_num_pulses = num_pulses;
			next_cycle_counter = cycle_counter;
			next_clk_counter = clk_counter + 1;
			if (clk_counter == 3) next_state = STATE_CDS_WAIT;
			else next_state = STATE_CDS_TRIG;
		end
		STATE_CDS_WAIT: begin
			next_num_pulses = num_pulses;
			next_cycle_counter = cycle_counter;
			next_clk_counter = 0;
			if (neuron_idle) next_state = STATE_SAMPLE_TRIG;
			else next_state = STATE_CDS_WAIT;
		end
		STATE_SAMPLE_TRIG: begin
			next_num_pulses = num_pulses;
			next_cycle_counter = cycle_counter;
			next_clk_counter = clk_counter + 1;
			if (clk_counter == 3) next_state = STATE_SAMPLE_WAIT;
			else next_state = STATE_SAMPLE_TRIG;
		end
		STATE_SAMPLE_WAIT: begin
			next_num_pulses = num_pulses;
			next_cycle_counter = cycle_counter;
			next_clk_counter = 0;
			if (neuron_idle) begin
				next_state = STATE_CYCLE_DONE;
			end else begin
				next_state = STATE_SAMPLE_WAIT;
			end
		end
		STATE_CYCLE_DONE: begin
			next_cycle_counter = cycle_counter + 1;
			next_clk_counter = 0;
			if (cycle_counter[0] == 1) begin
				next_num_pulses = num_pulses << 1;
				if (cycle_counter[3:1] == num_bits) next_state = STATE_IDLE;
				else next_state = STATE_EXT_INF_ON;
			end else begin
				next_num_pulses = num_pulses;
				next_state = STATE_EXT_INF_ON;
			end
		end
		default: begin
			next_num_pulses = 0;
			next_cycle_counter = 0;
			next_clk_counter = 0;
			next_state = STATE_IDLE;
		end
	endcase
end

endmodule