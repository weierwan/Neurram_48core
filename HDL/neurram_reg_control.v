`timescale 1ns / 1ps
//------------------------------------------------------------------------
// neurram_reg_control.v
// Author: Weier Wan
//
//
//
//------------------------------------------------------------------------

module neurram_reg_control(
	input wire clk,
	input wire rst,

	input wire spi_trigger,
	input wire rand_access_trigger,
	input wire [1:0] neuron_read_trigger,

	input wire state_spi_clk,
	input wire state_spi_idle,

	output reg [1:0] spi_clk
	);


reg [2:0] state, next_state;

parameter [2:0] STATE_IDLE = 3'b000;
parameter [2:0] STATE_SPI_TRIG = 3'b001;
parameter [2:0] STATE_SPI = 3'b010;
parameter [2:0] STATE_RAND_ACCESS = 3'b011;
parameter [2:0] STATE_NEURON_READ0 = 3'b100;
parameter [2:0] STATE_NEURON_READ1 = 3'b101;

always @(posedge clk, posedge rst) begin
	if (rst) begin
		state <= STATE_IDLE;
	end else begin
		state <= next_state;
	end
end

always @(*) begin
	case (state)
		STATE_IDLE: begin
			spi_clk = 0;

			if (spi_trigger) next_state = STATE_SPI_TRIG;
			else if (rand_access_trigger) next_state = STATE_RAND_ACCESS;
			else if (neuron_read_trigger[0]) next_state = STATE_NEURON_READ0;
			else if (neuron_read_trigger[1]) next_state = STATE_NEURON_READ1;
			else next_state = STATE_IDLE;
		end
		STATE_SPI_TRIG: begin
			spi_clk = 0;

			if (!state_spi_idle) next_state = STATE_SPI;
			else next_state = STATE_SPI_TRIG;
		end
		STATE_SPI: begin
			spi_clk[0] = state_spi_clk;
			spi_clk[1] = state_spi_clk;

			if (state_spi_idle) next_state = STATE_IDLE;
			else next_state = STATE_SPI;			
		end
		STATE_RAND_ACCESS: begin
			spi_clk = 2'b11;
			next_state = STATE_IDLE;
		end
		STATE_NEURON_READ0: begin
			spi_clk = 2'b01;
			next_state = STATE_IDLE;
		end
		STATE_NEURON_READ1: begin
			spi_clk = 2'b10;
			next_state = STATE_IDLE;
		end
		default: begin
			spi_clk = 0;
			next_state = STATE_IDLE;
		end
	endcase
end

endmodule

