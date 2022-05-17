`timescale 1ns / 1ps
//------------------------------------------------------------------------
// resistance_read.v
// Author: Weier Wan
//
//
//
//------------------------------------------------------------------------

module resistance_read #(parameter num_adc = 2)
	(
	// General input
	input wire clk, //6.25MHz
	input wire rst,

	// End-Point (slave) interface
	input wire trigger,
	input wire ack,
	input wire [15:0] t_shld,
	input wire [15:0] t_delta,
	output reg ready,
	output reg [18*num_adc - 1 : 0] dout1,
	output reg [18*num_adc - 1 : 0] dout2,

	// Neurram (master) interface
	output reg vin_on,

	// MUX (master) interface
	output reg sel_opamp_in, // 0 selects the resistor to be measured, 1 selects the opamp output.

	// ADC (master) interface
	output wire adc_sck, //6.25MHz
	output wire convst,
	input wire adc_sdo
	);

// Internal Signals
reg adc_trigger, adc_ack;
wire adc_ready;
wire [18*num_adc - 1 : 0] adc_dout;

adc_daisy_control #(.num_adc(num_adc)) adc_daisy(
	.clk(clk),
	.rst(rst),
	.trigger(adc_trigger),
	.ack(adc_ack),
	.ready(adc_ready),
	.dout(adc_dout),
	.adc_sck(adc_sck),
	.convst(convst),
	.adc_sdo(adc_sdo)
	);


//FSM

reg [3:0] state, next_state;
reg [20:0] counter;
reg [18*num_adc - 1 : 0] next_dout1, next_dout2;


parameter [3:0] STATE_IDLE = 4'b0000;
parameter [3:0] STATE_VIN = 4'b0001;
parameter [3:0] STATE_ADC_TRIG0 = 4'b0010;
parameter [3:0] STATE_ADC_WAIT0 = 4'b0011;
parameter [3:0] STATE_ADC_READ0 = 4'b0100;
parameter [3:0] STATE_INTEG = 4'b0101;
parameter [3:0] STATE_ADC_TRIG1 = 4'b0110;
parameter [3:0] STATE_ADC_WAIT1 = 4'b0111;
parameter [3:0] STATE_ADC_READ1 = 4'b1000;
parameter [3:0] STATE_HOLD = 4'b1001;
parameter [3:0] STATE_OPAMP = 4'b1010;

// Counter
always @(posedge rst, posedge clk) begin
	if (rst || state == STATE_IDLE) begin
		counter <= 0;
	end else if (state >= STATE_VIN && state <= STATE_ADC_TRIG1) begin
		counter <= counter + 1;
	end
end

always @(posedge rst, posedge clk) begin
	if (rst) begin
		state <= STATE_IDLE;
		dout1 <= 0;
		dout2 <= 0;
	end else begin
		state <= next_state;
		dout1 <= next_dout1;
		dout2 <= next_dout2;
	end
end

always @(*) begin
	case(state)
		STATE_IDLE: begin
			ready = 0;
			vin_on = 0;
			sel_opamp_in = 1;
			adc_trigger = 0;
			adc_ack = 0;
			next_dout1 = 0;
			next_dout2 = 0;

			if (trigger) next_state = STATE_OPAMP;
			else next_state = STATE_IDLE;
		end
		STATE_OPAMP: begin
			ready = 0;
			vin_on = 0;
			sel_opamp_in = 0;
			adc_trigger = 0;
			adc_ack = 0;
			next_dout1 = 0;
			next_dout2 = 0;

			next_state = STATE_VIN;
		end
		STATE_VIN: begin
			ready = 0;
			vin_on = 1;
			sel_opamp_in = 0;
			adc_trigger = 0;
			adc_ack = 0;
			next_dout1 = 0;
			next_dout2 = 0;

			if (counter[20:4] >= t_shld - 1) next_state = STATE_ADC_TRIG0;
			else next_state = STATE_VIN;
		end
		STATE_ADC_TRIG0: begin
			ready = 0;
			vin_on = 1;
			sel_opamp_in = 0;
			adc_trigger = 1;
			adc_ack = 0;
			next_dout1 = 0;
			next_dout2 = 0;

			next_state = STATE_ADC_WAIT0;
		end
		STATE_ADC_WAIT0: begin
			ready = 0;
			vin_on = 1;
			sel_opamp_in = 0;
			adc_trigger = 0;
			adc_ack = 0;
			next_dout2 = 0;

			if (adc_ready) begin
				next_state = STATE_ADC_READ0;
				next_dout1 = adc_dout;
			end else begin
				next_state = STATE_ADC_WAIT0;
				next_dout1 = 0;
			end
		end
		STATE_ADC_READ0: begin
			ready = 0;
			vin_on = 1;
			sel_opamp_in = 0;
			adc_trigger = 0;
			adc_ack = 1;
			next_dout1 = dout1;
			next_dout2 = 0;

			next_state = STATE_INTEG;
		end
		STATE_INTEG: begin
			ready = 0;
			vin_on = 1;
			sel_opamp_in = 0;
			adc_trigger = 0;
			adc_ack = 0;
			next_dout1 = dout1;
			next_dout2 = 0;

			if (counter[20:4] == t_shld + t_delta - 1) next_state = STATE_ADC_TRIG1;
			else if (counter[20:4] > t_shld + t_delta - 1) next_state = STATE_IDLE;
			else next_state = STATE_INTEG;
		end
		STATE_ADC_TRIG1: begin
			ready = 0;
			vin_on = 1;
			sel_opamp_in = 0;
			adc_trigger = 1;
			adc_ack = 0;
			next_dout1 = dout1;
			next_dout2 = 0;

			next_state = STATE_ADC_WAIT1;
		end
		STATE_ADC_WAIT1: begin
			ready = 0;
			vin_on = 1;
			sel_opamp_in = 0;
			adc_trigger = 0;
			adc_ack = 0;
			next_dout1 = dout1;

			if (adc_ready) begin
				next_state = STATE_ADC_READ1;
				next_dout2 = adc_dout;
			end else begin
				next_state = STATE_ADC_WAIT1;
				next_dout2 = 0;
			end
		end
		STATE_ADC_READ1: begin
			ready = 0;
			vin_on = 0;
			sel_opamp_in = 1;
			adc_trigger = 0;
			adc_ack = 1;
			next_dout1 = dout1;
			next_dout2 = dout2;

			next_state = STATE_HOLD;
		end
		STATE_HOLD: begin
			ready = 1;
			vin_on = 0;
			sel_opamp_in = 1;
			adc_trigger = 0;
			adc_ack = 0;
			next_dout1 = dout1;
			next_dout2 = dout2;

			if (ack) next_state = STATE_IDLE;
			else next_state = STATE_HOLD;
		end
		default: begin
			ready = 0;
			vin_on = 0;
			sel_opamp_in = 1;
			adc_trigger = 0;
			adc_ack = 0;
			next_dout1 = 0;
			next_dout2 = 0;
			next_state = STATE_IDLE;
		end
	endcase
end

endmodule