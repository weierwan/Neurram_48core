`timescale 1ns / 1ps
//------------------------------------------------------------------------
// auto_ranging.v
// Author: Weier Wan
//
//
//
//------------------------------------------------------------------------

module auto_ranging (
	input wire clk,
	input wire rst,

	// End-Point (slave) interface
	input wire trigger,
	input wire [15:0] t_shld_init,
	input wire [15:0] t_delta_init,
	input wire [17:0] vref,	
	input wire ack,
	output reg [17:0] dout1,
	output reg [17:0] dout2,
	// output reg error_polarity,
	output reg valid,

	// resistance_read (master) interface
	output reg rr_trigger,
	output reg ack_trigger,
	output reg [15:0] t_shld,
	output reg [15:0] t_delta,
	input wire rr_ready,
	input wire [17:0] rr_dout1,
	input wire [17:0] rr_dout2
	);


reg [2:0] state, next_state;
reg [15:0] next_t_shld, next_t_delta;
reg [17:0] next_dout1, next_dout2;


parameter [2:0] STATE_IDLE = 3'd0;
parameter [2:0]	STATE_TRIG = 3'd1;
parameter [2:0]	STATE_WAIT = 3'd2;
parameter [2:0] STATE_ACK = 3'd3;
// parameter [2:0] STATE_ERR = 3'd4;
parameter [2:0] STATE_VALID = 3'd4;

always @(posedge clk, posedge rst) begin
	if (rst) begin
		state <= STATE_IDLE;
		t_shld <= 0;
		t_delta <= 0;
		dout1 <= 0;
		dout2 <= 0;
	end else begin
		state <= next_state;
		t_shld <= next_t_shld;
		t_delta <= next_t_delta;
		dout1 <= next_dout1;
		dout2 <= next_dout2;
	end
end

always @(*) begin
	case (state)
		STATE_IDLE: begin
			rr_trigger = 0;
			ack_trigger = 0;
			next_t_shld = t_shld_init;
			next_t_delta = t_delta_init;
			next_dout1 = 0;
			next_dout2 = 0;
			// error_polarity = 0;
			valid = 0;

			if (trigger) begin
				next_state = STATE_TRIG;
			end else begin
				next_state = STATE_IDLE;
			end
		end
		STATE_TRIG: begin
			rr_trigger = 1;
			ack_trigger = 0;
			next_t_shld = t_shld;
			next_t_delta = t_delta;
			next_dout1 = 0;
			next_dout2 = 0;
			// error_polarity = 0;
			valid = 0;

			next_state = STATE_WAIT;
		end
		STATE_WAIT: begin
			rr_trigger = 0;
			ack_trigger = 0;
			next_t_shld = t_shld;
			next_t_delta = t_delta;
			next_dout1 = 0;
			next_dout2 = 0;
			// error_polarity = 0;
			valid = 0;

			if (rr_ready) begin
				next_dout1 = rr_dout1;
				next_dout2 = rr_dout2;
				next_state = STATE_ACK;
			end else begin
				next_state = STATE_WAIT;
			end
		end
		STATE_ACK: begin
			rr_trigger = 0;
			ack_trigger = 1;
			next_t_shld = t_shld;
			next_t_delta = t_delta;
			next_dout1 = dout1;
			next_dout2 = dout2;
			// error_polarity = 0;
			valid = 0;

			if ((t_delta > 10) && (t_delta < 30000)) begin
				if (vref > dout2) begin
					if ((vref/3*2) < dout2) begin
						next_t_shld = t_shld << 1;
						next_t_delta = t_delta << 1;
						next_state = STATE_TRIG;
					end else if ((vref/9) > dout2) begin
						next_t_shld = t_shld >> 1;
						next_t_delta = t_delta >> 1;
						next_state = STATE_TRIG;
					end else begin
						next_state = STATE_VALID;
					end
				end else begin
					if ((vref/3*4) > dout2) begin
						next_t_shld = t_shld << 1;
						next_t_delta = t_delta << 1;
						next_state = STATE_TRIG;
					end else if ((vref/9*17) < dout2) begin
						next_t_shld = t_shld >> 1;
						next_t_delta = t_delta >> 1;
						next_state = STATE_TRIG;
					end else begin
						next_state = STATE_VALID;
					end
				end
			end else begin
				next_state = STATE_VALID;
			end
		end
		STATE_VALID: begin
			rr_trigger = 0;
			ack_trigger = 0;
			next_t_shld = t_shld;
			next_t_delta = t_delta;
			next_dout1 = dout1;
			next_dout2 = dout2;
			valid = 1;

			if (ack) next_state = STATE_IDLE;
			else next_state = STATE_VALID;
		end
		default: begin
			rr_trigger = 0;
			ack_trigger = 0;
			next_t_shld = t_shld_init;
			next_t_delta = t_delta_init;
			next_dout1 = 0;
			next_dout2 = 0;
			// error_polarity = 0;
			valid = 0;
			next_state = STATE_IDLE;
		end
	endcase
end

endmodule