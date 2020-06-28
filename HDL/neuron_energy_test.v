module neuron_energy_test (
	input wire clk,
	input wire rst,

	input wire dp_trigger,
	input wire neuron_idle,
	input wire [31:0] cycles,
	output reg neuron_trigger,
	output reg idle_out
	);

reg idle;
reg [2:0] state, next_state;
reg [31:0] cycle_counter, next_cycle_counter;


parameter [2:0] STATE_IDLE = 3'd0;
parameter [2:0] STATE_NEURON = 3'd1;
parameter [2:0] STATE_NEURON_1 = 3'd2;


always @(posedge clk, posedge rst) begin
	if (rst) begin
		state <= STATE_IDLE;
		cycle_counter <= 0;
		idle_out <= 0;
	end else begin
		state <= next_state;
		cycle_counter <= next_cycle_counter;
		idle_out <= idle;
	end
end

always @(*) begin
	case (state)
		STATE_IDLE: begin
			neuron_trigger = 0;
			idle = 1;
			next_cycle_counter = 0;

			if (dp_trigger) next_state = STATE_NEURON;
			else next_state = STATE_IDLE;
		end
		STATE_NEURON: begin
			neuron_trigger = 1;
			idle = 0;
			next_cycle_counter = cycle_counter + 1;

			next_state = STATE_NEURON_1;
		end
		STATE_NEURON_1: begin
			neuron_trigger = 0;
			idle = 0;
			next_cycle_counter = cycle_counter;

			if (neuron_idle) begin
				if (cycle_counter == cycles) next_state = STATE_IDLE;
				else next_state = STATE_NEURON;
			end else next_state = STATE_NEURON_1;
		end
		default: begin
			neuron_trigger = 0;
			idle = 0;
			next_cycle_counter = 0;

			next_state = STATE_IDLE;
		end
	endcase
end

endmodule
