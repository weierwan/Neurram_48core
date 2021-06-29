//------------------------------------------------------------------------
// neurram_top.v
//
// 
//------------------------------------------------------------------------

`timescale 1ns / 1ps

module neurram_top(
	input  wire [4:0]   okUH,
	output wire [2:0]   okHU,
	inout  wire [31:0]  okUHU,
	inout  wire         okAA,
	input  wire         sys_clkn,
   input  wire         sys_clkp,

	// Neurram interface
	output	wire [7:0]	addr_local,
	output	wire 		clear_horz_b,
	output	wire 		clear_vert_b,
	output	wire 		dec_enable_horz,
	output	wire 		dec_enable_vert,
	output	wire 		enable_ota,
	output	wire 		ext_1n,
	output	wire 		ext_1n_bl_sel,
	output	wire 		ext_1n_sl_sel,
	output	wire 		ext_3n,
	output	wire 		ext_3n_bl_sel,
	output	wire 		ext_3n_sl_sel,
	output	wire 		ext_inference_enable_bl,
	output	wire 		ext_inference_enable_sl,
	output	wire 		ext_precharge_bl,
	output	wire 		ext_precharge_sl,
	output	wire 		ext_turn_on_wl,
	output	wire 		forward,
	output	wire 		ifat_mode,
	output	wire 		inference_mode,
	output	wire 		latch_en,
	output	wire 		lfsr_mode,
	output	wire 		lfsrhorz_clk,
	output	wire 		lfsrpulse,
	output	wire 		reg_controlled_wl,
	output	wire 		reg_write_enable_horz,
	output	wire 		reg_write_enable_vert,
	output	wire 		register_mode,
	output	wire [1:0] 	select_write_reg,
	output	wire 		sw_cds,
	output	wire 		sw_cds_vref,
	output	wire 		sw_feedback,
	output	wire 		sw_in_sample,
	output	wire 		sw_integ,
	output	wire 		sw_ota_loop,
	output	wire 		trigger_in,
	output	wire 		turn_off_precharge,
	output	wire [1:0]	vcomp_config,
	output	wire [1:0]	vlevel1_config,
	output	wire [1:0]	vlevel2_config,
	output	wire [1:0]	vupdate_config,
	output	wire [1:0]	wupdate_clk,
	output	wire 		wupdate_mode,
	output	wire 		wupdate_pulse,
	output	wire 		lfsrhorz_set_b,
	output	wire [1:0]	reg_config,

	output	wire [1:0]	spi_clock_0,
	output	wire [1:0]	spi_clock_1_3,
	output	wire [1:0]	spi_clock_4_7,
	output  wire [2:0]  addr_horz,
	output  wire [2:0]  addr_vert,
	output  wire 		core_enable_clk,
	output  wire 		core_enable_in,
	output  wire 		core_enable_reset_b,
	output	wire [15:0]	shift2chip,
	input	wire [15:0]	shift2ok,
	input	wire [1:0]	lfsr_out,

	// DMUX, DAC, ADC, MUX interface
	output  wire [1:0]	dmux_oe_b,
	output 	wire 		adc_rram_sck,
	output 	wire 		adc_rram_cnv,
	input 	wire 		adc_rram_sdo,
	output 	wire 		sel_tia,
	output 	wire 		sel_vfb,
	output  wire 		sel_spare,
	output 	wire 		dac_cs_b,
	output 	wire 		dac_sck,
	output 	wire 		dac_sdi
	);

parameter ADC_RRAM_NUM = 1;
parameter DAC_NUM = 4;
parameter NMLO_LENGTH = 128;
parameter NMLO_CORE = 3;

// Clocks
wire sys_clk;
IBUFGDS osc_clk(.O(sys_clk), .I(sys_clkp), .IB(sys_clkn));
wire clk_50MHz, clk_25MHz, clk_12MHz, clk_6MHz;

reg [3:0] clk_cnt;
always @(posedge sys_clk) begin
	clk_cnt <= clk_cnt + 1;
end

assign clk_50MHz = clk_cnt[0];
assign clk_25MHz = clk_cnt[1];
assign clk_12MHz = clk_cnt[2];
assign clk_6MHz = clk_cnt[3];

wire dac_clk; // DAC clock
wire adc_rram_clk; // ADC/TIA clock
wire neurram_clk; // Neuron control clock
wire spi_ctrl_clk; // SPI interface clock

assign dac_clk = clk_6MHz;
assign adc_rram_clk = clk_12MHz;
assign neurram_clk = sys_clk; //clk_12MHz;
assign spi_ctrl_clk = clk_25MHz;

// Target interface bus:
wire         okClk;
wire [112:0] okHE;
wire [64:0]  okEH;

// Endpoint connections:
wire [31:0]  ep00wire;
wire [31:0]  ep01wire; // DAC input
wire [31:0]  ep02wire; // RRAM TIA/ADC t_shld and t_delta
wire [31:0]  ep03wire; // I/O registers control
wire [31:0]  ep04wire; // I/O registers random access interface
wire [31:0]  ep05wire; // Neurram operating modes
wire [31:0]  ep06wire; // Other Neurram controls
wire [31:0]  ep07wire; // RRAM SET and RESET pulse width
wire [31:0]  ep08wire; // external control signals
wire [31:0]  ep09wire; // neuron configuration signals
wire [31:0]  ep0Awire; // LFSR configuration signal
wire [31:0]  ep0Bwire; // Weight update voltage muxes configuration
wire [31:0]  ep0Cwire; // Weight update mode pulse width
wire [31:0]  ep0Dwire; // SPI read and write config
wire [31:0]  ep0Ewire; // matmul unsigned control signals
wire [31:0]  ep0Fwire; // multi-level output control signals
wire [31:0]  ep10wire; // debugging controls
wire [31:0]  ep11wire; // neurram energy testing cycles
wire [31:0]  ep12wire; // Auto-ranging vref
wire [31:0]  ep13wire; // global core address
wire [31:0]  ep14wire; // core select register controls
wire [31:0]  ep15wire; // global mask for SPI triggers

wire [31:0]  ep20wire; // DAC & ADC status
wire [31:0]  ep21wire, ep22wire; // ADC output
// wire [31:0]  ep23wire, ep24wire, ep25wire, ep26wire; // ADC/TIA output
// wire [31:0]  ep27wire; // Neurram random access dataout
wire [31:0]  ep28wire; // Neuron status
wire [31:0]  ep29wire, ep2Awire; // Auto-ranging output
wire [31:0]  ep2Bwire; // auto-ranging t_shld & t_delta

wire [31:0]  ep40wire; // sys_clk synchronized trigger
wire [31:0]  ep41wire; // dac_clk trigger
// wire [31:0]  ep42wire; // 
wire [31:0]  ep43wire; // adc_rram_clk trigger
wire [31:0]  ep44wire; // neurram_clk trigger
wire [31:0]  ep45wire; // neurram pulse trigger

wire [31:0]  ep80pipe; // Pipe-in for neurram SPI
wire         ep80write;
wire [31:0]	 ep81pipe; // Pipe-in for neuron multi-level output address mask
wire         ep81write;
wire [31:0]  epA0pipe; // Pipe-out for neurram SPI
wire         epA0read;
wire [31:0]  epA1pipe; // Pipe-out for LFSR
wire         epA1read;
wire [31:0]  epA2pipe; // Pipe-out for neuron multi-level outputs
wire         epA2read;


// Instantiate the okHost and connect endpoints.
wire [65*12-1:0]  okEHx;
okHost okHI(
	.okUH(okUH),
	.okHU(okHU),
	.okUHU(okUHU),
	.okAA(okAA),
	.okClk(okClk),
	.okHE(okHE), 
	.okEH(okEH)
);

okWireOR # (.N(12)) wireOR (okEH, okEHx);

okWireIn     ep00 (.okHE(okHE),                             .ep_addr(8'h00), .ep_dataout(ep00wire));
okWireIn     ep01 (.okHE(okHE),                             .ep_addr(8'h01), .ep_dataout(ep01wire));
okWireIn     ep02 (.okHE(okHE),                             .ep_addr(8'h02), .ep_dataout(ep02wire));
okWireIn     ep03 (.okHE(okHE),                             .ep_addr(8'h03), .ep_dataout(ep03wire));
okWireIn     ep04 (.okHE(okHE),                             .ep_addr(8'h04), .ep_dataout(ep04wire));
okWireIn     ep05 (.okHE(okHE),                             .ep_addr(8'h05), .ep_dataout(ep05wire));
okWireIn     ep06 (.okHE(okHE),                             .ep_addr(8'h06), .ep_dataout(ep06wire));
okWireIn     ep07 (.okHE(okHE),                             .ep_addr(8'h07), .ep_dataout(ep07wire));
okWireIn     ep08 (.okHE(okHE),                             .ep_addr(8'h08), .ep_dataout(ep08wire));
okWireIn     ep09 (.okHE(okHE),                             .ep_addr(8'h09), .ep_dataout(ep09wire));
okWireIn     ep0A (.okHE(okHE),                             .ep_addr(8'h0a), .ep_dataout(ep0Awire));
okWireIn     ep0B (.okHE(okHE),                             .ep_addr(8'h0b), .ep_dataout(ep0Bwire));
okWireIn     ep0C (.okHE(okHE),                             .ep_addr(8'h0c), .ep_dataout(ep0Cwire));
okWireIn     ep0D (.okHE(okHE),                             .ep_addr(8'h0d), .ep_dataout(ep0Dwire));
okWireIn     ep0E (.okHE(okHE),                             .ep_addr(8'h0e), .ep_dataout(ep0Ewire));
okWireIn     ep0F (.okHE(okHE),                             .ep_addr(8'h0f), .ep_dataout(ep0Fwire));
okWireIn     ep10 (.okHE(okHE),                             .ep_addr(8'h10), .ep_dataout(ep10wire));
okWireIn     ep11 (.okHE(okHE),                             .ep_addr(8'h11), .ep_dataout(ep11wire));
okWireIn     ep12 (.okHE(okHE),                             .ep_addr(8'h12), .ep_dataout(ep12wire));
okWireIn     ep13 (.okHE(okHE),                             .ep_addr(8'h13), .ep_dataout(ep13wire));
okWireIn     ep14 (.okHE(okHE),                             .ep_addr(8'h14), .ep_dataout(ep14wire));
okWireIn     ep15 (.okHE(okHE),                             .ep_addr(8'h15), .ep_dataout(ep15wire));


okWireOut    ep20 (.okHE(okHE), .okEH(okEHx[ 0*65 +: 65 ]), .ep_addr(8'h20), .ep_datain(ep20wire));
okWireOut    ep21 (.okHE(okHE), .okEH(okEHx[ 1*65 +: 65 ]), .ep_addr(8'h21), .ep_datain(ep21wire));
okWireOut    ep22 (.okHE(okHE), .okEH(okEHx[ 2*65 +: 65 ]), .ep_addr(8'h22), .ep_datain(ep22wire));
// okWireOut    ep23 (.okHE(okHE), .okEH(okEHx[ 3*65 +: 65 ]), .ep_addr(8'h23), .ep_datain(ep23wire));
// okWireOut    ep24 (.okHE(okHE), .okEH(okEHx[ 4*65 +: 65 ]), .ep_addr(8'h24), .ep_datain(ep24wire));
// okWireOut    ep25 (.okHE(okHE), .okEH(okEHx[ 5*65 +: 65 ]), .ep_addr(8'h25), .ep_datain(ep25wire));
// okWireOut    ep26 (.okHE(okHE), .okEH(okEHx[ 6*65 +: 65 ]), .ep_addr(8'h26), .ep_datain(ep26wire));
// okWireOut    ep27 (.okHE(okHE), .okEH(okEHx[ 7*65 +: 65 ]), .ep_addr(8'h27), .ep_datain(ep27wire));
okWireOut    ep28 (.okHE(okHE), .okEH(okEHx[ 3*65 +: 65 ]), .ep_addr(8'h28), .ep_datain(ep28wire));
okWireOut    ep29 (.okHE(okHE), .okEH(okEHx[ 4*65 +: 65 ]), .ep_addr(8'h29), .ep_datain(ep29wire));
okWireOut    ep2A (.okHE(okHE), .okEH(okEHx[ 5*65 +: 65 ]), .ep_addr(8'h2a), .ep_datain(ep2Awire));
okWireOut    ep2B (.okHE(okHE), .okEH(okEHx[ 6*65 +: 65 ]), .ep_addr(8'h2b), .ep_datain(ep2Bwire));

okTriggerIn  ti40 (.okHE(okHE),                             .ep_addr(8'h40), .ep_clk(sys_clk), .ep_trigger(ep40wire));
okTriggerIn  ti41 (.okHE(okHE),                             .ep_addr(8'h41), .ep_clk(dac_clk), .ep_trigger(ep41wire));
// okTriggerIn  ti42 (.okHE(okHE),                             .ep_addr(8'h42), .ep_clk(adc_sf_clk), .ep_trigger(ep42wire));
okTriggerIn  ti43 (.okHE(okHE),                             .ep_addr(8'h43), .ep_clk(adc_rram_clk), .ep_trigger(ep43wire));
okTriggerIn  ti44 (.okHE(okHE),                             .ep_addr(8'h44), .ep_clk(spi_ctrl_clk), .ep_trigger(ep44wire));
okTriggerIn  ti45 (.okHE(okHE),                             .ep_addr(8'h45), .ep_clk(neurram_clk), .ep_trigger(ep45wire));

okPipeIn     pi80 (.okHE(okHE), .okEH(okEHx[ 7*65 +: 65 ]), .ep_addr(8'h80), .ep_dataout(ep80pipe), .ep_write(ep80write));
okPipeIn     pi81 (.okHE(okHE), .okEH(okEHx[ 8*65 +: 65 ]), .ep_addr(8'h81), .ep_dataout(ep81pipe), .ep_write(ep81write));
okPipeOut    poA0 (.okHE(okHE), .okEH(okEHx[ 9*65 +: 65 ]), .ep_addr(8'ha0), .ep_datain(epA0pipe), .ep_read(epA0read));
okPipeOut    poA1 (.okHE(okHE), .okEH(okEHx[ 10*65 +: 65 ]), .ep_addr(8'ha1), .ep_datain(epA1pipe), .ep_read(epA1read));
okPipeOut    poA2 (.okHE(okHE), .okEH(okEHx[ 11*65 +: 65 ]), .ep_addr(8'ha2), .ep_datain(epA2pipe), .ep_read(epA2read));


// DAC daisy chain control.
wire dac_rst, dac_program, dac_update;
wire [31:0] dac_din;
wire dac_fifo_full, dac_fifo_empty, dac_fifo_ack, dac_state_idle;

assign dac_rst = ep00wire[0];
assign dac_program = ep40wire[1];
assign dac_update = ep41wire[0];
assign dac_din = ep01wire;
assign ep20wire[0] = dac_fifo_full;
assign ep20wire[1] = dac_fifo_ack;
assign ep20wire[3] = dac_fifo_empty;
assign ep20wire[5] = dac_state_idle;

dac_daisy_control dac_ctrl(
	.clk(sys_clk),
	.clk_dac(dac_clk),
	.rst(dac_rst),
	.program(dac_program),
	.update(dac_update),
	.ep_din(dac_din),
	.fifo_full(dac_fifo_full),
	.fifo_wr_done(dac_fifo_ack),
	.fifo_empty(dac_fifo_empty),
	.dac_state_idle(dac_state_idle),
	.dac_sck(dac_sck),
	.dac_cs_b(dac_cs_b),
	.dac_sdi(dac_sdi)
	);


// TIA, ADC control.
wire adc_rram_rst, adc_rram_trigger, adc_rram_ack;
wire adc_rram_valid;
wire [18*ADC_RRAM_NUM-1:0] adc_rram_dout1, adc_rram_dout2;
wire [15:0] rram_t_shld, rram_t_delta;
wire vread_on;

assign adc_rram_rst = ep00wire[0];
assign adc_rram_trigger = ep43wire[0] | atrg_rr_trigger;
assign adc_rram_ack = ep43wire[1] | atrg_rr_ack_trigger;
assign ep20wire[8] = adc_rram_valid;
assign ep21wire[18*ADC_RRAM_NUM-1:0] = adc_rram_dout1;
assign ep22wire[18*ADC_RRAM_NUM-1:0] = adc_rram_dout2;
// assign rram_t_shld = ep02wire[15:0];
// assign rram_t_delta = ep02wire[31:16];

resistance_read #(.num_adc(ADC_RRAM_NUM)) tia_adc(
	.clk(adc_rram_clk),
	.rst(adc_rram_rst),
	.trigger(adc_rram_trigger),
	.ack(adc_rram_ack),
	.t_shld(rram_t_shld),
	.t_delta(rram_t_delta),
	.ready(adc_rram_valid),
	.dout1(adc_rram_dout1),
	.dout2(adc_rram_dout2),
	.vin_on(vread_on),
	.sel_opamp_in(sel_tia),
	.adc_sck(adc_rram_sck),
	.convst(adc_rram_cnv),
	.adc_sdo(adc_rram_sdo)
	);


wire auto_range_trigger, auto_range_ack;
wire [17:0] auto_range_vref;
wire atrg_rr_trigger, atrg_rr_ack_trigger;
wire [15:0] t_shld_init, t_delta_init;
wire [17:0] atrg_dout1, atrg_dout2;
wire atrg_valid;

assign auto_range_trigger = ep43wire[2];
assign auto_range_ack = ep43wire[3];
assign auto_range_vref = ep12wire[17:0];
assign t_shld_init = ep02wire[15:0];
assign t_delta_init = ep02wire[31:16];
assign ep29wire[17:0] = atrg_dout1;
assign ep2Awire[17:0] = atrg_dout2;
assign ep20wire[2] = atrg_valid;
assign ep2Bwire[15:0] = rram_t_shld;
assign ep2Bwire[31:16] = rram_t_delta;

// assign ep20wire[3] = atrg_error_polarity;

auto_ranging at_rg(
	.clk(adc_rram_clk),
	.rst(adc_rram_rst),
	.trigger(auto_range_trigger),
	.t_shld_init(t_shld_init),
	.t_delta_init(t_delta_init),
	.vref(auto_range_vref),
	.ack(auto_range_ack),
	.dout1(atrg_dout1),
	.dout2(atrg_dout2),
	.valid(atrg_valid),
	.rr_trigger(atrg_rr_trigger),
	.ack_trigger(atrg_rr_ack_trigger),
	.t_shld(rram_t_shld),
	.t_delta(rram_t_delta),
	.rr_ready(adc_rram_valid),
	.rr_dout1(adc_rram_dout1),
	.rr_dout2(adc_rram_dout2)
	);


wire backward;
assign backward = ~forward;

// Neurram interface
assign addr_local = ep04wire[7:0];
assign clear_horz_b = ~(ep03wire[0] | matmul_reg_reset | nmlo_reg_reset);
assign clear_vert_b = ~(ep03wire[1] | matmul_reg_reset | nmlo_reg_reset);
// assign dec_enable_horz = ep03wire[2];
// assign dec_enable_vert = ep03wire[3];
assign enable_ota = ep06wire[6];
// assign enable_sf = ep06wire[7];
assign ext_1n = ep08wire[3] | neuron_ext_1n;
assign ext_1n_bl_sel = (ep05wire[0] | ep05wire[1]) & forward;
assign ext_1n_sl_sel = (ep05wire[0] | ep05wire[1]) & backward;
assign ext_3n = ep08wire[0] | neuron_ext_3n | lfsr_mode;
assign ext_3n_bl_sel = ((ep05wire[0] | register_mode) & backward) | ((ep05wire[1] & ~register_mode) & forward);
assign ext_3n_sl_sel = ((ep05wire[0] | register_mode) & forward) | ((ep05wire[1] & ~register_mode) & backward);
assign ext_inference_enable_bl = ep08wire[8] | neuron_inf_enable;
assign ext_inference_enable_sl = ep08wire[9] | neuron_inf_enable;
assign ext_precharge_bl = ep08wire[6] | neuron_precharge_on;
assign ext_precharge_sl = ep08wire[7] | neuron_precharge_on;
assign ext_turn_on_wl = ep08wire[10] | (ep05wire[0] & neuron_wl_on);
// assign flip_neuron_horz_1 = ep06wire[1];
// assign flip_neuron_horz_2 = ep06wire[2];
// assign flip_neuron_vert_1 = ep06wire[3];
// assign flip_neuron_vert_2 = ep06wire[4];
assign forward = ep05wire[5];
assign ifat_mode = 1'b0;
assign inference_mode = 1'b0;
// assign latch_en = ep10wire[6];
assign lfsr_mode = ep05wire[2] | lfsr_mode_on; //& (~(neuron_compare_write | register_mode | ep05wire[0] | ep05wire[1] | ep05wire[4]));
// assign lfsrhorz_clk = 1'b0;
// assign lfsrpulse = 1'b0;
assign reg_controlled_wl = ep06wire[5];
// assign reg_read_enable_horz = ep03wire[4];
// assign reg_read_enable_vert = ep03wire[5];
// assign reg_write_enable_horz = ep03wire[6];
// assign reg_write_enable_vert = ep03wire[7];
// assign register_mode = ep05wire[3];
assign select_write_reg = ep04wire[9:8];
// assign select_write_reg_vert = ep04wire[11:10];
// assign sw_cds = ep10wire[0];
// assign sw_cds_vref = ep10wire[1];
// assign sw_feedback = ep10wire[2];
assign sw_in_sample = sw_in_sample_neuron | lfsr_neuron_on;
// assign sw_integ = ep10wire[4];
// assign sw_ota_loop = ep10wire[5];
// assign trigger_in = ep45wire[0];
assign turn_off_precharge = ep08wire[11];
// assign vcomp_config = ep10wire[8:7];
// assign vlevel1_config = ep10wire[10:9];
// assign vlevel2_config = ep10wire[12:11];
// assign vupdate_config = 2'b00;
// assign wupdate_clk = 2'b00;
assign wupdate_mode = (wupdate_mode_pulse | ep05wire[4]) & (~(register_mode | ep05wire[0] | ep05wire[1] | ep05wire[2]));
// assign wupdate_pulse = 1'b0;
assign lfsrhorz_set_b = ~ep0Awire[0];
//assign sel_vfb = 1'b0; //TODO


assign addr_horz = ep13wire[2:0];
assign addr_vert = ep13wire[5:3];
assign dec_enable_horz = ep13wire[6];
assign dec_enable_vert = ep13wire[7];
assign core_enable_clk = ep14wire[1];
assign core_enable_in = ep14wire[2];
assign core_enable_reset_b = ~ep14wire[0];

assign dmux_oe_b[0] = ~(reg_config == 2'b11);
assign dmux_oe_b[1] = ~(reg_config == 2'b01);
assign sel_spare = 0;

// pipepin2fifo arbiter
wire neurram_rst;
wire [7:0] core_select;
wire [9:0] write_num_words;
wire arb_pipein2spi_idle, spi_pipe_in_full;
wire [255:0] data_pipein2spi;
wire [7:0] write_pipein2spi;
wire [7:0] fifo_full_spi2pipein;

assign core_select = ep15wire[7:0];
assign write_num_words = ep15wire[17:8];
assign ep28wire[5] = arb_pipein2spi_idle;
assign ep28wire[10] = spi_pipe_in_full;

arbiter_pipein2fifo arbiter_pipein2spi(
	.clk(sys_clk),
	.ok_clk(okClk),
	.rst(neurram_rst),
	.pipe_in(ep80pipe),
	.pipe_in_write(ep80write),
	.core_select(core_select),
	.num_words(write_num_words),
	.padding_words(2'd0),
	.idle(arb_pipein2spi_idle),
	.pipe_in_full(spi_pipe_in_full),
	.data2fifo(data_pipein2spi),
	.wr_en_2fifo(write_pipein2spi),
	.rd_en_2ok(~fifo_full_spi2pipein)
	);


// fifo2pipeout arbiter
wire [7:0] read_num_words;
wire arb_spi2pipeout_idle, spi_pipe_out_empty;
wire [255:0] data_spi2pipeout;
wire [7:0] read_pipeout2spi;
wire [7:0] empty_spi2pipeout, valid_spi2pipeout;

assign read_num_words = ep15wire[25:18];
assign ep28wire[6] = arb_spi2pipeout_idle;
assign ep28wire[11] = spi_pipe_out_empty;

arbiter_fifo2pipeout arbiter_spi2pipeout(
	.clk(sys_clk),
	.ok_clk(okClk),
	.rst(neurram_rst),
	.pipe_out(epA0pipe),
	.pipe_out_read(epA0read),
	.core_select(core_select),
	.num_words(read_num_words),
	.idle(arb_spi2pipeout_idle),
	.pipe_out_empty(spi_pipe_out_empty),
	.data_from_fifo(data_spi2pipeout),
	.empty_from_fifo(empty_spi2pipeout),
	.valid_from_fifo(valid_spi2pipeout),
	.rd_en_from_ok(read_pipeout2spi)
	);

// SPI clock

wire reg_spi_trigger, reg_rand_access_trigger;
wire [7:0] ml_read_spi_trigger;
wire [1:0] reg_neuron_read_trigger;
wire [7:0] state_spi_clk, state_spi_idle;
wire [1:0] reg_shift_out, reg_shift_in;
wire [1:0] spi_config;
wire [3:0] spi_shift_multiplier, spi_pipe_in_steps, spi_pipe_out_steps, spi_extra_shift_cycles;
wire [NMLO_LENGTH*NMLO_CORE-1:0] spi_from_neurram[7:0];
wire [255:0] spi_single_core[7:0];
wire all_spi_idle;
wire [1:0] spi_clk_random_and_neuron;
wire record_spi;
wire shift_fwd;
wire [3:0] nmlo_shift_multiplier;
wire nmlo_spi_trigger;

assign neurram_rst = ep00wire[1];
assign reg_spi_trigger = ep44wire[0] | matmul_spi_trig;
assign reg_rand_access_trigger = ep44wire[1];
assign spi_config = ml_read_idle ? ep0Dwire[1:0] : 2'b00;
assign spi_shift_multiplier = ml_read_idle ? ep0Dwire[5:2] : nmlo_shift_multiplier;
assign spi_pipe_in_steps = ep0Dwire[9:6];
assign spi_pipe_out_steps = ep0Dwire[13:10];
assign spi_extra_shift_cycles = ep0Dwire[18:15];
// assign reg_neuron_read_trigger = ep44wire[3:2];
assign all_spi_idle = & state_spi_idle;
assign ep28wire[2] = all_spi_idle;
assign record_spi = ~ml_read_idle;
assign shift_fwd = ml_read_idle ? ep0Dwire[14] : 1'b0;
assign nmlo_spi_trigger = | ml_read_spi_trigger;


neurram_reg_control neurram_reg(
	.clk(spi_ctrl_clk),
	.rst(neurram_rst),
	.spi_trigger(reg_spi_trigger | nmlo_spi_trigger),
	.rand_access_trigger(reg_rand_access_trigger),
	.neuron_read_trigger(reg_neuron_read_trigger),
	.shift_fwd(shift_fwd),
	.rand_access_vert(ep04wire[10]),
	.inf_fwd(forward),
	.state_spi_clk(1'b0),
	.state_spi_idle(all_spi_idle),
	.spi_clk(spi_clk_random_and_neuron),
	.reg_config(reg_config),
	.reg_write_enable_horz(reg_write_enable_horz),
	.reg_write_enable_vert(reg_write_enable_vert)
	);

assign spi_clock_0[0] = state_spi_clk[0] | spi_clk_random_and_neuron[0];
assign spi_clock_0[1] = state_spi_clk[0] | spi_clk_random_and_neuron[1];
assign spi_clock_1_3[0] = state_spi_clk[1] | state_spi_clk[2] | state_spi_clk[3] | spi_clk_random_and_neuron[0];
assign spi_clock_1_3[1] = state_spi_clk[1] | state_spi_clk[2] | state_spi_clk[3] | spi_clk_random_and_neuron[1];
assign spi_clock_4_7[0] = state_spi_clk[4] | state_spi_clk[5] | state_spi_clk[6] | state_spi_clk[7] | spi_clk_random_and_neuron[0];
assign spi_clock_4_7[1] = state_spi_clk[4] | state_spi_clk[5] | state_spi_clk[6] | state_spi_clk[7] | spi_clk_random_and_neuron[1];

genvar i;
generate
for (i=0; i<8; i=i+1) begin: gen_spi
    neurram_spi_control #(.nmlo_length(NMLO_LENGTH), .nmlo_core(NMLO_CORE)) neurram_spi(
		.clk(spi_ctrl_clk),
		.ok_clk(sys_clk),
		.rst(neurram_rst),
		.spi_trigger((reg_spi_trigger | ml_read_spi_trigger[i]) & core_select[i]),
		.spi_config(spi_config),
		.shift_multiplier(spi_shift_multiplier),
		.pipe_in_steps(spi_pipe_in_steps),
		.pipe_out_steps(spi_pipe_out_steps),
		.extra_shift_cycles(spi_extra_shift_cycles),
		.spi_idle(state_spi_idle[i]),
		.pipe_in(data_pipein2spi[i*32 +: 32]),
		.in_fifo_wr_en(write_pipein2spi[i]),
		.in_fifo_full(fifo_full_spi2pipein[i]),
		.pipe_out(data_spi2pipeout[i*32 +: 32]),
		.out_fifo_rd_en(read_pipeout2spi[i]),
		.out_fifo_empty(empty_spi2pipeout[i]),
		.out_fifo_valid(valid_spi2pipeout[i]),
		.spi_clk(state_spi_clk[i]),
		.shift_out(shift2chip[i*2 +: 2]),
		.shift_in(shift2ok[i*2 +: 2]),
		.record_spi(record_spi),
		.spi_from_neurram(spi_from_neurram[i]),
		.spi_single_core(spi_single_core[i])
	);
end
endgenerate


// Weight Update Control

wire program_trigger, wupdate_mode_trigger, program_ack;
wire program_done;
wire [31:0] pulse_width;
wire [31:0] wupdate_mode_width;
wire wupdate_mode_pulse;

assign program_trigger = ep40wire[2] | ep40wire[3];
assign wupdate_mode_trigger = ep40wire[5];
assign program_ack = ep40wire[4];
assign pulse_width = ep07wire[31:0];
assign wupdate_mode_width = ep0Cwire[31:0];
assign ep20wire[4] = program_done;


neurram_wupdate_control neurram_wupdate(
	.clk(sys_clk),
	.rst(neurram_rst),
	.read_trigger(auto_range_trigger),
	.read_ack(atrg_valid),
	.vread_on(vread_on),
	.program_trigger(program_trigger),
	.wupdate_mode_trigger(wupdate_mode_trigger),
	.pulse_width(pulse_width),
	.wupdate_mode_width(wupdate_mode_width),
	.program_ack(program_ack),
	.program_done(program_done),
	.wupdate_pulse(wupdate_pulse),
	.wupdate_mode(wupdate_mode_pulse)
	);

assign wupdate_clk = 2'b00;
assign vupdate_config = ep0Bwire[1:0];
//assign wupdate_pulse = ep10wire[4];

// LFSR Control

wire lfsr_shift_trigger, lfsr_pulse_trigger;
wire lfsr_neuron_on, lfsr_inf_mode_off, lfsr_ext_inf_on;
wire lfsr_mode_on;
wire [3:0] lfsr_pulse_width;
wire lfsr_integ_trig;

assign lfsr_shift_trigger = ep44wire[4];
assign lfsr_pulse_trigger = ep44wire[5];
assign lfsr_pulse_width = ep10wire[3:0];

lfsr_control lfsr(
	.clk(spi_ctrl_clk),
	.ok_clk(okClk),
	.rst(neurram_rst),
	.lfsr_shift_trigger(lfsr_shift_trigger),
	.lfsr_pulse_trigger(lfsr_pulse_trigger),
	.pipe_out(epA1pipe),
	.out_fifo_rd_en(epA1read),
	.out_fifo_empty(),
	.lfsr_clk(lfsrhorz_clk),
	.lfsr_in(lfsr_out),
	.lfsr_pulse(lfsrpulse),
	.lfsr_neuron_on(lfsr_neuron_on),
	.inf_mode_off(lfsr_inf_mode_off),
	.ext_inf(lfsr_ext_inf_on),
	.lfsr_mode(lfsr_mode_on),
	.pulse_width(lfsr_pulse_width),
	.sw_integ(sw_integ),
	.integ_trig(lfsr_integ_trig)
	);

//Neuron Control

wire neuron_run_all, neuron_run_all_reset, keep_wl_on, neuron_multi_sampling;
wire [1:0] neuron_comparison_phase, neuron_partial_reset_phase;
wire neuron_idle;
wire neuron_ext_1n, neuron_ext_3n, neuron_wl_on, neuron_inf_enable, neuron_precharge_on;
wire [7:0] t_opamp, t_sample;
wire [3:0] neuron_precharge_time;
wire sw_in_sample_neuron;
wire neuron_sample_trigger, neuron_integ_trig;
wire [7:0] neuron_num_pulses;
wire neuron_cds_trigger;
wire neuron_partial_reset_trigger, neuron_compare_write_trigger, ml_read_neuron_trigger;

assign neuron_run_all = ep09wire[0];
assign neuron_run_all_reset = ep09wire[1];
assign keep_wl_on = ep09wire[22];
assign neuron_multi_sampling = ep09wire[23];
assign t_opamp = ep09wire[11:4];
assign t_sample = ep10wire[11:4];
assign neuron_comparison_phase = ep09wire[13:12];
assign neuron_partial_reset_phase = ep09wire[13:12];
assign ep28wire[0] = neuron_idle;
assign neuron_sample_trigger = ep45wire[1] | matmul_neuron_sample_trig | in_energy_sample_trigger;
assign neuron_integ_trig = ep45wire[3] | lfsr_integ_trig;
assign neuron_num_pulses =  matmul_unsigned_idle ? (in_energy_idle ? ep09wire[21:14] : in_energy_num_pulses) : matmul_num_pulses;
assign neuron_cds_trigger = ep45wire[0] | in_energy_cds_trigger | out_energy_cds_trigger | matmul_neuron_cds_trig;
assign neuron_partial_reset_trigger = ep45wire[4] | out_energy_reset_trigger;
assign neuron_compare_write_trigger = ep45wire[2] | ml_read_neuron_trigger;

// assign neuron_precharge_time = ep10wire[3:0];

neuron_control neuron(
	.clk(neurram_clk),
	.rst(neurram_rst),
	.cds_trigger(neuron_cds_trigger),
	.pulse_trigger(neuron_sample_trigger),
	.compare_write_trigger(neuron_compare_write_trigger),
	.integ_trigger(neuron_integ_trig),
	.run_all(neuron_run_all),
	.run_all_reset(neuron_run_all_reset),
	.keep_wl_on(keep_wl_on),
	.multi_sampling(neuron_multi_sampling),
	.reset_latch_on(ep09wire[24]),
	.comparison_phase(neuron_comparison_phase),
	.partial_reset_phase(neuron_partial_reset_phase),
	.idle(neuron_idle),
	.t_sample(t_sample),
	.t_cds(ep09wire[31:25]),
	.t_integ(t_opamp),
	.t_comp(ep10wire[19:12]),
	.t_reset(ep10wire[27:20]),
	.sw_cds(sw_cds),
	.sw_cds_vref(sw_cds_vref),
	.sw_in_sample(sw_in_sample_neuron),
	.sw_integ(sw_integ),
	.sw_ota_loop(sw_ota_loop),
	.sw_feedback(sw_feedback),
	.sw_latch_en(latch_en),
	.vcomp_config(vcomp_config),
	.v_level1_config(vlevel1_config),
	.v_level2_config(vlevel2_config),
	.ns_pulse_trig(trigger_in),
	.ext_1n(neuron_ext_1n),
	.ext_3n(neuron_ext_3n),
	.neuron_read_trigger(reg_neuron_read_trigger),
	.register_mode(register_mode),
	.num_pulses(neuron_num_pulses),
	.sel_vfb(sel_vfb),
	.partial_reset_trigger(neuron_partial_reset_trigger),
	.wl_on(neuron_wl_on),
	.inf_enable(neuron_inf_enable),
	.precharge_on(neuron_precharge_on),
	.state(ep28wire[20:15])
	);

wire in_energy_trigger, out_energy_trigger;
wire [31:0] energy_cycles;
wire in_energy_idle, out_energy_idle;
wire in_energy_sample_trigger, in_energy_cds_trigger, out_energy_reset_trigger, out_energy_cds_trigger;
wire [7:0] in_energy_num_pulses;
wire [7:0] out_energy_steps;

wire [2:0] matmul_unsigned_num_bits;
wire [4:0] matmul_unsigned_pulse_mult;

assign in_energy_trigger = ep45wire[10];
assign out_energy_trigger = ep45wire[11];
assign energy_cycles = ep11wire;
assign out_energy_steps = ep0Fwire[15:8];
assign ep28wire[1] = in_energy_idle;
assign ep28wire[14] = out_energy_idle;

matmul_input_energy_test in_energy(
	.clk(neurram_clk),
	.rst(neurram_rst),
	.trigger(in_energy_trigger),
	.num_bits(matmul_unsigned_num_bits),
	.pulse_multiplier(matmul_unsigned_pulse_mult),
	.iterations(energy_cycles),
	.idle(in_energy_idle),
	.neuron_idle(neuron_idle),
	.neuron_sample_trigger(in_energy_sample_trigger),
	.neuron_cds_trigger(in_energy_cds_trigger),
	.num_pulses(in_energy_num_pulses)
	);


matmul_output_energy_test out_energy(
	.clk(neurram_clk),
	.rst(neurram_rst),
	.trigger(out_energy_trigger),
	.cds(ep0Fwire[16]),
	.steps(out_energy_steps),
	.iterations(energy_cycles),
	.idle(out_energy_idle),
	.neuron_idle(neuron_idle),
	.neuron_cds_trigger(out_energy_cds_trigger),
	.neuron_reset_trigger(out_energy_reset_trigger)
	);


wire arb_pipein2nmlo_idle, nmlo_pipe_in_full;
wire [255:0] data_pipein2nmlo;
wire [7:0] write_pipein2nmlo;
wire [7:0] fifo_full_nmlo2pipein;
wire [2:0] nmlo_num_core;
wire [9:0] nmlo_pipein_num_words;

assign ep28wire[7] = arb_pipein2nmlo_idle;
assign ep28wire[12] = nmlo_pipe_in_full;
assign nmlo_num_core = ep0Fwire[2:0];
assign nmlo_pipein_num_words = NMLO_LENGTH * NMLO_CORE / 32;


arbiter_pipein2fifo #(.FIFO_SIZE(64)) arbiter_pipein2nmlo (
	.clk(sys_clk),
	.ok_clk(okClk),
	.rst(neurram_rst),
	.pipe_in(ep81pipe),
	.pipe_in_write(ep81write),
	.core_select(core_select),
	.num_words(nmlo_pipein_num_words),
	.padding_words(2'd0),
	.idle(arb_pipein2nmlo_idle),
	.pipe_in_full(nmlo_pipe_in_full),
	.data2fifo(data_pipein2nmlo),
	.wr_en_2fifo(write_pipein2nmlo),
	.rd_en_2ok(~fifo_full_nmlo2pipein)
	);

wire arb_nmlo2pipeout_idle, nmlo_pipe_out_empty;
wire [255:0] data_nmlo2pipeout;
wire [7:0] read_pipeout2nmlo;
wire [7:0] empty_nmlo2pipeout, valid_nmlo2pipeout;
wire [7:0] nmlo_pipeout_num_words;

assign ep28wire[8] = arb_nmlo2pipeout_idle;
assign ep28wire[13] = nmlo_pipe_out_empty;
assign nmlo_pipeout_num_words = nmlo_single_core? 64 : (nmlo_num_core * NMLO_LENGTH / 4);


arbiter_fifo2pipeout #(.FIFO_SIZE(4096)) arbiter_nmlo2pipeout(
	.clk(sys_clk),
	.ok_clk(okClk),
	.rst(neurram_rst),
	.pipe_out(epA2pipe),
	.pipe_out_read(epA2read),
	.core_select(core_select),
	.num_words(nmlo_pipeout_num_words),
	.idle(arb_nmlo2pipeout_idle),
	.pipe_out_empty(nmlo_pipe_out_empty),
	.data_from_fifo(data_nmlo2pipeout),
	.empty_from_fifo(empty_nmlo2pipeout),
	.valid_from_fifo(valid_nmlo2pipeout),
	.rd_en_from_ok(read_pipeout2nmlo)
	);


wire ml_read_trigger, ml_y_addr_trigger;
wire ml_read_idle;
wire [7:0] ml_read_idle_i;
wire [7:0] ml_neuron_trigger;
wire matmul_d2a_nmlo_trigger;
wire [7:0] nmlo_inf_mode_off_i, nmlo_ext_inf_enable_i, nmlo_reg_reset_i;
wire nmlo_inf_mode_off, nmlo_ext_inf_enable, nmlo_reg_reset;
wire nmlo_single_core;

assign ml_read_trigger = ep45wire[5] | matmul_d2a_nmlo_trigger;
assign ml_y_addr_trigger = ep45wire[6];
assign ml_read_idle = & ml_read_idle_i;
assign ep28wire[3] = ml_read_idle;
assign ml_read_neuron_trigger = | ml_neuron_trigger;
assign nmlo_shift_multiplier = ep0Fwire[6:3];
assign nmlo_inf_mode_off = | nmlo_inf_mode_off_i;
assign nmlo_ext_inf_enable = | nmlo_ext_inf_enable_i;
assign nmlo_reg_reset = | nmlo_reg_reset_i;
assign nmlo_single_core = ep0Fwire[7];


generate
for (i=0; i<8; i=i+1) begin: gen_nmlo
	neuron_multi_level_output #(.length(NMLO_LENGTH), .total_cores(NMLO_CORE)) nmlo(
		.clk(neurram_clk),
		.ok_clk(sys_clk),
		.rst(neurram_rst),
		.output_trigger(ml_read_trigger & core_select[i]),
		.y_addr_trigger(ml_y_addr_trigger & core_select[i]),
		.single_core(nmlo_single_core),
		.num_core(nmlo_num_core),
		.idle(ml_read_idle_i[i]),
		.pipe_in(data_pipein2nmlo[32*i +: 32]),
		.in_fifo_wr_en(write_pipein2nmlo[i]),
		.in_fifo_full(fifo_full_nmlo2pipein[i]),
		.pipe_out(data_nmlo2pipeout[32*i +: 32]),
		.out_fifo_rd_en(read_pipeout2nmlo[i]),
		.out_fifo_empty(empty_nmlo2pipeout[i]),
		.out_fifo_valid(valid_nmlo2pipeout[i]),
		.neuron_idle(neuron_idle),
		.spi_valid(all_spi_idle),
		.spi_input_row(spi_from_neurram[i]),
		.spi_input_single_core(spi_single_core[i]),
		.spi_read_trigger(ml_read_spi_trigger[i]),
		.neuron_reset_trigger(ml_neuron_trigger[i]),
		.turn_off_inference(nmlo_inf_mode_off_i[i]),
		.ext_inference_enable(nmlo_ext_inf_enable_i[i]),
		.reg_reset(nmlo_reg_reset_i[i])
	);
end
endgenerate




// neuron_multi_level_output nmlo(
// 	.clk(neurram_clk),
// 	.ok_clk(okClk),
// 	.rst(neurram_rst),
// 	.output_trigger(ml_read_trigger),
// 	.y_addr_trigger(ml_y_addr_trigger),
// 	.idle(ml_read_idle),
// 	.pipe_in(ep81pipe),
// 	.in_fifo_wr_en(ep81write),
// 	.in_fifo_full(),
// 	.pipe_out(epA2pipe),
// 	.out_fifo_rd_en(epA2read),
// 	.out_fifo_empty(),
// 	.out_fifo_valid(),
// 	.neuron_idle(neuron_idle),
// 	.spi_valid(all_spi_idle),
// 	.spi_input(spi_from_neurram[0]),
// 	.reg_config(ml_read_reg_config),
// 	.spi_read_trigger(ml_read_spi_trigger[0]),
// 	.neuron_reset_trigger(ml_read_neuron_trigger)
// );




wire matmul_unsigned_trigger, matmul_unsigned_cds, matmul_unsigned_idle;
wire matmul_neuron_sample_trig, matmul_neuron_cds_trig, matmul_spi_trig;
wire matmul_inference_mode_off, matmul_ext_inf_enable;
wire [7:0] matmul_num_pulses;
wire matmul_d2a_unsigned_trigger;
wire matmul_reg_reset;
// wire neuron_num_pulses_matmul;

assign matmul_unsigned_trigger = ep45wire[7] | matmul_d2a_unsigned_trigger;
assign ep28wire[4] = matmul_unsigned_idle;
assign matmul_unsigned_cds = ep0Ewire[0];
// assign neuron_num_pulses_matmul = ep0Ewire[1];
assign matmul_unsigned_num_bits = ep0Ewire[4:2];
assign matmul_unsigned_pulse_mult = ep0Ewire[9:5];

matmul_unsigned_helper matmul_unsigned(
	.clk(neurram_clk),
	.rst(neurram_rst),
	.trigger(matmul_unsigned_trigger),
	.cds(matmul_unsigned_cds),
	.reset(ep0Ewire[1]),
	.num_bits(matmul_unsigned_num_bits),
	.pulse_multiplier(matmul_unsigned_pulse_mult),
	.idle(matmul_unsigned_idle),
	.neuron_idle(neuron_idle),
	.spi_idle(all_spi_idle),
	.neuron_sample_trigger(matmul_neuron_sample_trig),
	.neuron_cds_trigger(matmul_neuron_cds_trig),
	.spi_write_trigger(matmul_spi_trig),
	.turn_off_inference(matmul_inference_mode_off),
	.ext_inference_enable(matmul_ext_inf_enable),
	.num_pulses(matmul_num_pulses),
	.reg_reset(matmul_reg_reset)
	);


matmul_dac2adc matmul_d2a(
	.clk(neurram_clk),
	.rst(neurram_rst),
	.trigger(ep45wire[8]),
	.iteration(ep0Ewire[17:10]),
	.idle(ep28wire[9]),
	.matmul_unsigned_idle(matmul_unsigned_idle),
	.nmlo_idle(ml_read_idle),
	.matmul_unsigned_trigger(matmul_d2a_unsigned_trigger),
	.nmlo_trigger(matmul_d2a_nmlo_trigger)
	);


endmodule


