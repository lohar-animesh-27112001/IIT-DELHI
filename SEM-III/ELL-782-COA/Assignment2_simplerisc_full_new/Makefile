TOP=tb/tb_simplerisc.v
RTL=rtl/simplerisc_top.v rtl/decode.vh rtl/immu.v rtl/alu.v rtl/regfile.v rtl/control_unit.v rtl/imem.v rtl/dmem.v
TB=tb/tb_simplerisc.v
VVP=sim.vvp

all: run

build:
	iverilog -g2012 -o $(VVP) $(TB) $(RTL)

run: build
	vvp $(VVP)

wave: run
	@echo "Open wave.vcd in GTKWave"

clean:
	rm -f $(VVP) wave.vcd
