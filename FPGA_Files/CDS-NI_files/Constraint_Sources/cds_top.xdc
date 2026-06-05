## cds_tops.xdc - Pin constraints for CDS-NI on Digilent Basys 3 (XC7A35T)
## you can set many of the LED and other elements to different ports if you want

## Clock (100 MHz oscillator)
set_property PACKAGE_PIN W5 [get_ports clk]
set_property IOSTANDARD LVCMOS33 [get_ports clk]
create_clock -period 10.000 -name sys_clk_pin [get_ports clk]

## Reset - center pushbutton (BTNC)
set_property PACKAGE_PIN U18 [get_ports reset]
set_property IOSTANDARD LVCMOS33 [get_ports reset]

## UART receive (USB-UART RX - directly from FTDI chip)
set_property PACKAGE_PIN B18 [get_ports rx_pin]
set_property IOSTANDARD LVCMOS33 [get_ports rx_pin]

## UART transmit (USB-UART TX - directly to FTDI chip)
set_property PACKAGE_PIN A18 [get_ports tx_pin]
set_property IOSTANDARD LVCMOS33 [get_ports tx_pin]

## LED 0 - decision bit 0
set_property PACKAGE_PIN U16 [get_ports {led_decision[0]}]
set_property IOSTANDARD LVCMOS33 [get_ports {led_decision[0]}]

## LED 1 - decision bit 1
set_property PACKAGE_PIN E19 [get_ports {led_decision[1]}]
set_property IOSTANDARD LVCMOS33 [get_ports {led_decision[1]}]

## LED 2 - prediction done indicator
set_property PACKAGE_PIN U19 [get_ports led_done]
set_property IOSTANDARD LVCMOS33 [get_ports led_done]

## FPGA configuration voltage
set_property CFGBVS VCCO [current_design]
set_property CONFIG_VOLTAGE 3.3 [current_design]
