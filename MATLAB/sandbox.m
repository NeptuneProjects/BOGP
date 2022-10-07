clear all;
addpath(genpath("~/at"))

modfil0 = char("MODFIL0.mod");
modfil1 = char("MODFIL1.mod");
freq = 0;

clear read_modes_bin
data0 = read_modes(modfil0, freq);
clear read_modes_bin
data1 = read_modes(modfil1, freq);
