using PyCall
using NPZ
using PyPlot

background = npzread("essential.npz")
r_Krishnendu = background["r"]

xi_Krishnendu = npzread("xisrc.npy")

