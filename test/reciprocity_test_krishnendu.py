import numpy as np
from pathlib import Path

Gfn_path = Path("/scratch/krishnendu/greenfn/reciprocity/deltac0.0-lmax-100lmin0numin2.0numax4.5")

G1 = np.load(Gfn_path/"omega-depth-0.4rsun-ell70-nu0500.npz")
G2 = np.load(Gfn_path/"omega-depth-0.6rsun-ell70-nu0500.npz")

params = np.load(Gfn_path/"essential.npz")
r = params["r"]

rsun = 6.9598e10
r /= rsun

src1=abs(r-0.4).argmin()
src2=abs(r-0.6).argmin()

ellgrid = np.arange()




