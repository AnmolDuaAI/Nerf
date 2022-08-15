from Data import DataReader
from Nerfs.original_nerf import NerfImplementation

basedir = "/home/user/anmol/Nerf/nerf-pytorch/data/nerf_llff_data/fern/"
data = DataReader(basedir)

nerf = NerfImplementation(data.get_data())
nerf.train()