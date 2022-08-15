import torch

basedir = "./logs/" # Where to store ckpts and logs
expname = "nerf_fern" # Experiment Name
device = torch.device("cuda")

multires = 10 # log 2 of max freq for positional encoding (3D Locations)
multires_views = 4 # log 2 of max freq for positional encoding (2D direction)
i_embed = 0 # 0 - for positional encoding, -1 - None
N_importance = 0 # Number of additional fine samples per ray
use_viewdirs = True # Use full 5D inputs instead of 3D

net_depth = 8 # layers in network
net_width = 256 # channels per layer
net_depthfine = 8 # Layers in fine network
net_widthfine = 256 # Channels in fine per layer

ft_path = None # specific weights npy file to reload for coarse network
no_reload = True # do not reload weights from saved ckpt

perturb = 1. # set to 0. for no jitter, 1. for jitter
N_samples = 64 # number of coarse samples per ray
white_bkgd = True # set to render synthetic data on a white bkgd (always use for dvoxels)
raw_noise_std = 0. # std dev of noise added to regularize sigma_a output, 1e0 recommended

dataset_type = "llff"
no_ndc = True # do not use normalized device coordinates (set for non-forward facing scenes)
lindisp = True # sampling linearly in disparity rather than depth
llffhold = 8 # will take every 1/N images as LLFF test set, paper uses 8

render_only = True # do not optimize, reload weights and render out render_poses path
render_test = True # render the test set instead of render_poses path
chunk = 1024 * 32 # number of rays processed in parallel, decrease if running out of memory
netchunk = 1024 * 64 # number of pts sent through network in parallel, decrease if running out of memory
render_factor = 0. # downsampling factor to speed up rendering, set 4 or 8 for fast preview

N_rand = 32 * 32 * 4 # Batch Size (Number of random rays per gradient step)
no_batching = True # only take random rays from 1 image at a time

precrop_frac = 0.5 # fraction of img taken for central crops
precrop_iters = 0 # number of steps to train on central crops

lrate = 5e-4 # Learning Rate
lrate_decay = 250 # Exponential Learning Rate Decay