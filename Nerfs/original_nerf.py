import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import Nerfs.original_nerf_config as config
from Nerfs.helpers import *
import time

img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))

'''
    Original Nerf Implementation
'''
class NerfImplementation:
    def __init__(self, res):
        torch.set_default_tensor_type('torch.cuda.FloatTensor') # A very important line 
        images, poses, bds, render_poses, i_test = res
        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        print('Loaded llff', images.shape, render_poses.shape, hwf)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if config.llffhold > 0:
            print('Auto LLFF holdout,', config.llffhold)
            i_test = np.arange(images.shape[0])[::config.llffhold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        if config.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
            
        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)        

        H, W, focal = hwf
        H, W = int(H), int(W)
        hwf = [H, W, focal]

        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

        if config.render_test:
            render_poses = np.array(poses[i_test])

        self.create_logdir()

        # Create nerf model
        render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf()

        global_step = start
        bds_dict = {'near' : near, 'far' : far,}
        render_kwargs_train.update(bds_dict)
        render_kwargs_test.update(bds_dict)

        # Move testing data to GPU
        render_poses = torch.Tensor(render_poses).to(config.device)


        # Storing States
        self.render_poses = render_poses
        self.hwf = hwf
        self.H = H
        self.W = W
        self.focal = focal
        self.K = K
        self.render_kwargs_train = render_kwargs_train
        self.render_kwargs_test = render_kwargs_test
        self.images = images
        self.poses = poses
        self.i_test = i_test
        self.i_val = i_val
        self.i_train = i_train
        self.grad_vars = grad_vars
        self.optimizer = optimizer
        self.start = start
        self.global_step = global_step

        print ("-------------- INITIALIZED ORIGINAL NERF ---------")

    # TODO Making log file
    def create_logdir(self):
        # Create log dir and copy the config file
        basedir = config.basedir
        expname = config.expname
        os.makedirs(os.path.join(basedir, expname), exist_ok=True)
        # f = os.path.join(basedir, expname, 'args.txt')
        # with open(f, 'w') as file:
        #     for arg in sorted(vars(args)):
        #         attr = getattr(args, arg)
        #         file.write('{} = {}\n'.format(arg, attr))
        # if args.config is not None:
        #     f = os.path.join(basedir, expname, 'config.txt')
        #     with open(f, 'w') as file:
        #         file.write(open(args.config, 'r').read())


    def render_only(self):
        print('RENDER ONLY')
        with torch.no_grad():
            if config.render_test:
                # render_test switches to test poses
                images = images[i_test]
            else:
                # Default is smoother render_poses path
                images = None

            testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('test' if config.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', self.render_poses.shape)

            rgbs, _ = render_path(self.render_poses, self.hwf, self.K, config.chunk, self.render_kwargs_test, gt_imgs=images, savedir=testsavedir, render_factor=config.render_factor)
            print('Done rendering', testsavedir)
            imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)

            return

    def train(self):
        '''
            Train Function!
        '''
        N_rand = config.N_rand
        use_batching = not config.no_batching
        if use_batching:
            # For random ray batching
            print('get rays')
            rays = np.stack([get_rays_np(H, W, K, p) for p in poses[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
            print('done, concats')
            rays_rgb = np.concatenate([rays, images[:,None]], 1) # [N, ro+rd+rgb, H, W, 3]
            rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
            rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0) # train images only
            rays_rgb = np.reshape(rays_rgb, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
            rays_rgb = rays_rgb.astype(np.float32)
            print('shuffle rays')
            np.random.shuffle(rays_rgb)

            print('done')
            i_batch = 0

        # Move training data to GPU
        if use_batching:
            images = torch.Tensor(images).to(config.device)
        poses = torch.Tensor(self.poses)
        if use_batching:
            rays_rgb = torch.Tensor(rays_rgb).to(config.device)

        N_iters = 200000 + 1
        print('Begin')
        print('TRAIN views are', self.i_train)
        print('TEST views are', self.i_test)
        print('VAL views are', self.i_val)
    
        self.start = self.start + 1
        for i in range(self.start, N_iters):
            time0 = time.time()

            # Sample random ray batch
            if use_batching:
                # Random over all images
                batch = rays_rgb[i_batch:i_batch+N_rand] # [B, 2+1, 3*?]
                batch = torch.transpose(batch, 0, 1)
                batch_rays, target_s = batch[:2], batch[2]

                i_batch += N_rand
                if i_batch >= rays_rgb.shape[0]:
                    print("Shuffle data after an epoch!")
                    rand_idx = torch.randperm(rays_rgb.shape[0])
                    rays_rgb = rays_rgb[rand_idx]
                    i_batch = 0

            else:
                # Random from one image
                img_i = np.random.choice(self.i_train)
                target = self.images[img_i]
                target = torch.Tensor(target).to(config.device)
                pose = poses[img_i, :3,:4]

                if N_rand is not None:
                    print (pose)
                    rays_o, rays_d = get_rays(self.H, self.W, self.K, pose)  # (H, W, 3), (H, W, 3)

                    if i < config.precrop_iters:
                        dH = int(self.H//2 * config.precrop_frac)
                        dW = int(self.W//2 * config.precrop_frac)
                        coords = torch.stack(
                            torch.meshgrid(
                                torch.linspace(self.H//2 - dH, self.H//2 + dH - 1, 2*dH), 
                                torch.linspace(self.W//2 - dW, self.W//2 + dW - 1, 2*dW)
                            ), -1)
                        if i == self.start:
                            print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")                
                    else:
                        coords = torch.stack(torch.meshgrid(torch.linspace(0, self.H-1, self.H), torch.linspace(0, self.W-1, self.W)), -1)  # (H, W, 2)

                    coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
                    select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
                    select_coords = coords[select_inds].long()  # (N_rand, 2)
                    rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                    rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                    batch_rays = torch.stack([rays_o, rays_d], 0)
                    target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

            batch_rays = batch_rays.to(config.device)
            #####  Core optimization loop  #####
            rgb, disp, acc, extras = render(self.H, self.W, self.K, chunk=config.chunk, rays=batch_rays,verbose=i < 10, retraw=True, **self.render_kwargs_train)

            self.optimizer.zero_grad()
            img_loss = img2mse(rgb, target_s)
            trans = extras['raw'][...,-1] # What is raw??
            loss = img_loss
            psnr = mse2psnr(img_loss)

            # TODO - What is rgb0?
            if 'rgb0' in extras:
                img_loss0 = img2mse(extras['rgb0'], target_s)
                loss = loss + img_loss0
                psnr0 = mse2psnr(img_loss0)

            loss.backward()
            self.optimizer.step()

            # NOTE: IMPORTANT! TODO - check global step is okay???
            ###   update learning rate   ###
            decay_rate = 0.1
            decay_steps = config.lrate_decay * 1000
            new_lrate = config.lrate * (decay_rate ** (self.global_step / decay_steps))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lrate

            print ("I am here!")
            exit()

            dt = time.time()-time0
            # print(f"Step: {global_step}, Loss: {loss}, Time: {dt}")
            #####           end            #####

            # Rest is logging
            if i%args.i_weights==0:
                path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
                torch.save({
                    'global_step': global_step,
                    'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                    'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, path)
                print('Saved checkpoints at', path)

            if i%args.i_video==0 and i > 0:
                # Turn on testing mode
                with torch.no_grad():
                    rgbs, disps = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test)
                print('Done, saving', rgbs.shape, disps.shape)
                moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
                imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
                imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)

                # if args.use_viewdirs:
                #     render_kwargs_test['c2w_staticcam'] = render_poses[0][:3,:4]
                #     with torch.no_grad():
                #         rgbs_still, _ = render_path(render_poses, hwf, args.chunk, render_kwargs_test)
                #     render_kwargs_test['c2w_staticcam'] = None
                #     imageio.mimwrite(moviebase + 'rgb_still.mp4', to8b(rgbs_still), fps=30, quality=8)

            if i%args.i_testset==0 and i > 0:
                testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
                os.makedirs(testsavedir, exist_ok=True)
                print('test poses shape', poses[i_test].shape)
                with torch.no_grad():
                    render_path(torch.Tensor(poses[i_test]).to(device), hwf, K, args.chunk, render_kwargs_test, gt_imgs=images[i_test], savedir=testsavedir)
                print('Saved test set')


        
            if i%args.i_print==0:
                tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")

if (__name__ == "__main__"):
    create_nerf()