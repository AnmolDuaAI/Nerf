import numpy as np
import os, imageio


def normalize(x):
    return x / np.linalg.norm(x)

def ptstocam(pts, c2w):
    tt = np.matmul(c2w[:3,:3].T, (pts-c2w[:3,3])[...,np.newaxis])[...,0]
    return tt

def spherify_poses(poses, bds):
    
    p34_to_44 = lambda p : np.concatenate([p, np.tile(np.reshape(np.eye(4)[-1,:], [1,1,4]), [p.shape[0], 1,1])], 1)
    
    rays_d = poses[:,:3,2:3]
    rays_o = poses[:,:3,3:4]

    def min_line_dist(rays_o, rays_d):
        A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0,2,1])
        b_i = -A_i @ rays_o
        pt_mindist = np.squeeze(-np.linalg.inv((np.transpose(A_i, [0,2,1]) @ A_i).mean(0)) @ (b_i).mean(0))
        return pt_mindist

    pt_mindist = min_line_dist(rays_o, rays_d)
    
    center = pt_mindist
    up = (poses[:,:3,3] - center).mean(0)

    vec0 = normalize(up)
    vec1 = normalize(np.cross([.1,.2,.3], vec0))
    vec2 = normalize(np.cross(vec0, vec1))
    pos = center
    c2w = np.stack([vec1, vec2, vec0, pos], 1)

    poses_reset = np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:,:3,:4])

    rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:,:3,3]), -1)))
    
    sc = 1./rad
    poses_reset[:,:3,3] *= sc
    bds *= sc
    rad *= sc
    
    centroid = np.mean(poses_reset[:,:3,3], 0)
    zh = centroid[2]
    radcircle = np.sqrt(rad**2-zh**2)
    new_poses = []
    
    for th in np.linspace(0.,2.*np.pi, 120):

        camorigin = np.array([radcircle * np.cos(th), radcircle * np.sin(th), zh])
        up = np.array([0,0,-1.])

        vec2 = normalize(camorigin)
        vec0 = normalize(np.cross(vec2, up))
        vec1 = normalize(np.cross(vec2, vec0))
        pos = camorigin
        p = np.stack([vec0, vec1, vec2, pos], 1)

        new_poses.append(p)

    new_poses = np.stack(new_poses, 0)
    
    new_poses = np.concatenate([new_poses, np.broadcast_to(poses[0,:3,-1:], new_poses[:,:3,-1:].shape)], -1)
    poses_reset = np.concatenate([poses_reset[:,:3,:4], np.broadcast_to(poses[0,:3,-1:], poses_reset[:,:3,-1:].shape)], -1)
    
    return poses_reset, new_poses, bds

def viewmatrix(z, up, pos):
    vec2 = normalize(z) # Normalizing the z-axis
    vec1_avg = up # y-axis
    vec0 = normalize(np.cross(vec1_avg, vec2)) # x-axis
    vec1 = normalize(np.cross(vec2, vec0)) # y-axis
    m = np.stack([vec0, vec1, vec2, pos], 1) # (x,y,z,center)
    return m

def ptstocam(pts, c2w):
    tt = np.matmul(c2w[:3,:3].T, (pts-c2w[:3,3])[...,np.newaxis])[...,0]
    return tt

def poses_avg(poses):

    hwf = poses[0, :3, -1:] # Getting height, width, focal length that is col4 (col0,col1,col2,col3,col4) from 3 x 5 matrix

    center = poses[:, :3, 3].mean(0) # Getting mean of all poses transalation components(column)
    vec2 = normalize(poses[:, :3, 2].sum(0)) # Getting normalized of total sum of z-axis component of rotation matrix
    up = poses[:, :3, 1].sum(0) # Getting total sum of y-axis (upwards direction) component of rotation matrix
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1) # (x,y,z,center,hwf) => pose_matrix(center to world)
    
    return c2w


def recenter_poses(poses):
    # poses - 20 x 3 x 5
    poses_ = poses+0
    bottom = np.reshape([0,0,0,1.], [1,4]) # 1 x 4
    c2w = poses_avg(poses) # 3x5 matrix - center to world pose matrix
    c2w = np.concatenate([c2w[:3,:4], bottom], -2) # Homogeneous 4 x 4 Matrix (Rotation+Translation)
    bottom = np.tile(np.reshape(bottom, [1,1,4]), [poses.shape[0],1,1]) # 20 x 1 x 4 matrix
    poses = np.concatenate([poses[:,:3,:4], bottom], -2)# Poses of camera images (Homogenous) - 20 x 4 x 4

    poses = np.linalg.inv(c2w) @ poses # Poses of images w.r.t center  = w2c @ poses(w.r.t w)
    poses_[:,:3,:4] = poses[:,:3,:4] # New Poses w.r.t center (rotation, translation component)
    poses = poses_ # Including hwf
    return poses

def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    render_poses = []
    rads = np.array(list(rads) + [1.]) # Radius - 1 x 4(including 1) - 90% percentile of poses translation component
    hwf = c2w[:,4:5] # height,width,focal length

    # Iteration theta from 0 - 2*pi*number of rotations with steps of number of views(120)
    for theta in np.linspace(0., 2. * np.pi * rots, N+1)[:-1]:
        c = np.dot(c2w[:3,:4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta*zrate), 1.]) * rads) 
        z = normalize(c - np.dot(c2w[:3,:4], np.array([0,0,-focal, 1.])))
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
    return render_poses


class DataReader:
    def __init__(self, basedir):
        self.basedir = basedir
        self.factor = 8
        self.recenter = True
        self.bd_factor = 0.75
        self.spherify = False
        path_zflat = False

        # Poses - 3 x 5 x 20
        # Bds - 2 x 20
        # Imgs - 378 x 504 x 3 x 20
        poses, bds, imgs = self._load_data(basedir, factor=self.factor) # factor=8 downsamples original imgs by 8x
        print('Loaded', basedir, bds.min(), bds.max())

        # Correct rotation matrix ordering and move variable dim to axis 0
        poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1) # col0,col1,col2,col3,col4 -> col1, col0, col2, col3, col4 # Rotation Matrix ordering changed
        poses = np.moveaxis(poses, -1, 0).astype(np.float32) # 20 x 3 x 5
        imgs = np.moveaxis(imgs, -1, 0).astype(np.float32) # 20 x 378 x 504 x 3
        images = imgs
        bds = np.moveaxis(bds, -1, 0).astype(np.float32) # 20 x 2

        # Rescale if bd_factor is provided
        sc = 1. if self.bd_factor is None else 1./(bds.min() * self.bd_factor)
        poses[:,:3,3] *= sc # Multiplying translation component by scaling factor
        bds *= sc # Rescaling bds
        
        if self.recenter:
            poses = recenter_poses(poses) # 20 x 3 x 5 - poses changed from w.r.t world to w.r.t center
            
        if self.spherify:
            poses, render_poses, bds = spherify_poses(poses, bds)

        else:
            c2w = poses_avg(poses) # Calculating avg poses center (3 x 5)
            print('recentered', c2w.shape)
            print(c2w[:3,:4])

            ## Get spiral
            # Get average pose
            up = normalize(poses[:, :3, 1].sum(0)) # Normalized y-axis component of all poses

            # Find a reasonable "focus depth" for this dataset
            close_depth, inf_depth = bds.min()*.9, bds.max()*5. # Setting close depth = 0.9 x min bound and inf depth = 5 x max bound
            print ("Close Depth : " + str(close_depth))
            print ("Inf Depth : " + str(inf_depth))
            dt = .75
            mean_dz = 1./(((1.-dt)/close_depth + dt/inf_depth)) # Mean depth = 1/(dt * close_disparity + (1-dt) * inf_disparity) , close_disparity = 1/inf_depth , inf_disparity = 1/close_depth
            focal = mean_dz # Setting focal radii
            print ("Focal : " + str(focal))

            # Get radii for spiral path
            shrink_factor = .8
            zdelta = close_depth * .2
            tt = poses[:,:3,3] # ptstocam(poses[:3,3,:].T, c2w).T
            rads = np.percentile(np.abs(tt), 90, 0)
            c2w_path = c2w
            N_views = 120
            N_rots = 2
            if path_zflat: 
                zloc = -close_depth * .1
                c2w_path[:3,3] = c2w_path[:3,3] + zloc * c2w_path[:3,2]
                rads[2] = 0.
                N_rots = 1
                N_views/=2

            # Generate poses for spiral path
            render_poses = render_path_spiral(c2w_path, up, rads, focal, zdelta, zrate=.5, rots=N_rots, N=N_views)
            
            
        render_poses = np.array(render_poses).astype(np.float32)

        c2w = poses_avg(poses)
        print('Data:')
        print(poses.shape, images.shape, bds.shape)
        
        dists = np.sum(np.square(c2w[:3,3] - poses[:,:3,3]), -1)
        i_test = np.argmin(dists)
        print('HOLDOUT view is', i_test)
        
        images = images.astype(np.float32)
        poses = poses.astype(np.float32)

        self.images = images
        self.poses = poses
        self.bds = bds
        self.render_poses = render_poses
        self.i_test = i_test

    def get_data(self):
        return self.images, self.poses, self.bds, self.render_poses, self.i_test

    def _minify(self, basedir, factors=[], resolutions=[]):
        needtoload = False
        for r in factors:
            imgdir = os.path.join(basedir, 'images_{}'.format(r))
            if not os.path.exists(imgdir):
                needtoload = True
        for r in resolutions:
            imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
            if not os.path.exists(imgdir):
                needtoload = True
        if not needtoload:
            return
        
        from shutil import copy
        from subprocess import check_output
        
        imgdir = os.path.join(basedir, 'images')
        imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
        imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
        imgdir_orig = imgdir
        
        wd = os.getcwd()

        for r in factors + resolutions:
            if isinstance(r, int):
                name = 'images_{}'.format(r)
                resizearg = '{}%'.format(100./r)
            else:
                name = 'images_{}x{}'.format(r[1], r[0])
                resizearg = '{}x{}'.format(r[1], r[0])
            imgdir = os.path.join(basedir, name)
            if os.path.exists(imgdir):
                continue
                
            print('Minifying', r, basedir)
            
            os.makedirs(imgdir)
            check_output('cp {}/* {}'.format(imgdir_orig, imgdir), shell=True)
            
            ext = imgs[0].split('.')[-1]
            args = ' '.join(['mogrify', '-resize', resizearg, '-format', 'png', '*.{}'.format(ext)])
            print(args)
            os.chdir(imgdir)
            check_output(args, shell=True)
            os.chdir(wd)
            
            if ext != 'png':
                check_output('rm {}/*.{}'.format(imgdir, ext), shell=True)
                print('Removed duplicates')
            print('Done')

    def _load_data(self, basedir, factor=None, width=None, height=None, load_imgs=True):
        
        poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy')) # 20 x 17
        poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1,2,0]) # 20 x 15 -> 20 x 3 x 5 -> 3 x 5 x 20
        bds = poses_arr[:, -2:].transpose([1,0]) # 20 x 2 -> 2 x 20
        
        img0 = [os.path.join(basedir, 'images', f) for f in sorted(os.listdir(os.path.join(basedir, 'images'))) \
                if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]
        sh = imageio.imread(img0).shape
        
        sfx = ''
        
        if factor is not None:
            sfx = '_{}'.format(factor)
            self._minify(basedir, factors=[factor])
            factor = factor
        elif height is not None:
            factor = sh[0] / float(height)
            width = int(sh[1] / factor)
            self._minify(basedir, resolutions=[[height, width]])
            sfx = '_{}x{}'.format(width, height)
        elif width is not None:
            factor = sh[1] / float(width)
            height = int(sh[0] / factor) 
            self._minify(basedir, resolutions=[[height, width]])
            sfx = '_{}x{}'.format(width, height)
        else:
            factor = 1
        
        imgdir = os.path.join(basedir, 'images' + sfx)
        if not os.path.exists(imgdir):
            print( imgdir, 'does not exist, returning' )
            return
        
        imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
        if poses.shape[-1] != len(imgfiles):
            print( 'Mismatch between imgs {} and poses {} !!!!'.format(len(imgfiles), poses.shape[-1]) )
            return
        
        sh = imageio.imread(imgfiles[0]).shape

        poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1]) # Setting height and width of new image into pose matrix
        poses[2, 4, :] = poses[2, 4, :] * 1./factor # Focal Length with be changed by 1/factor

        # if not load_imgs:
        #     return poses, bds
        
        def imread(f):
            if f.endswith('png'):
                return imageio.imread(f, ignoregamma=True)
            else:
                return imageio.imread(f)
            
        imgs = imgs = [imread(f)[...,:3]/255. for f in imgfiles]
        imgs = np.stack(imgs, -1)  
        
        print('Loaded image data', imgs.shape, poses[:,-1,0])
        return poses, bds, imgs    
            
            
    

def Test():
    basedir = "/home/user/anmol/Nerf/nerf-pytorch/data/nerf_llff_data/fern/"
    data = DataReader(basedir)

if (__name__ == "__main__"):
    Test()

