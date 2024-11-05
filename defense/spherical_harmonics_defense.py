import numpy as np
import pyshtools as pysh
import torch
import plotly.graph_objects as go
from cv2 import getGaussianKernel
from plotly.subplots import make_subplots

def plot_pc(pc, second_pc=None, s=4, o=0.6):
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]],)
    fig.add_trace(
        go.Scatter3d(x=pc[:, 0], y=pc[:, 1], z=pc[:, 2], mode='markers', marker=dict(size=s, opacity=o)),
        row=1, col=1
    )
    if second_pc is not None:
        fig.add_trace(
            go.Scatter3d(x=second_pc[:, 0], y=second_pc[:, 1], z=second_pc[:, 2], mode='markers', marker=dict(size=s, opacity=o)),
            row=1, col=2
        )
    fig.update_layout(scene_aspectmode='data')
    fig.show()
    
def convert_pc_to_grid(pc, lmax, device="cuda"):
    """ Function for projecting a point cloud onto the surface of the unit sphere centered at its centroid. """

    pc = torch.from_numpy(pc).to(device)

    grid = pysh.SHGrid.from_zeros(lmax, grid='DH')
    nlon = grid.nlon
    nlat = grid.nlat
    ngrid = nlon * nlat
    grid_lon = torch.from_numpy(np.linspace(0, nlon * np.pi * 2 / (nlon - 1), num=nlon, endpoint=False)).to(device)
    grid_lat = torch.from_numpy(np.linspace(nlat // 2 * np.pi / -(nlat - 1), np.pi / 2, num=nlat, endpoint=True)).to(device)
    grid_lon = torch.broadcast_to(grid_lon, (nlat, nlon))
    grid_lat = torch.broadcast_to(grid_lat.view(nlat, 1), (nlat, nlon))
    grid_lon = grid_lon.reshape(1, ngrid)
    grid_lat = grid_lat.reshape(1, ngrid)

    origin = torch.mean(pc, axis=0)  # the center of the unit sphere
    pc -= origin  # for looking from the perspective of the origin
    npc = len(pc)
    origin = origin.to("cpu").numpy()

    pc_x, pc_y, pc_z = pc[:, 0], pc[:, 1], pc[:, 2]

    pc_r = torch.sqrt(pc_x * pc_x + pc_y * pc_y + pc_z * pc_z)
    pc_lat = torch.arcsin(pc_z / pc_r)
    pc_lon = torch.atan2(pc_y, pc_x)
    pc_r = pc_r.view(npc, 1)
    pc_lat = pc_lat.view(npc, 1)
    pc_lon = pc_lon.view(npc, 1)

    dist = -torch.cos(grid_lat) * torch.cos(pc_lat) * torch.cos(grid_lon - pc_lon) + torch.sin(grid_lat) * torch.sin(pc_lat)

    argmin = torch.argmin(dist, axis=0)
    grid_r = pc_r[argmin].view(nlat, nlon)
    grid.data = grid_r.to("cpu").numpy()  # data of the projection onto the unit sphere

    argmin = torch.argmin(dist, axis=1)  # argmin on a different axis
    flag = torch.zeros(ngrid, dtype=bool)
    flag[argmin] = True  # indicates the polar angles for which the grid data can be interpreted as a point
    flag = flag.to("cpu").numpy()

    return grid, flag, origin
    
def convert_pc_to_grid_np(pc, lmax):
    """ Function for projecting a point cloud onto the surface of the unit sphere centered at its centroid. """

    pc = np.copy(pc)  # for not changing the original input point cloud

    grid = pysh.SHGrid.from_zeros(lmax, grid='DH')
    nlon = grid.nlon
    nlat = grid.nlat
    ngrid = nlon * nlat

    grid_lon = np.linspace(0, nlon * np.pi * 2 / (nlon - 1), num=nlon, endpoint=False)
    grid_lat = np.linspace(nlat // 2 * np.pi / -(nlat - 1), np.pi / 2, num=nlat, endpoint=True)

    grid_lon = np.broadcast_to(grid_lon, (nlat, nlon))
    grid_lat = np.broadcast_to(grid_lat.reshape((nlat, 1)), (nlat, nlon))

    grid_lon = grid_lon.reshape((1, ngrid))
    grid_lat = grid_lat.reshape((1, ngrid))

    origin = np.average(pc, axis=0)  # the center of the unit sphere
    pc -= origin  # for looking from the perspective of the origin
    npc = len(pc)

    pc_x, pc_y, pc_z = pc[:, 0], pc[:, 1], pc[:, 2]

    pc_r = np.sqrt(pc_x * pc_x + pc_y * pc_y + pc_z * pc_z)
    pc_lat = np.arcsin(pc_z / pc_r)
    pc_lon = np.arctan2(pc_y, pc_x)

    pc_r = pc_r.reshape((npc, 1))
    pc_lat = pc_lat.reshape((npc, 1))
    pc_lon = pc_lon.reshape((npc, 1))

    dist = -np.cos(grid_lat) * np.cos(pc_lat) * np.cos(grid_lon - pc_lon) + np.sin(grid_lat) * np.sin(pc_lat)

    argmin = np.argmin(dist, axis=0)
    grid_r = pc_r[argmin].reshape((nlat, nlon))
    grid.data = grid_r  # data of the projection onto the unit sphere

    argmin = np.argmin(dist, axis=1)  # argmin on a different axis
    flag = np.zeros(ngrid, dtype=bool)
    flag[argmin] = True  # indicates the polar angles for which the grid data can be interpreted as a point

    return grid, flag, origin
    
def convert_grid_to_pc(grid, flag, origin):
    """ Function for reconstructing a point cloud from its projection onto the unit sphere. """

    nlon = grid.nlon
    nlat = grid.nlat
    lon = np.linspace(0, nlon * np.pi * 2 / (nlon - 1), num=nlon, endpoint=False)
    lat = np.linspace(nlat // 2 * np.pi / -(nlat - 1), np.pi / 2, num=nlat, endpoint=True)
    lon = np.broadcast_to(lon.reshape((1, nlon)), (nlat, nlon))
    lat = np.broadcast_to(lat.reshape((nlat, 1)), (nlat, nlon))
    r = grid.data

    z = np.sin(lat) * r
    t = np.cos(lat) * r
    x = t * np.cos(lon)
    y = t * np.sin(lon)

    pc = np.zeros(grid.data.shape + (3, ))
    pc[:, :, 0] = x
    pc[:, :, 1] = y
    pc[:, :, 2] = -z  # must have the minus
    pc = pc.reshape((-1, 3))
    pc = pc[flag, :]  # only the flagged polar angles must be used in the point cloud reconstruction
    pc += origin  # translate to the original origin

    return pc
    
def low_pass_filter(grid, sigma):
    ''' Function for diminishing high frequency components in the spherical harmonics representation. '''

    # transform to the frequency domain
    clm = grid.expand()

    # create filter weights
    weights = getGaussianKernel(clm.coeffs.shape[1] * 2 - 1, sigma)[clm.coeffs.shape[1] - 1:]
    weights /= weights[0]

    # low-pass filtering
    clm.coeffs *= weights

    # transform back into spatial domain
    low_passed_grid = clm.expand()

    return low_passed_grid
    
def duplicate_randomly(pc, size):  
    """ Make up for the point loss due to conflictions in the projection process. """
    loss_cnt = size - pc.shape[0]
    if loss_cnt <= 0:
        return pc
    rand_indices = np.random.randint(0, pc.shape[0], size=loss_cnt)
    dup = pc[rand_indices]
    return np.concatenate((pc, dup))
    
def our_method(pc, lmax, sigma, pc_size=1024, device="cuda"):
    grid, flag, origin = convert_pc_to_grid(pc, lmax, device)
    smooth_grid = low_pass_filter(grid, sigma)
    smooth_pc = convert_grid_to_pc(smooth_grid, flag, origin)
    smooth_pc = duplicate_randomly(smooth_pc, pc_size)
    return smooth_pc
