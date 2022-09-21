import torch
import numpy as np

def loaddata(type='random1D', size=100):
    if type == 'random1D':
        return torch.randn(size)
    elif type == 'random2D':
        return torch.randn(size, size)
    elif type == 'sines':
        frequency1 = 2
        weight1 = 1
        frequency2 = 5
        weight2 = 1
        grid_x = torch.linspace(0, 2*np.pi, size)
        return torch.sin(weight1 * frequency1 * grid_x) + \
            torch.sin(weight2 * frequency2 * grid_x)
    elif type == 'cryoEM':
        import gemmi
        ccp4_map = gemmi.read_ccp4_map('../data/betagal2984.map')
        data = torch.Tensor(np.array(ccp4_map.grid))
        from globalbiopak.EMutils import generate_random_proj
        proj, _, _, _ = generate_random_proj(data, num_proj=1)
        return torch.reshape(proj, (proj.shape[0], proj.shape[2])) 
    elif type == 'MNIST':
        from torchvision.datasets import MNIST
        mnist = MNIST('../data/', download=True)
        return torch.from_numpy(np.array((mnist[0][0])))
    elif type == 'cameraman':
        from PIL import Image
        im = Image.open('../data/cameraman.tif')
        return torch.from_numpy(np.array(im))
    elif type == 'cells':
        from PIL import Image
        im = Image.open('../data/biological_cells.tif')
        return torch.from_numpy(np.array(im, dtype=np.int16))
