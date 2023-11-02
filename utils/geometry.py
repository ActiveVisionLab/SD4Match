import torch


def expand_size(coord, size):
    """
    Expand 'size' to the same dimension as coord, so it can perform per batch operation
    IN:
        coord [torch.Tensor] (B x ... x 2): (x, y) in src_scale
        size [Tuple(int) or torch.Tensor]: (H, W) or (Bx2)
    OUT:
        H [torch.Tensor] (B x 1 x .. x 1): the same number of dimension as coord
        W [torch.Tensor] (B x 1 x .. x 1): the same number of dimension as coord
    """

    B = coord.shape[0]
    ndim = len(coord.shape)
    _device = coord.device

    if isinstance(size, tuple):
        size = torch.tensor(list(size), device=_device)
        size = size[None, :].expand(B, -1)
    elif isinstance(size, torch.Tensor):
        pass

    H, W = size[:, 0], size[:, 1]

    for i in range(ndim-1):
        H = H[:, None]
        W = W[:, None]

    return H, W


def normalise_coordinates(coord, size):
    '''
    Normalise the coordinate to the range of (-1, 1)
    IN:
        coord [torch.Tensor] (B x ... x 2): in (x, y)
        size [Tuple(int) or torch.Tensor]: (H, W) or (Bx2)
    OUT:
        coord [torch.Tensor] (B x ... x 2): Normalised coordinate
    '''

    H, W = expand_size(coord, size)

    coord = coord.clone()
    coord_shape = coord.shape
    
    coord[..., 0:1] = coord[..., 0:1] / (W-1) * 2 - 1
    coord[..., 1:2] = coord[..., 1:2] / (H-1) * 2 - 1

    return coord


def unnormalise_coordinates(coord, size):
    '''
    Unnormlised the coordinate from the range of (-1, 1) to the given range (0, H) and (0, W)
    IN:
        coord [torch.Tensor] (B x ... x 2): in (x, y)
        size [Tuple(int) or torch.Tensor]: (H, W) or (Bx2)
    OUT:
        coord [torch.Tensor] (B x ... x 2): Unnormalised coordinate
    '''

    H, W = expand_size(coord, size)

    coord = coord.clone()
    coord_shape = coord.shape
        
    coord[..., 0:1] = (coord[..., 0:1] + 1) / 2 * (W-1)
    coord[..., 1:2] = (coord[..., 1:2] + 1) / 2 * (H-1)

    return coord


def scaling_coordinates(coord, src_scale, trg_scale, mode='align_corner'):
    '''
    Scale the coordinate from src_scale to trg_scale
    IN:
        coord [torch.Tensor] (B x ... x 2): (x, y) in src_scale
        src_scale [Tuple(int) or torch.Tensor]: (H1, W1) or (Bx2)
        trg_scale [Tuple(int) or torch.Tensor]: (H2, W2) or (Bx2)
        mode [str]: align_corner | center. If align_corner, four corners would always be mapped as corner. 
                    If center: coord at smaller scale is treated as the center of squared patch of larger scale
    OUT:
        coord [torch.Tensor] (B x ... x 2): (x, y) in trg_scale
    '''
    assert mode in ['align_corner', 'center'], 'mode has to be either align_corner or center but got %s' % mode

    H1, W1 = expand_size(coord, src_scale)
    H2, W2 = expand_size(coord, trg_scale)

    coord = coord.clone()
    
    if mode == 'align_corner':
        coord[..., 0:1] = coord[..., 0:1] / (W1-1) * (W2-1)
        coord[..., 1:2] = coord[..., 1:2] / (H1-1) * (H2-1)
    else:
        coord[..., 0:1] = (coord[..., 0:1] + 1/2) * W2 / W1 - 1/2
        coord[..., 1:2] = (coord[..., 1:2] + 1/2) * H2 / H1 - 1/2

    return coord


def regularise_coordinates(coord, H, W, eps=0):
    '''
    Squeeze coordinate into bounded image size (H, W) if some of coordinates is out of bound
    IN:
        coord [torch.Tensor] (B x ... x 2): in (x, y)
        H [int]
        W [int]
        eps [float] a small number to add an offset to regularise coordinate, default is 0
    OUT:
        coord [torch.Tensor] (B x ... x 2): in (x, y) any out of bound are pushed into the bound right on the edge
    '''

    coord = coord.clone()

    coord[..., 0][coord[..., 0] <= 0] = 0 + eps
    coord[..., 0][coord[..., 0] >= W-1] = W-1 - eps
    coord[..., 1][coord[..., 1] <= 0] = 0 + eps
    coord[..., 1][coord[..., 1] >= H-1] = H-1 - eps

    return coord


def create_grid(H, W, gap=1, device='cpu'):
    '''
    Create a unnormalised meshgrid in pytorch
    IN:
        H [=int]: height of the grid
        W [int]: width of the grid
        gap [int]: the gap between consecutive step
        device [str]: on which device to put the generated grid
    OUT:
        grid [torch.Tensor] (H x W x 2): generated meshgrid with coordinates in form of (x, y)
    '''
    x = torch.linspace(0, W-1, W//gap)
    y = torch.linspace(0, H-1, H//gap)
    yg, xg = torch.meshgrid(y, x, indexing="ij")
    grid = torch.stack((xg, yg), dim=2)

    return grid.to(device)