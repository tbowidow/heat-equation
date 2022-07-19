"""
These functions taken from PyAMG 
https://github.com/pyamg/pyamg/
PyAMG Authors: Luke Olson, Jacob Schroder, Nathan Bell
"""
import numpy as np
import scipy.sparse as sparse


def poisson(grid, spacing=None, dtype=float, format=None, type='FD'):
    """Return a sparse matrix for the N-dimensional Poisson problem.
    The matrix represents a finite Difference approximation to the
    Poisson problem on a regular n-dimensional grid with unit grid
    spacing and Dirichlet boundary conditions.
    Parameters
    ----------
    grid : tuple of integers
        grid dimensions e.g. (100,100)
    Notes
    -----
    The matrix is symmetric and positive definite (SPD).
    Examples
    --------
    >>> from pyamg.gallery import poisson
    >>> # 4 nodes in one dimension
    >>> poisson( (4,) ).todense()
    matrix([[ 2., -1.,  0.,  0.],
            [-1.,  2., -1.,  0.],
            [ 0., -1.,  2., -1.],
            [ 0.,  0., -1.,  2.]])
    >>> # rectangular two dimensional grid
    >>> poisson( (2,3) ).todense()
    matrix([[ 4., -1.,  0., -1.,  0.,  0.],
            [-1.,  4., -1.,  0., -1.,  0.],
            [ 0., -1.,  4.,  0.,  0., -1.],
            [-1.,  0.,  0.,  4., -1.,  0.],
            [ 0., -1.,  0., -1.,  4., -1.],
            [ 0.,  0., -1.,  0., -1.,  4.]])
    """
    grid = tuple(grid)

    N = len(grid)  # grid dimension

    if N < 1 or min(grid) < 1:
        raise ValueError('invalid grid shape: %s' % str(grid))

    # create N-dimension Laplacian stencil
    if type == 'FD':
        stencil = np.zeros((3,) * N, dtype=dtype)
        for i in range(N):
            stencil[(1,)*i + (0,) + (1,)*(N-i-1)] = -1
            stencil[(1,)*i + (2,) + (1,)*(N-i-1)] = -1
        stencil[(1,)*N] = 2*N

    if type == 'FE':
        stencil = -np.ones((3,) * N, dtype=dtype)
        stencil[(1,)*N] = 3**N - 1

    return -1.0*stencil_grid(stencil, grid, format=format)



def stencil_grid(S, grid, dtype=None, format=None):
    """Construct a sparse matrix form a local matrix stencil.
    Parameters
    ----------
    S : ndarray
        matrix stencil stored in N-d array
    grid : tuple
        tuple containing the N grid dimensions
    dtype :
        data type of the result
    format : string
        sparse matrix format to return, e.g. "csr", "coo", etc.
    Returns
    -------
    A : sparse matrix
        Sparse matrix which represents the operator given by applying
        stencil S at each vertex of a regular grid with given dimensions.
    Notes
    -----
    The grid vertices are enumerated as arange(prod(grid)).reshape(grid).
    This implies that the last grid dimension cycles fastest, while the
    first dimension cycles slowest.  For example, if grid=(2,3) then the
    grid vertices are ordered as (0,0), (0,1), (0,2), (1,0), (1,1), (1,2).
    This coincides with the ordering used by the NumPy functions
    ndenumerate() and mgrid().
    Examples
    --------
    >>> from pyamg.gallery import stencil_grid
    >>> stencil = [-1,2,-1]  # 1D Poisson stencil
    >>> grid = (5,)          # 1D grid with 5 vertices
    >>> A = stencil_grid(stencil, grid, dtype=float, format='csr')
    >>> A.todense()
    matrix([[ 2., -1.,  0.,  0.,  0.],
            [-1.,  2., -1.,  0.,  0.],
            [ 0., -1.,  2., -1.,  0.],
            [ 0.,  0., -1.,  2., -1.],
            [ 0.,  0.,  0., -1.,  2.]])
    >>> stencil = [[0,-1,0],[-1,4,-1],[0,-1,0]] # 2D Poisson stencil
    >>> grid = (3,3)                            # 2D grid with shape 3x3
    >>> A = stencil_grid(stencil, grid, dtype=float, format='csr')
    >>> A.todense()
    matrix([[ 4., -1.,  0., -1.,  0.,  0.,  0.,  0.,  0.],
            [-1.,  4., -1.,  0., -1.,  0.,  0.,  0.,  0.],
            [ 0., -1.,  4.,  0.,  0., -1.,  0.,  0.,  0.],
            [-1.,  0.,  0.,  4., -1.,  0., -1.,  0.,  0.],
            [ 0., -1.,  0., -1.,  4., -1.,  0., -1.,  0.],
            [ 0.,  0., -1.,  0., -1.,  4.,  0.,  0., -1.],
            [ 0.,  0.,  0., -1.,  0.,  0.,  4., -1.,  0.],
            [ 0.,  0.,  0.,  0., -1.,  0., -1.,  4., -1.],
            [ 0.,  0.,  0.,  0.,  0., -1.,  0., -1.,  4.]])
    """
    S = np.asarray(S, dtype=dtype)
    grid = tuple(grid)

    if not (np.asarray(S.shape) % 2 == 1).all():
        raise ValueError('all stencil dimensions must be odd')

    if len(grid) != np.ndim(S):
        raise ValueError('stencil dimension must equal number of grid\
                          dimensions')

    if min(grid) < 1:
        raise ValueError('grid dimensions must be positive')

    N_v = np.prod(grid)  # number of vertices in the mesh
    N_s = (S != 0).sum()    # number of nonzero stencil entries

    # diagonal offsets
    diags = np.zeros(N_s, dtype=int)

    # compute index offset of each dof within the stencil
    strides = np.cumprod([1] + list(reversed(grid)))[:-1]
    indices = tuple(i.copy() for i in S.nonzero())
    for i, s in zip(indices, S.shape):
        i -= s // 2
        # i = (i - s) // 2
        # i = i // 2
        # i = i - (s // 2)
    for stride, coords in zip(strides, reversed(indices)):
        diags += stride * coords

    data = S[S != 0].repeat(N_v).reshape(N_s, N_v)

    indices = np.vstack(indices).T

    # zero boundary connections
    for index, diag in zip(indices, data):
        diag = diag.reshape(grid)
        for n, i in enumerate(index):
            if i > 0:
                s = [slice(None)] * len(grid)
                s[n] = slice(0, i)
                s = tuple(s)
                diag[s] = 0
            elif i < 0:
                s = [slice(None)]*len(grid)
                s[n] = slice(i, None)
                s = tuple(s)
                diag[s] = 0

    # remove diagonals that lie outside matrix
    mask = abs(diags) < N_v
    if not mask.all():
        diags = diags[mask]
        data = data[mask]

    # sum duplicate diagonals
    if len(np.unique(diags)) != len(diags):
        new_diags = np.unique(diags)
        new_data = np.zeros((len(new_diags), data.shape[1]),
                            dtype=data.dtype)

        for dia, dat in zip(diags, data):
            n = np.searchsorted(new_diags, dia)
            new_data[n, :] += dat

        diags = new_diags
        data = new_data

    return sparse.dia_matrix((data, diags),
                             shape=(N_v, N_v)).asformat(format)

