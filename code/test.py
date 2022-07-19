from numpy import *
from scipy.sparse import diags
from poisson import poisson

nt = 10

T = 2.0
n = 6
numranks = 3
for rank in range(numranks):

    # Declare time step size
    t0 = 0.0
    ht = (T - t0)/float(nt-1)

    # Declare spatial grid size.  Note that we divide by (n + 1) because we are
    # accounting for the boundary points, i.e., we really have n+2 total points
    h = 1.0 / (n+1.0)

    # Task in parallel:
    # Compute which portion of the spatial domain the current MPI rank owns,
    # i.e., compute "start", "end", "start_halo", and "end_halo"
    #
    #  - This will be similar to HW4.  Again, assume that n/nprocs divides evenly
    #
    #  - Because of the way that we are handling boundary domain rows
    #    (remember, we actually have n+2 domain rows), you may want to
    #    shift start and end up by "+1" when compared to HW4
    #
    #  - Lastly, just like with HW4, Cast start and end as integers, e.g.,
    #    start = int(....)
    #    end = int(....)
    #    start_halo = int(....)
    #    end_halo = int(....)

    start = int(rank*(n/numranks)) + 1
    end = int((rank+1)*(n/numranks)) + 1

    # Start halo
    if(rank == 0):
        start_halo = 1
    else:
        start_halo = start - 1

    # End Halo
    if(rank == (numranks - 1)):
        end_halo = end
    else:
        end_halo = end + 1

    # Remember, we assume a Dirichlet boundary condition, which simplifies
    # things a bit.  Thus, we only want a spatial grid from
    # [h, 2h, ..., 1-h] x [h, 2h, ..., 1-h].
    # We know what the solution looks like on the boundary, and don't need to solve for it.
    #
    # Task: fill in the right commands for the spatial grid vector "pts"
    # Task in parallel: Adjust these computations so that you only compute the local grid
    #                   plus halo region.  Mimic HW4 here.
    x_pts = linspace(h,1-h,n)
    #
    # y_pts contains all of the points in the y-direction for this thread's halo region
    # For the above example and thread 1 (k=1), this is y_pts = [0.33, 0.66, 1.0]
    b = (end_halo-1)*h
    a = start_halo*h
    y_pts = linspace(start_halo*h, (end_halo-1)*h, int(round((b-a)/h)) + 1)
    #print("rank ", rank, "ypts ", y_pts)


    X,Y = meshgrid(x_pts, y_pts)
    X = X.reshape(-1,)
    Y = Y.reshape(-1,)


    # Declare spatial discretization matrix
    # Task: what dimension should A be?  remember the spatial grid is from
    #       [h, 2h, ..., 1-h] x [h, 2h, ..., 1-h]
    #       Pass in the right size to poisson.
    # Task in parallel: Adjust the size of A, that is A will be just a processor's
    #                   local part of A, similar to HW4
    sizey = end_halo - start_halo
    sizex = n
    A = poisson((sizey, sizex), format='csr')

    print("rank ", rank, "A.shape", A.shape)
    print("rank ", rank, "A", A.todense())
    D = A.diagonal()
    print("rank ", rank, "d-1", diags(1.0/D, format='csr'))
