from numpy import *
from matplotlib import pyplot
from mpi4py import MPI
from poisson import poisson
from pdb import set_trace

from scipy.sparse import diags
# speye generates a sparse identity matrix
from scipy.sparse import eye as speye

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
numranks = comm.size

'''
    # Problem Preliminary: MPI cheat sheet

    # For the later parallel part, you'll need to use MPI.
    # Here are the most useful commands.

    # Import MPI at start of program
    from mpi4py import MPI

    # Initialize MPI
    comm = MPI.COMM_WORLD

    # Get your MPI rank
    rank = comm.Get_rank()

    # Send "data" (an array of doubles) to rank 1 from rank 0
    comm.Send([data, MPI.DOUBLE], dest=1, tag=77)

    # Receive "data" (the array of doubles) from rank 0 (on rank 1)
    comm.Recv([data, MPI.DOUBLE], source=0, tag=77)

    # Carry out an all-reduce, to sum over values collected from all processors
    # Note: "part_norm" and "global_norm" are length (1,) arrays.  The result
    #        of the global sum will reside in global_norm.
    # Note: If MPI.SUM is changed, then the reduce can multiple, subtract, etc.
    comm.Allreduce(part_norm, global_norm, op=MPI.SUM)

    # For instance, a simple Allreduce program is the following.
    # The result in global_norm will be the total number of processors
    from scipy import *
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    part_norm = array([1.0])
    global_norm = zeros_like(part_norm)

    comm.Allreduce(part_norm, global_norm, op=MPI.SUM)

    if (rank == 0):
        print(global_norm)
'''


##
# Problem Definition
# Task: figure out (1) the exact solution, (2) f(t,x,y), and (3) g(t,x,y)
#
# Governing PDE:
# u_t = u_xx + u_yy + f,    on the unit box [0,1] x [0,1] and t in a user-defined interval
#
# with an exact solution of
# u(t,x,y) = ...
#
# which in turn, implies a forcing term of f, where
# f(t,x,y) = ...
#
# an initial condition of
# u(t=0,x,y) = ...
#
# and a boundary condition in space when (x,y) is on the boundary of
# g(t,x,y) = u(t,x,y) = ...
#
##

# Declare the problem
def uexact(t,x,y):
    a = 0.25
    b = 0.50
    c = 0.75
    uexact = sin(pi*t+a)*sin(pi*x+b)*sin(pi*y+c)
    return uexact

def f(t,x,y):
    # Forcing term
    # This should equal u_t - u_xx - u_yy
    a = 0.25
    b = 0.50
    c = 0.75

    # Task: fill in forcing term
    forcing = pi*cos(pi*t+a)*sin(pi*x+b)*sin(pi*y+c) + 2*(pi**2)*sin(pi*t+a)*sin(pi*x+b)*sin(pi*y+c)
    return forcing

##
# Task in serial: Implement Jacobi

# Task in parallel: Extend Jacobi to parallel (as described below), with a
#                   parallel matrix-vector product and parallel vector norm.
#                   It is suggested to write separate subroutines for the
#                   parallel norm and the parallel matrix-vector product, as
#                   this will make your code much, much easier to debug.
#
#                   For instance, the instructor wrote a routine
#                   "matrix_vector()" which computes A*x with an interface
#                          matrix_vector(A, x, N, comm)
#                   where A is the matrix, x is the vector to multiply,
#                   N is the number of domain rows (excluding boundary rows)
#                   (like 8 for a 8x8 grid), and comm is the MPI communicator.
##
def jacobi(A, b, x0, tol, maxiter, n):
    '''
    Carry out the Jacobi method to invert A

    Input
    -----
    A <CSR matrix>  : Matrix to invert
    b <array>       : Right hand side
    x0 <array>      : Initial solution guess

    Output
    ------
    x <array>       : Solution to A x = b
    '''

    # This useful function returns an array containing diag(A)
    D = A.diagonal()

    # compute initial residual norm
    if (numranks != 1):
         if (rank == 0):
             comm.Send([x0[-(2*n):-n], MPI.DOUBLE], dest=1, tag=77)
         elif (rank == (numranks - 1)):
             comm.Send([x0[n:2*n], MPI.DOUBLE], dest=rank-1, tag=77)
         else:
             comm.Send([x0[-(2*n):-n], MPI.DOUBLE], dest=rank+1, tag=77)
             comm.Send([x0[n:2*n], MPI.DOUBLE], dest=rank-1, tag=77)
           
         if (rank == 0):
             comm.Recv([x0[-n:], MPI.DOUBLE], source=rank+1, tag=77)
         elif (rank == (numranks - 1)):
             comm.Recv([x0[0:n], MPI.DOUBLE], source=rank-1, tag=77)
         else:
             comm.Recv([x0[0:n], MPI.DOUBLE], source=rank-1, tag=77)
             comm.Recv([x0[-n:], MPI.DOUBLE], source=rank+1, tag=77)
    sol0 = A*x0

    r0 = ravel(b - sol0)
    if (rank == 0):
        if (numranks != 1):
            r0 = r0[0:-n]
        else:
            r0 = r0
    elif (rank == (numranks - 1)):
        if (numranks != 1):
            r0 = r0[n:]
    else:
        r0 = r0[n:-n]
        
    #print('r0 ', r0)
    r0r0 = array([dot(r0, r0)])
    #print("r0r0 ", r0r0)
    globalr0r0 = array([0.0])
    comm.Allreduce(r0r0, globalr0r0, op=MPI.SUM)
    globalr0 = array([0.0])
    globalr0[0] = sqrt(globalr0r0[0])

    # Max number of iterations
    # maxiter = 250

    # Generates the identity matrix
    I = speye(A.shape[0], format='csr')

    # Generates the D inverse
    Dinv = diags(1.0/D, format='csr')

    # Generates D inverse * b
    Db = Dinv*b

    # set_trace()
    # Generates the (I-Dinv*G)
    Iterm = I - Dinv*A

    x = x0

    # Start Jacobi iterations
    # Task in serial: implement Jacobi method and halting tolerance based on the residual norm
    for k in range(maxiter):
        #print("rank ", rank, "x befOre ", x)
        if (numranks != 1):
            if (rank == 0):
                comm.Send([x[-(2*n):-n], MPI.DOUBLE], dest=1, tag=77)
            elif (rank == (numranks - 1)):
                comm.Send([x[n:2*n], MPI.DOUBLE], dest=rank-1, tag=77)
            else:
                comm.Send([x[-(2*n):-n], MPI.DOUBLE], dest=rank+1, tag=77)
                comm.Send([x[n:2*n], MPI.DOUBLE], dest=rank-1, tag=77)
           
            if (rank == 0):
                comm.Recv([x[-n:], MPI.DOUBLE], source=rank+1, tag=77)
            elif (rank == (numranks - 1)):
                comm.Recv([x[0:n], MPI.DOUBLE], source=rank-1, tag=77)
            else:
                comm.Recv([x[0:n], MPI.DOUBLE], source=rank-1, tag=77)
                comm.Recv([x[-n:], MPI.DOUBLE], source=rank+1, tag=77)
        
        #print("rank ", rank, "x after ", x)
        
        x = Iterm*x + Db

        if (numranks != 1):
            if (rank == 0):
                comm.Send([x[-(2*n):-n], MPI.DOUBLE], dest=1, tag=77)
            elif (rank == (numranks - 1)):
                comm.Send([x[n:2*n], MPI.DOUBLE], dest=rank-1, tag=77)
            else:
                comm.Send([x[-(2*n):-n], MPI.DOUBLE], dest=rank+1, tag=77)
                comm.Send([x[n:2*n], MPI.DOUBLE], dest=rank-1, tag=77)
           
            if (rank == 0):
                comm.Recv([x[-n:], MPI.DOUBLE], source=rank+1, tag=77)
            elif (rank == (numranks - 1)):
                comm.Recv([x[0:n], MPI.DOUBLE], source=rank-1, tag=77)
            else:
                comm.Recv([x[0:n], MPI.DOUBLE], source=rank-1, tag=77)
                comm.Recv([x[-n:], MPI.DOUBLE], source=rank+1, tag=77)

        sol = A*x
        #print("local b ", local_b)
        #print("local sol ", local_sol)
        rk = ravel(b - sol)
        if (rank == 0):
            if (numranks != 1):
                local_rk = rk[0:-n]
            else:
                local_rk = rk
        elif (rank == (numranks - 1)):
            if (numranks != 1):
                local_rk = rk[n:]
        else:
            local_rk = rk[n:-n]

        rkrk = array([dot(local_rk, local_rk)])
        globalrkrk = array([0.0])
        comm.Allreduce(rkrk, globalrkrk, op=MPI.SUM)
        globalrk = array([0.0])
        globalrk[0] = sqrt(globalrkrk[0])

        residual = (globalrk[0])/(globalr0[0])
        #print("The current rk residual is ", globalrk[0])
        #print("The current r0 residual is ", globalr0[0])

	
        #if (rank == 0):
            #print(str(globalrk[0]))
	
        if (k == maxiter - 1 and residual > tol):
            #print('rkrk  ', rkrk)
            #print("local rk  ", local_rk)
            #print("b ", b, " sol ", sol)
            if (rank == 0):
                print("Jacobi did not converge at K=", k)
        if (residual <= tol):
            if (rank == 0):
                print("Jacobi converged at K: ", k)
            #print("rkrk ", rkrk)
            #print("local rk  ", local_rk)
            #print("b ", b, " sol ", sol)
            break

    # Task in parallel: extend the matrix-vector multiply to the parallel setting.
    #                   Additionally, you'll need to compute a norm of the residual in parallel.

    # Work on later
    # for i in range(maxiter):
        # << Jacobi algorithm goes here >>


    # Task: Print if Jacobi did not converge. In parallel, only rank 0 should print.
    #set_trace()
    return x


def euler_backward(A, u, ht, f, g, n):
    '''
    Carry out backward Euler for one time step

    Input
    -----
    A <CSR matrix>  : Discretization matrix of Poisson operator
    u <array>       : Current solution vector at previous time step
    ht <scalar>     : Time step size
    f <array>       : Current forcing vector
    g <array>       : Current vector containing boundary condition information

    Output
    ------
    u at the next time step

    '''

    # Task: Form the system matrix for backward Euler
    I = speye(A.shape[0], format='csr')
    G = I - ht*A
    #set_trace()

    # Check the g and f values (gi+1, fi+1)
    b = u + (ht*g) + (ht*f)
    tol = 1e-6
    maxiter = 250

    # Task: return solution from Jacobi, which takes a time-step forward in time by "ht"
    return jacobi(G, b, u, tol, maxiter, n)


# Helper function provided by instructor for debugging.  See how matvec_check
# is used below.
def matvec_check( A, X, Y, N, comm):
    '''
    This function runs

       (h**2)*A*ones()

    which should yield an output that is zero for interior points,
    -1 for points next to a Wall, and -2 for the four corner points

    All the results are printed to the screen.

    Further, it is assumed that you have a function called "matrix_vector()"
    that conforms to the interface described above for the Jacobi routine.  It
    is assumed that the results of matrix_vector are only accurate for non-halo
    rows (similar to compute_fd)./
    '''

    nprocs = comm.size
    my_rank = comm.Get_rank()

    o = ones((A.shape[0],))
    oo = matrix_vector(A, o, N, comm)
    if my_rank != 0:
        oo = oo[N:]
        X = X[N:]
        Y = Y[N:]
    if my_rank != (nprocs-1):
        oo = oo[:-N]
        X = X[:-N]
        Y = Y[:-N]
    import sys
    for i in range(oo.shape[0]):
        sys.stderr.write("X,Y: (%1.2e, %1.2e),  Ouput: %1.2e\n"%(X[i], Y[i], oo[i]))


###########
# This code block chooses the final time and problem sizes (nt, n) to loop over.
# - You can automate the selection of problem sizes using if/else statements or
#   command line parameters.  Or, you can simply comment in (and comment out)
#   lines of code to select your problem sizes.
#
# - Note that N_values corresponds to the number of _interior_ (non-boundary) grid
#   points in one coordinate direction.  Your total number of grid points in one
#   coordinate direction would be (N_values[k] + 2).
#
# - The total number of spatial points is (N_values[k] + 2)^2

# Use these problem sizes for your error convergence studies
Nt_values = array([8, 8*4, 8*4*4, 8*4*4*4])
N_values = array([8, 16, 32, 64 ])
T = 0.5

# One very small problem for debugging
#Nt_values = array([8])
#N_values = array([8])
#T = 0.5

# Parallel Task: Change T and the problem sizes for the weak and strong scaling studies
#
# For instance, for the strong scaling, you'll want
# Nt_values = array([1024])
# N_values = array([512])
# T = ...
#
# And for the first weak scaling run, you'll want
# Nt_values = array([16])
# N_values = array([48])
# T = 0.03

###########

# Define list to contain the discretization error from each problem size
error = []

# Begin loop over various numbers of time points (nt) and spatial grid sizes (n)
for (nt, n) in zip(Nt_values, N_values):

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

    # Task: scale A by the grid size
    A *= 1.0/(h**2.0)


    # Declare initial condition
    #   This initial condition obeys the boundary condition.
    u0 = uexact(0, X, Y)


    # Declare storage
    # Task: Declare "u" and "ue".  What sizes should they be?  "u" will store the
    #       numerical solution, and "ue" will store the exact solution.
    # Task in parallel: Adjust the sizes to be only for this processor's
    #                   local portion of the domain.
    u = zeros((nt, sizex*sizey))
    ue = zeros((nt, sizex*sizey))

    # Set initial condition
    u[0,:] = u0
    ue[0,:] = u0

    # Testing harness for parallel part: Only comment-in and run for the smallest
    # problem size of 8 time points and an 8x8 grid
    #   - Assumes specific structure to your mat-vec multiply routine
    #    (described above in comments)
    #matvec_check( (h**2)*A, X, Y, n-2, comm)

    # Run time-stepping over "nt" time points
    for i in range(1,nt):
        # Task: We need to store the exact solution so that we can compute the error
        ue[i,:] = uexact(i*ht, X, Y)

        # Task: Compute boundary contributions for the current time value of i*ht
        #       Different from HW4, need to account for numeric error, hence "1e-12" and not "0"
        g = zeros((A.shape[0],))
        boundary_points = abs(Y - h) < 1e-12        # Do this instead of " boundary_points = (Y == h) "
        #g[boundary_points] += ...
        g[boundary_points] += (1/(h**2))*uexact(i*ht, X[boundary_points], Y[boundary_points]-h)

        #print("X boundary points are : ", X[boundary_points], " and Y boundary are : ", Y[boundary_points]-h)
        boundary_points = abs(1 - (Y + h)) < 1e-12
        g[boundary_points] += (1/(h**2))*uexact(i*ht, X[boundary_points], Y[boundary_points]+h)
        #print("X boundary points are : ", X[boundary_points], " and Y boundary are : ", Y[boundary_points]+h)
        boundary_points = abs(X - h) < 1e-12
        g[boundary_points] += (1/(h**2))*uexact(i*ht, X[boundary_points]-h, Y[boundary_points])
        #print("X boundary points are : ", X[boundary_points]-h, " and Y boundary are : ", Y[boundary_points])
        boundary_points = abs(1 - (X + h)) < 1e-12
        g[boundary_points] += (1/(h**2))*uexact(i*ht, X[boundary_points]+h, Y[boundary_points])

        #print("X boundary points are : ", X[boundary_points]+h, " and Y boundary are : ", Y[boundary_points])

        # Backward Euler
        # Task: fill in the arguments to backward Euler
        forcing = f(i*ht, X, Y)
        #set_trace()
        u[i,:] = (euler_backward(A, (u[i-1,:]), ht, g, forcing, n))

    #set_trace()
    # Compute L2-norm of the error at final time
    #e = (u[nt-1,:] - ue[nt-1,:])
    #enorm = h * ((sum(e ** 2)) ** 0.5)

    # Task: compute the L2 norm over space-time here.  In serial this is just one line.  In parallel...
    # Parallel task: In parallel, write a helper function to compute the norm of "e" in parallel
    e = (u[nt-1,:] - ue[nt-1,:])
    if (rank == 0):
        if (numranks != 1):
            local_e = e[0:-n]
        else:
            local_e = e
    elif (rank == (numranks - 1)):
        if (numranks != 1):
            local_e = e[n:]
    else:
        local_e = e[n:-n]
    
    
    dotlocale = array([dot(local_e, local_e)])
    globaldotlocale = array([0.0])
    comm.Allreduce(dotlocale, globaldotlocale, op=MPI.SUM)
    globallocale = array([0.0])
    globallocale[0] = h * sqrt(globaldotlocale[0])
    
    if (rank == 0):
        print("Nt, N, Error is:  " + str(nt) + ",  " + str(n) + ",  " + str(globallocale[0]))
        
    error.append(globallocale[0])


    # You can turn this on to visualize the solution.  Possibly helpful for debugging.
    # Only works in serial.  Parallel visualizations will require that you stitch together
    # the solution from each processor before one single processor generates the graphic.
    # But, you can use the imshow command for your report graphics, as done below.
    if True:
        if (rank == 0):
            if (numranks != 1):
                local_u0 = u[0,0:-n]
                local_uf = u[nt-1,0:-n]
                local_uef = ue[nt-1,0:-n]
            else:
                local_u0 = u[0]
                local_uf = u[nt-1]
                local_uef = ue[nt-1]
        elif (rank == (numranks - 1)):
            if (numranks != 1):
                local_u0 = u[0,n:]
                local_uf = u[nt-1,n:]
                local_uef = ue[nt-1,n:]
        else:
            local_u0 = u[0,n:-n]
            local_uf = u[nt-1,n:-n]
            local_uef = ue[nt-1,n:-n]
        
        recvu0arr = comm.gather(local_u0, root=0)
        recvufarr = comm.gather(local_uf, root=0)
        recvuefarr = comm.gather(local_uef, root=0)

        if (rank == 0):
            
            globalu0 = concatenate(recvu0arr, axis=0)
            globaluf = concatenate(recvufarr, axis=0)
            globaluef = concatenate(recvuefarr, axis=0)
            
            pyplot.figure(1)
            im = pyplot.imshow(globalu0.reshape(n,n), origin='lower', extent=(0, 1, 0, 1))
            pyplot.colorbar(im, shrink=0.5)
            pyplot.xlabel('X')
            pyplot.ylabel('Y')
            pyplot.title("Initial Condition")
            pyplot.savefig('initialcondplot_parallelranks' + str(numranks) + '.png', dpi=500, format='png',)

            pyplot.figure(3)
            im = pyplot.imshow(globaluf.reshape(n,n), origin='lower', extent=(0, 1, 0, 1))
            pyplot.colorbar(im, shrink=0.5)
            pyplot.xlabel('X')
            pyplot.ylabel('Y')
            pyplot.title("Solution at final time")
            pyplot.savefig('approxfinalsolplot_parallelranks' + str(numranks) + '.png', dpi=500, format='png',)

            pyplot.figure(4)
            im = pyplot.imshow(globaluef.reshape(n,n), origin='lower', extent=(0, 1, 0, 1))
            pyplot.colorbar(im, shrink=0.5)
            pyplot.xlabel('X')
            pyplot.ylabel('Y')
            pyplot.title("Exact Solution at final time")
            pyplot.savefig('exactfinalsolplot_parallelranks' + str(numranks) + '.png', dpi=500, format='png',)

# Plot convergence
if True and (rank == 0):
    # When generating this plot in parallel, you should have only rank=0
    # save the graphic to a .PNG
    pyplot.figure(5)
    pyplot.loglog(1./N_values, 1./N_values**2, '-ok')
    pyplot.loglog(1./N_values, array(error), '-sr')
    pyplot.tick_params(labelsize='large')
    pyplot.xlabel(r'Spatial $h$', fontsize='large')
    pyplot.ylabel(r'$||e||_{L_2}$', fontsize='large')
    pyplot.legend(['Ref Quadratic', 'Computed Error'], fontsize='large')
    #pyplot.show()
    pyplot.savefig('errconvplot_parallelranks' + str(numranks) + '.png', dpi=500, format='png', bbox_inches='tight', pad_inches=0.0,)
