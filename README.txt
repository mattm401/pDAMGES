Call the program <input file for a> <rank> <input file for b> <method> <iterations>
Ex.
./serial _Input/sparse_rank900_fdlaplacian 900 _Input/rhs_fdlaplacian_900x1 1 1000
./parallel _Input/sparse_rank900_fdlaplacian 900 _Input/rhs_fdlaplacian_900x1 1 1000

You will probably want to pipe the stdout to a file using the '>' operator.


NOTES:
<method> is 1 - Gauss Siedel; 2 - Jacobi

When using the parallel version, don't forget to set OMP_NUM_THREADS to the desired number of threads.
Thread counts tested: 1, 2, 4, 8, 16, 24

When using serial cblas compilation, make sure OMP_NUM_THREADS=1.

GOTO_NUM_THREADS has not be tested thoroughly, do not use.


MAKEFILE:

"make" will compile serial.c into serial and parallel.c into parallel FOR ATLAS
"make parallel" will just compile parallel FOR ATLAS
"make serial" will just compile serial FOR ATLAS
"make goto" will compile serial.c into serial and parallel.c into parallel FOR GotoBLAS2
"make clean" will clean directory 
