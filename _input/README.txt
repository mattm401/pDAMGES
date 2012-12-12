

Call the program <input file for a> <rank> <input file for b> <method> <iterations>
Ex.
./serial _Input/sparse_rank900_fdlaplacian 900 _Input/rhs_fdlaplacian_900x1 1 1000
./parallel _Input/sparse_rank900_fdlaplacian 900 _Input/rhs_fdlaplacian_900x1 1 1000


NOTES:
<method> is 1 - Gauss Siedel; 2 - Jacobi
When using the parallel version, don't forget to set OMP_NUM_THREADS to the desired number of threads.
Thread counts tested: 1, 2, 4, 8, 16, 24


MAKEFILE:

"make" will compile serial.c into serial and parallel.c into parallel
"make serial" will just compile serial
"make cblas" will just compile parallel
"make clean" will clean directory 
