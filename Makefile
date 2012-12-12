TOPDIR = ..
include $(TOPDIR)/Makefile.system

CFLAGS += -DADD$(BU) -DCBLAS

LIB = $(TOPDIR)/$(LIBNAME)

CEXTRALIB =

main:
	gcc -std=c99 -pthread -fopenmp -o parallel parallel.c -lm -L _lib/ -llapack -lf77blas -lcblas -latlas
	gcc -std=c99 -o serial parallel.c -lm -L _lib/ -llapack -lf77blas -lcblas -latlas

parallel: parallel.c
	gcc -std=c99 -fopenmp -o parallel parallel.c -lm -L _lib/ -llapack -lf77blas -lcblas -latlas

serial: parallel.c
	gcc -std=c99 -o serial parallel.c -pg -lm -L _lib/ -llapack -lf77blas -lcblas -latlas

goto: parallel.c
	gcc $(FLDFLAGS) -std=c99 -fopenmp -o parallel parallel.c -lm $(LIB) $(EXTRALIB) $(CENTRALIB)
	gcc $(FLDFLAGS) -std=c99 -o serial parallel.c -lm $(LIB) $(EXTRALIB) $(CENTRALIB)

clean:
	rm *~ 
	rm serial
	rm parallel
