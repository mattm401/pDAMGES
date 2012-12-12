/* 
 * This file attempts to create a general Gauss-Seidel Solver * CMSC 714 - Fall 2012 - Dr. Alan Sussman
 * Benjamin Jimenz, Viriginia Forstall, and Matthew Mauriello
 * 
 */

#ifdef _OPENMP
  #include <omp.h>
#else
  #define omp_get_thread_num() 0
#endif

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include "_include/cblas.h"
//#include "../cblas.h"
//#include "_include/clapack.h"

int  Procces_Command_Line( int *col, int *method, int *max_iter, int num, char *list[]); 
void Print_Vars(double **A, double *x, double *b, int col);
void gauss_siedel_omp(double **A, double *x, double *xPrime, double *b, int col, int p);
void jacobi_omp(double **A, double *x, double *xPrime, double *b, int col, int p);
int  coarse_op_omp(int n, double **A, double **P, int p);


int main(int argc, char *argv[]) 
{  
  int col, method, max_iter = 0;
  char *junk = NULL;
  FILE *fp = NULL;
  int n = 0;                          /* Coarsening Parameter (10)*/
  int i,j,k = 0;                      /* Generic Loop Counters    */
  int ret = 0;                        /* Generic return variable  */
  double **A, *Agrid, **P, *Pgrid, *x, *xPrime, *b, *swap = NULL;
  double *resVec, *coarse_res, **AP, **Acoarse, *AP_grid, *Acoarse_grid; 
  double res = 1000.0;
  int coarse_pts   = 0.0;
  double threshold = .0000001;
  int info = 0;
  int *permute;
  int numthreads = 0;

  double time,  wall_start, wall_end;
  clock_t clock_start, clock_end;


  #ifdef _OPENMP
  wall_start = omp_get_wtime();
  numthreads=omp_get_max_threads();
  #else
  numthreads=1;
  #endif


  /* ------------------------ */
  /* Process Command Line     */ 
  /* ------------------------ */
  fprintf(stderr, "Solver Starting...\n");
  ret = Procces_Command_Line(&col, &method, &max_iter, argc, argv); 
  if (ret == -1) {
    fprintf(stderr, "Process_Command_Line() Failure!\n");
    return 0;
  } else {
    /* Proccessed command line; update variables as necessary */
   n = col;
   fprintf(stderr, "Command Line Parsed Successfully\n");
  }
  

  /* ------------------------ */
  /* Dynamic Memory Allocation*/ 
  /* ------------------------ */
  A = calloc(col, sizeof(double *));
  P = calloc(col, sizeof(double *));
  x = calloc(col, sizeof(double *));
  xPrime = calloc(col, sizeof(double *));
  b = calloc(col, sizeof(double *));
  Agrid = calloc((col * col), sizeof(double *)); 
  Pgrid = calloc((col * col), sizeof(double *));
  resVec = calloc(col, sizeof(double *));
  AP = calloc(col, sizeof(double *));  

  for(i = 0; i < col; i++) {
    A[i] = (double *) Agrid + (col * i);
    P[i] = (double *) Pgrid + (col * i);
  }

  if (A == NULL || x == NULL || xPrime == NULL || b == NULL || Agrid == NULL) {
     fprintf(stderr, "Dynamic Memory Allocation Failure!\n");
     return 0;
  } else {
    fprintf(stderr, "Dynamic Memory Allocation Completed\n");
  }


  /* ------------------------ */
  /* Load Data From Files     */ 
  /* ------------------------ */
  fp = fopen(argv[1], "r");
  if (fp == NULL) {
    fprintf(stderr, "A File Read Error!\n");
    return 0;
  } else {
    for(i = 0; i < col; i++) {
      for(j = 0; j < col; j++){
	fscanf(fp, "%lf", &A[i][j]);
        fscanf(fp, "%c", &junk);
        fscanf(fp, "%c", &junk);
      }
    }
    /* -----------------------------------  */
    /*   Uncomment below for sparse         */
    /*   And above for exact representation */
    /* ------------------------------------ */
    /* int nnz; 
    fscanf(fp, "%d", &nnz); 
    printf("About to read: %d\n", nnz);
    for(k = 0; k < nnz; k++) {
      fscanf(fp, "%d", &i);
      fscanf(fp, "%d", &j);
      fscanf(fp, "%lf", &A[i-1][j-1]);
      fprintf(stderr, "[%d][%d]: %0.15f\n", i, j, A[i-1][j-1]);
      }*/ 
  }
  fprintf(stderr, "A Read Successful\n");

  fp = fopen(argv[3], "r");
  if (fp == NULL) {
    fprintf(stderr, "b File Read Error!\n");
    return 0;
   } else {
    for(i = 0; i < col; i++) {
      fscanf(fp, "%lf", &b[i]);
    }
  }
  fprintf(stderr, "b Read Successful\n");


  /* ------------------------ */
  /* Start Timing Algorithm   */ 
  /* ------------------------ */
  clock_start = clock();
  
  
  /* ------------------------ */
  /* Coarsening Routine       */ 
  /* ------------------------ */
  coarse_pts = coarse_op_omp(n, A, P, numthreads);
  coarse_res = calloc(coarse_pts, sizeof(double *));
  permute = calloc(coarse_pts, sizeof(int *));
  Acoarse = calloc(coarse_pts, sizeof(double *));

  Acoarse_grid = calloc((coarse_pts * coarse_pts), sizeof(double *)); 
  AP_grid = calloc((col * coarse_pts), sizeof(double *));

  for(i = 0; i < col; i++) {
    AP[i] = (double *) AP_grid + (coarse_pts * i);
    if ( i < coarse_pts) {
      Acoarse[i] = (double *) Acoarse_grid + (coarse_pts * i);
    }
  }

    fprintf(stderr, "Coarsening Completed\n");
  /* ------------------------ */
  /* Lapack Init Routines     */ 
  /* ------------------------ */
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, coarse_pts, n, 1, Agrid, n, Pgrid, n, 0, AP_grid, coarse_pts);
  cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, coarse_pts, coarse_pts, n, 1, Pgrid, n, AP_grid, coarse_pts, 0, Acoarse_grid, coarse_pts);   //Acoarse = P^T(tempAP)
//  clapack_dgetrf(CblasRowMajor, coarse_pts, coarse_pts, Acoarse_grid, coarse_pts, permute);
  fprintf(stderr, "Lapack Init Completed\n");
  
  /* ------------------------ */
  /* Beginning Iterations     */ 
  /* ------------------------ */
  fprintf(stderr, "Iterating...\n");
  for(k = 0; k < max_iter; k++) {
    if (res <= threshold)
      break;

    for (i = 0; i < col; i++){
      resVec[i] = b[i];
    }

    if (method == 1){
	gauss_siedel_omp(A, x, xPrime, b, col, numthreads);
    } else{
	jacobi_omp(A, x, xPrime, b, col, numthreads);
    }
 
    /* Swap Pointers */    
    swap = x;
    x = xPrime;
    xPrime = swap;


    if (method == 1){
	gauss_siedel_omp(A, x, xPrime, b, col, numthreads);
    } else {
	jacobi_omp(A, x, xPrime, b, col, numthreads);
    }
    
    /* Swap Pointers */    
    swap = x;
    x = xPrime;
    xPrime = swap;


    //Solve Coarse System
    cblas_dgemv(CblasRowMajor, CblasNoTrans, n, n, -1, Agrid, n, x, 1, 1,resVec, 1);   //temp = b - Ax
    cblas_dgemv(CblasRowMajor, CblasTrans, n, coarse_pts, 1, Pgrid, n, resVec, 1, 0, coarse_res, 1); //P^T(temp)

    //maybe need to cast P as a 2D array (not a 1d array)
// clapack_dgesv(CblasRowMajor, coarse_pts, 1, Acoarse_grid, coarse_pts, permute, coarse_res, n); //coarse_res = A\coarse_res; 
//  clapack_dgetrs(CblasRowMajor, CblasNoTrans, coarse_pts, 1, Acoarse_grid, coarse_pts, permute, coarse_res, coarse_pts);
    cblas_dgemv(CblasRowMajor, CblasNoTrans, n, coarse_pts, 1, Pgrid, n, coarse_res, 1, 1, x, 1); //x = x + Pcoar

    
    if (method == 1) {
	gauss_siedel_omp(A, x, xPrime, b, col, numthreads);
    } else {
	jacobi_omp(A, x, xPrime, b, col, numthreads);
    }

    /* Swap Pointers */    
    swap = x;
    x = xPrime;
    xPrime = swap;
    
    
    if (method == 1) {
	gauss_siedel_omp(A, x, xPrime, b, col, numthreads);
    } else {
	jacobi_omp(A, x, xPrime, b, col, numthreads);
    }
    
    /* Swap Pointers */    
    swap = x;
    x = xPrime;
    xPrime = swap;


    for (i = 0; i < col; i++){
      resVec[i] = b[i];
    }
    cblas_dgemv(CblasRowMajor, CblasNoTrans, n, n, -1, Agrid, n, x, 1, 1, resVec, 1);   //temp = b - Ax
    res = cblas_dnrm2(n, resVec, 1);
    fprintf(stderr, "Iteration [%d] Completed: x[0] = %0.15f\n", k, x[0]);
} 

/* End of Iterations */


  /* ------------------------ */
  /* Print Results            */ 
  /* ------------------------ */
  #ifdef _OPENMP
    wall_end = omp_get_wtime() - wall_start;
    fprintf(stderr, "%d threads: iterations = %d, walltime = %.3f\n", numthreads, k, wall_end);
    for(i = 0; i < col; i++)
      printf("%0.15f\n", x[i]);
  #else
    clock_end = clock() - clock_start;
    fprintf(stderr, "%d threads: iterations = %d,  time on clock() = %0.3f\n",  numthreads, k, (double) (clock_end) / CLOCKS_PER_SEC);
    for(i = 0; i < col; i++)
      printf("%0.15f\n", x[i]);
  #endif

  /* ------------------------ */
  /* Dyn Memory De-Allocation */ 
  /* ------------------------ */
  free(A);
  free(P);
  free(x);
  free(xPrime);
  free(b);
  free(Agrid);
  free(Pgrid);
  free(resVec);
  free(coarse_res);
  free(permute);
  free(AP);
  free(Acoarse); 
  free(AP_grid); 
  free(Acoarse_grid);

  clock_end = clock() - clock_start;
  fprintf(stderr, "%d threads: iterations = %d,  time on clock() = %0.3f\n",  numthreads, k, (double) (clock_end) / CLOCKS_PER_SEC);

  return 0;
}


void gauss_siedel_omp(double **A, double *x, double *xPrime, double *b, int col, int p)
{
  int i,j,k = 0; /* General Purpose Counters */
  double sum;
  int dnp = col/p;
  int id, jstart, jstop = 0;
  int numthreads = p;

  #pragma omp parallel private(id, i, j, sum, jstart, jstop)
  {  
    id = omp_get_thread_num();
    jstart = id * dnp;
    if (id == numthreads - 1) 
      jstop = col;
    else
      jstop = jstart + dnp;
    if (jstart % 2 != 0) 
      jstart++;
    
    for (i = jstart; i < jstop; i += 2){
      sum = 0.0;
      for(j = 0; j < col; j++){
	if (j != i)
	  sum = sum + A[i][j]*x[j];
      }
      xPrime[i] = (1/A[i][i]) * (b[i] - sum);
    }
  }
  
  #pragma omp parallel private(id, i, j, sum, jstart, jstop)
  {
    id = omp_get_thread_num();
    jstart = id * dnp;
    if (id == numthreads - 1) 
      jstop = col;
    else
      jstop = jstart + dnp;
    if (jstart % 2 == 0) 
      jstart++;    

    for (i = jstart; i < jstop; i += 2) {
      sum = 0.0;
      for(j = 0; j < col; j++){
	if (j != i) {
	  if (j % 2 == 0)
	    sum = sum + A[i][j]*xPrime[j];
	  else
	    sum = sum + A[i][j]*x[j];
	}
      }
      xPrime[i] = (1/A[i][i]) * (b[i] - sum);
    }
  }
} /* function */


void jacobi_omp(double **A, double *x, double *xPrime, double *b, int col, int p)
{
  int i,j,k = 0; /* General Purpose Counters */
  double sum;
  int dnp = col/p;
  int id, jstart, jstop = 0;
  int numthreads = p;
  
  #pragma omp parallel private(id, i, j, sum, jstart, jstop)
  {  
    id = omp_get_thread_num();
    jstart = id * dnp;
    if (id == numthreads - 1) 
      jstop = col;
    else
      jstop = jstart + dnp;
    
    for (i = jstart; i < jstop; i++){
      sum = 0.0;
      for(j = 0; j < col; j++){
	if (j != i)
	  sum = sum + A[i][j]*x[j];
      }
      xPrime[i] = (1/A[i][i]) * (b[i] - sum);
    }
  }
} /* function */


int Procces_Command_Line(int *col, int *method, int *max_iter,  int num, char *list[]) 
{
  if (num == 6) {

    *col = atoi(list[2]);
    if (*col <= 0) {
      return -1;
    }

    *method = atoi(list[4]);
    if(*method != 1 && *method != 2) {
      return -1;
    }

    *max_iter = atoi(list[5]);
    if(max_iter <= 0) {
      return -1;
    }

  } else {
    return -1;
  }
  return 0;
}

void Print_Vars(double **A, double *x, double *b, int col)
{
  int i,j,k = 0; /* General Purpose Counters */

 /* Lets print A, x, b */
  printf("A:\n");
  for(i = 0; i < col; i++) {
    for(j = 0; j < col; j++) {
      printf("%0.15f  ", A[i][j]);
    }
    printf("\n");
  }
  printf("-----------\n");
  printf("x:\n");
  for(i = 0; i < col; i++) {
    printf("%0.15f\n", x[i]);
  }
  printf("-----------\n");
  printf("b:\n");
    for(i = 0; i < col; i++) {
    printf("%0.15f\n", b[i]);
  }
  printf("-----------\n");
}


int coarse_op_omp(int n, double **A, double **P, int p)
{
  int i,j,k,ii,jj,kk,l,m,mm,cii,my_cpu_id;
  int lC,lF,lU,lCi,lCj,lCk,lDi,exit,fail; 
  double theta,aik,y,topsum,bottomsum,temp,temp2; 
  int **S,**Ci,**Di; 
  int *lambda,*U,*C,*F; 
  int numthreads = p;

  S = (int**)malloc(n * sizeof(int*)); 
  lambda = (int*)malloc(n*sizeof(int));
  C =  (int*)malloc(n*sizeof(int));
  F = (int*)malloc(n*sizeof(int));
  U = (int*)malloc(n*sizeof(int)); 
  Ci =  (int**)malloc(n*sizeof(int*));
  Di =  (int**)malloc(n*sizeof(int*));

  //  printf("number of threads %d \n", numthreads); 
  for (i = 0; i<n;i++){
    S[i] = (int*)malloc(n * sizeof(int)); 
    Ci[i] = (int*)malloc(numthreads * sizeof(int)); 
    Di[i] = (int*)malloc(numthreads * sizeof(int));
  }
  //  printf("Done with Dynamic Alloc\n");
  //#pragma omp parallel for private(j)
  for (i = 0; i < n; i++){
    for (j = 0; j < n; j++){
      S[i][j] = 0; 
    }
    lambda[i] = 0; 
  }
  //  printf("S and lambda Loop\n");
  //parameter for the Runge-Stuben algorithm (strength tolerance)  
  theta = 0.25; 
 
  //preprocessing: compute the strength matrix, S
  //#pragma omp parallel for private(j, aik)
  for (i =0; i < n; i++){
    
    //find the maximum off diagonal element in the ith row
    aik = -1000000.0; 
    for (j = 0; j<n; j++) {
      if (i!=j) {
	if (-A[i][j] > aik ){
	  aik = -A[i][j]; 
	}
      }
    }
 
    aik = theta * aik; 
    for (j = 0; j < n; j++) {
      if (-A[i][j] >= aik){
	if (j != i) {
	  S[i][j] = 1; 
	}
      }
    }
 
  }
  //  printf("Done with Preprocessing\n");
  lF = 0; 
  lC = 0; 
  lU = n; 
#pragma omp parallel for private(i,j,temp)
  for(i=0;i<n;i++){
    temp = 0; 
    for (j=0;j<n;j++){
      temp = temp + S[i][j]; 
    }
    lambda[i] = temp;
    if (temp == 0) {
      
      #pragma omp critical 
/*      #pragma omp atomic */
      {
	F[lF] = i+1;
	lF = lF + 1;
	lU = lU - 1;  
      }
      U[i] = 0; 
    } else {
      U[i] = i + 1; 
    }
    
  }
  //  printf("Done with lambda\n");
  //  printf("Done with U\n");
  while (lU > 0) {
    //finding maximum value of lambda, not parallelized yet. (is there a reduction ??) 
    i = 0; 
    y = lambda[i]; 
    for (j = 1; j<n; j++){ 
      if (lambda[j] > y){
	y = lambda[j]; 
	i = j; 
      }//if
    }//forj
    //add this value to the coarse grid, can't be parallelized
    lambda[i] = 0; 
    C[lC] = U[i]; 
    lC = lC + 1;
    U[i] = 0;
    lU = lU - 1; 
    //add the fine points to the coarse grid
    //#pragma omp parallel for private(k)
    for (j = 0; j<n; j++) {
      if ((S[j][i] == 1) && (U[j] != 0)) {
	//#pragma omp critical
	//{
	F[lF] = j+1;
	lF = lF + 1;
	lU = lU - 1;
	//}//critical
	lambda[j] = 0; 
	U[j] = 0; 
	for (k = 0; k< n; k++){
	  if (U[k] > 0) {
	    if (S[j][k] == 1) {
	      //#pragma omp critical
	      //{
	      lambda[k] = lambda[k] + 1; 
	      //}//critical
	    } //if
	  } //if
	}//fork
      }//if
    }//forj (parallel)
  }//end while
  //printf("Done with 1st pass\n");
  //second pass  
  ii = 0; 
  while (ii < lF) {
    i = F[ii]-1; 
    exit = 0; 
    jj = 0;
    while (exit == 0 && jj < lF) {//   and i not equal j, repeat of condition of S(i,j) == 1
      j = F[jj]-1;
      if (S[i][j] ==  1) { //  %j in S_i
	//%C intersect S_i intersect S_j empty ?
	lCi = 0; 
	for (k=0; k< lC; k++) {
	  if (S[i][C[k]-1] == 1) {
	    Ci[lCi][0] = C[k];
	    lCi = lCi + 1;              
	  }
	}//fork 
	lCj = 0; 
	for (k = 0; k<lC;k++){
	  if (S[j][Ci[k][0]-1] == 1){
	    lCj = lCj + 1; 
	  }
	}//fork
	
	//add j to Ctemp
	Ci[lCi][0] = j+1;
	lCi =  lCi + 1; 
	if (lCj == 0){
	  //loop through k in D_i S
	  fail = 0;
	  for (kk =0;kk<lF;kk++) {
	    k = F[kk]-1;
	    if ( k != j && S[i][k] == 1 ){// %belongs to D_iS 
                           
	      //C_i union C_temp, intersect (F intersection S_k) = empty ?
	      lCk = 0; 
	      for (l = 0;l <lCi; l++){
		if (S[k][Ci[l][0]-1] == 1){
		  lCk = lCk + 1; 
		}
	      }//forl
	      if (lCk == 0) {
		fail = 1;
	      }
                           
	    }
	  }//end forkk
                    
	  if (fail == 1) {
	    C[lC] =  i+1; 
	    lC = lC + 1; 
	    F[ii] = F[lF-1]; 
	  } else {
	    C[lC] = j+1;
	    lC = lC + 1; 
	    F[jj] = F[lF-1]; 
                       
	  }
	  lF = lF - 1; 
	  exit = 1; 
	}//if lCj
      }//if sij
      jj = jj + 1; 
    }//while jj
    if (exit == 0) {
      ii = ii + 1; 
    }
  }//while ii
  /*printf("C = ["); 
    for (i=0;i<300;i++){
       printf("%d ", C[i]); 
       }
       printf("] \n");
       printf("%d \n",lC); 
  */
  //Build interpolation matrix
#pragma omp parallel for
  for (i = 0; i< lC;i++) {
    P[C[i]-1][i] = 1; 
  }
  int s = lF/numthreads;   //number of iterations to run per thread
  int beg, end; 
#pragma omp parallel private(i,j,k,ii,lCi, my_cpu_id,beg,end,lDi,mm,m,temp,temp2,cii,topsum,bottomsum)
  {
#ifdef _OPENMP
    my_cpu_id = omp_get_thread_num();
#else
    my_cpu_id = 0;
#endif 
    //Assign block of indices to this thread
    beg = s*my_cpu_id; 
    if (my_cpu_id < (numthreads - 1))
      end = (my_cpu_id + 1)*s; 
    else
      end = lF;
    //printf("process %d starts with beginning %d and ends with %d \n", my_cpu_id,beg,end);  
    for (ii = beg; ii < end; ii++){
      i = F[ii]-1; 
      lCi = 0; 
      for (k = 0;k<lC;k++){
	if (S[i][C[k]-1] == 1) {
	  Ci[lCi][my_cpu_id] = C[k]; 
	  lCi = lCi + 1; 
	}
      }//fork
      lDi = 0; 
      for (k =0;k<lF;k++){
	if (S[i][F[k]-1] == 1){
	  Di[lDi][my_cpu_id] = F[k]; 
	  lDi = lDi + 1;
	}
      }//fork 
      for (j =0;j<lC;j++){
	if (S[i][C[j]-1] == 1){
	  //%top row sum over Di
	  topsum = 0; 
	  for (mm = 0; mm<lDi;mm++){
	    m = Di[mm][my_cpu_id]-1; 
	    temp = A[i][m] * A[m][C[j]-1]; //aim*amj
	    temp2 = 0; 
	    for (cii =0;cii<lCi;cii++){
	      temp2 = temp2 + A[m][Ci[cii][my_cpu_id]-1]; 
	    }
	    
	    temp = temp/temp2; 
	    topsum = topsum + temp; 
	  }//mm
	  topsum = topsum + A[i][C[j]-1];    
	  bottomsum = 0; 
	  for( m = 0;m<n;m++){
	    if (S[i][m] == 0){
	      bottomsum = bottomsum + A[i][m]; 
	    }
	  }//form
	  bottomsum = bottomsum + A[i][i]; 
	  P[i][j] = -topsum/bottomsum; 
	}//if sij
      }//forj
    }//for ii
  } //end parallel pragma


  return lC;
} //coarse_op
