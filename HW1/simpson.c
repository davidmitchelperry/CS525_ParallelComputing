/* omp-trap-pi1.c -- shared memory code, trapezoidal rule for computing pi in double precision
 *Version 1: uses reduction operator 
 *
 * Calculate definite integral using trapezoidal rule.
 * The function f(x)  = 1/ (1 + x*x) is hardwired.
 * Input: a, b, n. For computing pi, a = 0, b= 1. 
 * Output: estimate of integral from a to b of f(x)
 *    using n trapezoids. 
 *
 * * NOTE: n is the number of intervals. 
 * * The number of interior points, (n-1), must be divisible by the number of threads. 
 *
 * PI16digits = 3.1415 9265 3589 793  
 */

#include <omp.h>
#include <stdio.h>

main() {
    double  integral;   /* Store result in integral   */
    double  a, b;       /* Left and right endpoints   */
    int    n;      /* Number of intervals       */
    double  h;          /* interval  width       */
    double  x;
    int    i;
    int nthreads; 
    double start, end, compute_time; 
    double PI16D = 3.141592653589793;  
    double x_2;
    double x_2_1;

    double f(double x);  /* Function we're integrating */
 
    a= 0.0; 
    b= 1.0; 
    printf("Enter the number of intervals,  n. (n-1) must be divisible by no. of threads. \n");
    scanf("%d",  &n);

    h = (b-a)/n;
    integral = 0;//(f(a) - f(b));
    x = a;
    nthreads = omp_get_max_threads(); 
    printf("No. of threads = %d \n", nthreads); 
    start = omp_get_wtime(); 
    #pragma omp parallel for shared(h,a,n) private(x, x_2, x_2_1) reduction(+:integral)
    for (i = 1; i <= n/2; i += 1) { /* (n/2) iterations */
	x_2 = a + (2.0 * i) * h;
 	x_2_1 = a + (2.0 * i - 1) * h;
 	integral = integral + (4.0 * f(x_2_1)) + (2.0 * f(x_2));
    }
    integral = (f(a) - f(b) + integral) * (h/3.0) * 4.0;  /* we calculate pi/4 */
    end = omp_get_wtime();   
    compute_time = end - start; 

    printf("With nthreads = %d threads, and n = %d intervals, the error in PI \n",
	   nthreads, n);
    printf(" = %25.16e\n",  PI16D - integral);
    printf("Time taken with reduction operator is %15.2e\n",  compute_time);  

} /* main */


double f(double x) {
    double return_val;
    /* Calculate f(x).  Store calculation in return_val. */

    return_val = 1.0/(1.0 + x*x);
    return return_val;
} /* f */
