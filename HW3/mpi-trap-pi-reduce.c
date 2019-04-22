/* trap-pi-reduce.c -- Parallel Trapezoidal Rule, modified version for pi, double  precision
 *                     Uses the reduction operation to compute the integral
 *
 * Input: n, the number of intervals.
 * Output:  Estimate of the integral from a to b of f(x)
 *    using the trapezoidal rule and n trapezoids.
 *
 * Algorithm:
 *    1.  Each process calculates "its" interval of
 *        integration.
 *    2.  Each process estimates the integral of f(x)
 *        over its interval using the trapezoidal rule.
 *    3.  The processes together compute the sum of the integrals 
 *        by the reduction operation in MPI. 
 *        Process 0 has the value of the integral. 
 *
 * Notes:  
 *    1.  f(x), a, b, are all hardwired.
 *    2.  The number of processes (p) should evenly divide
 *        the number of trapezoids (n, value is input)
 *
 * From  Chap. 4, pp. 56 & ff. in PPMPI.
 * Modified by Alex Pothen, Feb 2014. 
 */
#include <stdio.h>

/* We'll be using MPI routines, definitions, etc. */
#include "mpi.h"


main(int argc, char** argv) {
    int         my_rank;   /* My process rank           */
    int         p;         /* The number of processes   */
    double       a = 0.0;   /* Left endpoint             */
    double       b = 1.0;   /* Right endpoint            */
    int         n ;         /* Number of trapezoids      */
    double       h;         /* Trapezoid base length     */
    double       local_a;   /* Left endpoint my process  */
    double       local_b;   /* Right endpoint my process */
    int         local_n;   /* Number of trapezoids for  */
                           /* my calculation            */
    double       integral;  /* Integral over my interval */
    double       total;     /* Total integral            */
    double       PI16D = 3.141592653589793;  /* value of pi to 16 digits */
    int         source;    /* Process sending integral  */
    int         dest = 0;  /* All messages go to 0      */
    int         tag = 0;
    double      start_time; 
    double      elapsed_time; 

    MPI_Status  status;

    double Trap(double local_a, double local_b, int local_n,
              double h);    /* Calculate local integral  */

    /* Let the system do what it needs to start up MPI */
    MPI_Init(&argc, &argv);

    /* Get my process rank */
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    /* Find out how many processes are being used */
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    /* Process 0 reads the value of n, then sends to all other processes */ 
    if (my_rank == 0){
      printf("Enter the no. of intervals n.\n");
      printf("Must be divisible by no. of processes. \n");
      scanf("%d",  &n);
    }        

     start_time = MPI_Wtime(); 

    /* Process 0 broadcasts the value of n to all processes */ 
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD); 
    
    h = (b-a)/n;    /* h is the same for all processes */
    local_n = n/p;  /* p must divide n, so  number of trapezoids is same for all processes */

    /* Length of each process' interval of
     * integration = local_n*h.  So my interval
     * starts at: */
    local_a = a + my_rank*local_n*h;
    local_b = a + (my_rank+1)* local_n*h;        
    /* the right endpoint of a process's interval and the left endpoint of the next process's 
            interval are the same*/
    integral = Trap(local_a, local_b, local_n, h);

    /* Add up the integrals calculated by each process. We use the MPI_Reduce operation here */
    MPI_Reduce(&integral, &total, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD); 

    /* total is an approximation to the integral = pi/4 */
    /* Print the result */
    if (my_rank == 0) {
        printf("With n = %d trapezoids, the error in PI is \n",
            n);
        printf("%25.16e\n",
             PI16D - 4 * total);
        elapsed_time = MPI_Wtime() - start_time; 
        printf("Elapsed time with reduction is %13.3e\n", elapsed_time); 
    }

    /* Shut down MPI */
    MPI_Finalize();
} /*  main  */


double Trap(
          double  local_a   /* in */,
          double  local_b   /* in */,
          int    local_n   /* in */,
          double  h         /* in */) {

    double integral;   /* Store result in integral  */
    double x;
    int i;

    double f(double x); /* function we're integrating */

    integral = (f(local_a) + f(local_b))/2.0;
    x = local_a;
    for (i = 1; i <= local_n-1; i++) {
        x = x + h;
        integral = integral + f(x);
    }
    integral = integral*h;
    return integral;
} /*  Trap  */


double f(double x) {
    double return_val;
    /* Calculate f(x) = 1/(1+x^2). */
    /* Store calculation in return_val. */
    return_val = 1/(1 + x*x);
    return return_val;
} /* f */


