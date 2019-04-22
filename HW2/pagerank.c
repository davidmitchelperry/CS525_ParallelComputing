#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#define CONST_ALPHA	0.85
#define CONST_TOL	0.00001

void readfile(char *fileName, int *n, double **value, int **colind, int **rbegin){
	FILE *fp=fopen(fileName,"r");
	char buf[200];
	int i,j,k,l;
	int p,q;
	double w;
	int m;
	if ((fp=fopen(fileName,"r"))==NULL){
	  fprintf(stderr,"Open file errer, check filename please.\n");
	}
	fgets(buf,200,fp);
	*n=atoi(buf);
	l=0;
	while(buf[l++]!=' ');
	m=atoi(&buf[l]);
	printf("Matrix size: %d, #Non-zeros: %d\n",*n,m);
	(*value)=(double*)malloc(sizeof(double)*m);
	(*colind)=(int*)malloc(sizeof(int)*m);
	(*rbegin)=(int*)malloc(sizeof(int)*((*n)+1));
	k=-1;
	for(i=0;i<m;i++){
	  fgets(buf,200,fp);
	  l=0;p=atoi(&buf[l]);
	  while(buf[l++]!=' '); q=atoi(&buf[l]);
	  while(buf[l++]!=' '); w=atof(&buf[l]);
	  (*value)[i]=w;
	  (*colind)[i]=q;
	  if(p!=k){
	    k=p;
	    (*rbegin)[p]=i;
	  }
	}
	(*rbegin)[*n]=m;
	close(fp);
}


double* pageRank(int n, double *value, int* colind, int *rbegin){

	int i, j, ct, iter;
	int k1, k2, k;
	double *y, *x, *prev_x;
	double gamma = (((double)1 - CONST_ALPHA)/(double)n);
	int first = 1;
	int converged = 0;
	int iterCT = 0;

	y=(double*)malloc(sizeof(double)*n);
	x=(double*)malloc(sizeof(double)*n);
	prev_x=(double*)malloc(sizeof(double)*n);

	//Make all initial values in x = 1/n
	for (ct =0; ct < n; ct++) {
		x[ct] = (double)1/n;
	}

	do {	
		//If this is not the first iteration
		if (!first) {
			//set the value of x to y, therefore making x hold the previous y value
			//for next calculation
			for (i = 0; i < n; i++) {
				x[i] = y[i];
			}
		}
		
		//Once here it will never be the first iteration again
		first = 0;

		//Do the calculations in parallel, changing the chunk size effects runtime
		//I found a chunksize of 1000 to be efficient in many cases
		#pragma omp parallel for shared(y, rbegin, x) private(i, k1, k2, k, j) schedule(dynamic, 1000)
		for (i =0; i < n; i++) {
			//initialize current value of y
			y[i] = 0;
			k1 = rbegin[i];
			k2 = rbegin[i+1] - 1;
			//get the current y value
			for (k = k1; k <= k2; k++) {
				j = colind[k];
				y[i] += value[k] * x[j];
			}
			//take account for muliplying by alpha and adding gamma
			y[i] = (y[i] * CONST_ALPHA) + gamma;
		}
		//test if any of the values have not converged
		converged = 1;
		for (i =0; i < n && converged; i++) {
			if (fabs(y[i] - x[i]) > CONST_TOL) {
				converged = 0;
			}
		}
	} while (!converged);//rerun the calculations if all values have not converged

	return y;
}


void output(int n, double *answer){
	FILE *fp=fopen("output.dat","w");
	int i;
	for(i=0;i<n;i++){
	  fprintf(fp,"%.16f\n",answer[i]);
	}
	close(fp);
}
int main(int argc, char* argv[]){
	double *value=NULL;
	int *colind=NULL;
	int *rbegin=NULL;
	double *answer=NULL;
	int n;
	struct timeval tv1,tv2;
	readfile(argv[1],&n,&value,&colind,&rbegin);
	gettimeofday(&tv1,NULL);
	answer=pageRank(n,value,colind,rbegin);
	gettimeofday(&tv2,NULL);
	printf("#Threads=%d, Time=%10.5f\n",omp_get_max_threads(),((double)(tv2.tv_sec-tv1.tv_sec)+(double)(tv2.tv_usec-tv1.tv_usec)/1000000));
	output(n,answer);
	if (value!=NULL) {free(value);value=NULL;}
	if (colind!=NULL) {free(colind);colind=NULL;}
	if (rbegin!=NULL) {free(rbegin);rbegin=NULL;}
	if (answer!=NULL) {free(answer);answer=NULL;}
}
