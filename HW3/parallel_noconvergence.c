#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define CONST_TOL	0.00001
#define MPI_RANK_ROOT	0
#define MAX_ITER	1000

void readFile(char *fileName, int *n, double **value, int **colind, int **rbegin, double **b){
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
	(*b)=(double*)malloc(sizeof(double)*(*n));
	for(i=0;i<(*n);i++){(*b)[i]=1.0;}
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
	fclose(fp);
}
void scatterData(int *n, int *m, double **value, int **colind, int **rbegin, double **b, int rank, int nproc){
	int np;
	int *sendcnts;
	int *displs;
	double *gvalue;
	int *gcolind;
	int *grbegin;
	double *gb;
	int i,j;
	if (rank==MPI_RANK_ROOT){
	  sendcnts=(int*)malloc(sizeof(int)*(nproc));
	  displs=(int*)malloc(sizeof(int)*(nproc));
	  np=(*n)/nproc;
	  gvalue=(*value);(*value)=NULL;
	  gcolind=(*colind);(*colind)=NULL;
	  grbegin=(*rbegin);(*rbegin)=NULL;
	  gb=(*b);(*b)=NULL;
	  for(i=0;i<nproc;i++){
	    sendcnts[i]=0;
	    for(j=i*np;j<(i+1)*np;j++){
	      sendcnts[i]+=grbegin[j+1]-grbegin[j];
	    }//could be simplified
	  }
	  displs[0]=0;
	  for(i=1;i<nproc;i++){
	    displs[i]=displs[i-1]+sendcnts[i-1];
	  }
	}
	fflush(stdout);
	MPI_Bcast(n,1,MPI_INT,MPI_RANK_ROOT,MPI_COMM_WORLD);
	MPI_Scatter(sendcnts,1,MPI_INT,m,1,MPI_INT,MPI_RANK_ROOT,MPI_COMM_WORLD);
	np=(*n)/nproc;
	printf("Process %d get n=%d, m=%d\n",rank,*n,*m);
	fflush(stdout);
	(*value)=(double*)malloc(sizeof(double)*(*m));
	(*colind)=(int*)malloc(sizeof(int)*(*m));
	(*rbegin)=(int*)malloc(sizeof(int)*(np+1));
	(*b)=(double*)malloc(sizeof(double)*np);
	MPI_Scatterv(gvalue, sendcnts, displs, MPI_DOUBLE, (*value), (*m),MPI_DOUBLE, MPI_RANK_ROOT, MPI_COMM_WORLD);
	MPI_Scatterv(gcolind, sendcnts, displs, MPI_INT, (*colind), (*m),MPI_INT, MPI_RANK_ROOT, MPI_COMM_WORLD);
	
	MPI_Scatter(grbegin, np, MPI_INT, (*rbegin), np, MPI_INT, MPI_RANK_ROOT, MPI_COMM_WORLD);

	MPI_Scatter(gb, np, MPI_DOUBLE, (*b), np, MPI_DOUBLE, MPI_RANK_ROOT, MPI_COMM_WORLD);

	int offset=(*rbegin)[0];
	for(i=0;i<np;i++){
	  (*rbegin)[i]-=offset;
	}
	(*rbegin)[np]=(*m);
	if (rank==MPI_RANK_ROOT){
	  free(gvalue);
	  free(gcolind);
	  free(grbegin);
	  free(gb);
	  free(sendcnts);
	  free(displs);
	}

}
void writeFile(int n,double *answer){
	FILE *fp=fopen("output.dat","w");
	int i;
	for(i=0;i<n;i++){
	  fprintf(fp,"%.10f\n",answer[i]);
	}
	fclose(fp);
}
double* cg(int n, double *value, int* colind, int* rbegin, double *b, int rank, int nproc){

	double *answer = malloc(sizeof(double) * n);
	double r[n/nproc];
	double x[n/nproc];
	double p_local[n/nproc];
	double p_global[n];
	double s[n/nproc];
	int i, j, k, ct, ct1, ct2; 
	int converged = 0;
	double alpha, beta;
	double left_value, right_value, left_value2;
	double left_value_global, right_value_global, left_value2_global;
	double max_local, max_global;

	for(i = 0; i < n/nproc; i++) {
		x[i] = 0;
		s[i] = 0;
		r[i] = 1.0;
		p_local[i] = 1.0;
	}

	i = 0;
	printf("---1---");
	MPI_Allgather(&p_local, n/nproc, MPI_DOUBLE, &p_global, n/nproc, MPI_DOUBLE, MPI_COMM_WORLD);

	while(i <= 100) {
		for (j =0; j < n/nproc; j++) {
			s[j] = 0;
			ct1 = rbegin[j];
			ct2 = rbegin[j+1] - 1;
			for (ct = ct1; ct <= ct2; ct++) {
				//s[j] += value[ct] * p_local[colind[ct]];
				s[j] += value[ct] * p_global[colind[ct]];
			}
		}
		left_value = 0;
		right_value = 0;
	printf("---2---");
		for(k = 0; k < n/nproc; k++) {
			left_value += r[k] * r[k];
			right_value += p_local[k] * s[k];
		}
		MPI_Allreduce(&left_value, &left_value_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); 
		MPI_Allreduce(&right_value, &right_value_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); 

		alpha = left_value_global / right_value_global;
		//alpha = left_value / right_value;

	printf("---3---");
		for(k = 0; k < n/nproc; k++) {
			x[k] = x[k] + alpha * p_local[k];
			r[k] = r[k] - alpha * s[k];
		}

			//test for convergence
			
	printf("---4---");
		left_value2 = 0;
		for(k = 0; k < n/nproc; k++) {
			//can be calculated in loop above
			left_value2 += r[k] * r[k];
		}
		MPI_Allreduce(&left_value2, &left_value2_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		//beta = left_value2 /left_value;
		beta = left_value2_global /left_value_global;

	printf("---5---");
		for(k = 0; k < n/nproc; k++) {
			p_local[k] = r[k] + beta * p_local[k];
		}
		MPI_Allgather(&p_local, n/nproc, MPI_DOUBLE, &p_global, n/nproc, MPI_DOUBLE, MPI_COMM_WORLD);

		i++;
	}


	printf("---6---");
	MPI_Allgather(&x, n/nproc, MPI_DOUBLE, answer, n/nproc, MPI_DOUBLE, MPI_COMM_WORLD);
	//for (i = 0; i < n/nproc; i++) {
	//	answer[i] = x[i];
	//}

	return answer;
}
int main(int argc, char* argv[]){
  int n;
  int m;
  double *value=NULL;
  int *colind=NULL;
  int *rbegin=NULL;
  double *answer=NULL;
  double *b=NULL;

  int nproc,rank,namelen;
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  MPI_Init(&argc,&argv);
  MPI_Comm_size(MPI_COMM_WORLD,&nproc);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Get_processor_name(processor_name,&namelen);
  printf("Process %d on %s out of %d\n",rank, processor_name, nproc);
  fflush(stdout);
  if (rank==MPI_RANK_ROOT){
  //printf("iiiiiiiiiii");
    readFile(argv[1],&n,&value,&colind,&rbegin,&b);
  //printf("jjjjjjjjjjj");
    scatterData(&n,&m,&value,&colind,&rbegin,&b,rank,nproc);
  //printf("kkkkkkkkkkk");
  }
  else{
    scatterData(&n,&m,&value,&colind,&rbegin,&b,rank,nproc);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  //printf("process %d on %s gets m=%d\n", rank, processor_name, m);

  double tv1,tv2;
  tv1=MPI_Wtime();
  //printf("ddddddddddd");
  answer=cg(n,value,colind,rbegin,b,rank,nproc);
  tv2=MPI_Wtime();
  if (rank==MPI_RANK_ROOT){
    printf("Process %d takes %.10f seconds\n",rank,tv2-tv1);
  }
 // if (value!=NULL) {free(value);}
 // if (colind!=NULL) {free(colind);}
 // if (rbegin!=NULL) {free(rbegin);}
 // if (b!=NULL) {free(b);}
  if (rank==MPI_RANK_ROOT) {
    writeFile(n,answer);
  }
 //if (answer!=NULL) {free(answer);}
  MPI_Finalize();
  return 1;  
}
