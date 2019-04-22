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
	double *answer;

	double gatherd_p[n];
	double r[n/nproc];
	double r_k1[n/nproc];
	double p[n/nproc];
	double x[n/nproc];
	//double k[n/nproc];
	double s[n/nproc];
	double alpha;
	int k = 0;
	int i, j;
	int ct, ct1, ct2;
	double max_r;
	double leftproduct_global, rightproduct_global;
	double beta;

	int rct, lct;
	double local_alpha;
	double left_product, right_product;
	double local_max;

	//initialize x 
	for(i = 0; i < (n/nproc); i++) {
		x[i] = 0;
		s[i] = 0;
		r[i] = 1.0;
		p[i] = 1.0;
	}

	


	MPI_Allgather(&p, n/nproc, MPI_DOUBLE, &gatherd_p, n/nproc, MPI_DOUBLE, MPI_COMM_WORLD);
	printf("**%d**\n", n/nproc);
	for (i = 0; i < n/nproc; i++) {
		printf("%f", p[i]);
	}

	while(k <= n) {
		printf("***1***\n");
		for (j =0; j < n/nproc; j++) {
			ct1 = rbegin[j];
			ct2 = rbegin[j+1] - 1;
			for (ct = ct1; ct <= ct2; ct++) {
				s[j] += value[ct] * gatherd_p[colind[ct]];
			}
		}
	printf("**%d**\n", n/nproc);
	for (i = 0; i < n/nproc; i++) {
		printf("%f", p[i]);
	}

		left_product = 0;
		printf("***2***\n");
		for(lct = 0; lct < n/nproc; lct++) {
			left_product += (r[lct] * r[lct]);
		}
		right_product = 0;
	printf("**%d**\n", n/nproc);
	for (i = 0; i < n/nproc; i++) {
		printf("%f", p[i]);
	}
		printf("***3***\n");
		for(rct = 0; rct < n/nproc; rct++) {
			right_product += (p[rct] * s[rct]);
		}
	printf("**%d**\n", n/nproc);
	for (i = 0; i < n/nproc; i++) {
		printf("%f", p[i]);
	}
		MPI_Allreduce(&left_product, &leftproduct_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(&right_product, &rightproduct_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		alpha = leftproduct_global / rightproduct_global;
		
	printf("**%d**\n", n/nproc);
	for (i = 0; i < n/nproc; i++) {
		printf("%f", p[i]);
	}
		printf("***4***\n");
		for (j=0; j < n/nproc; j++) {
			printf("%f\n", p[j]);
			printf("%f\n", x[j]);
			x[j] = x[j] + (alpha * p[j]);
			r_k1[j] = r[j] - alpha * s[j];
		}

		printf("***5***\n");
		local_max = r_k1[0];
		for (j=0; j < n/nproc; j++) {
			if(r_k1[j] > local_max) {
				local_max = r_k1[j];
			}
		}
		printf("%f\n", local_max);	
		MPI_Allreduce(&local_max, &max_r, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
		printf("***5.1***\n");
		if (max_r <= CONST_TOL) {
			break;
		}
		
		left_product = 0;
		printf("***6***\n");
		for(lct = 0; lct < n/nproc; lct++) {
			left_product += (r_k1[lct] * r_k1[lct]);
		}
		right_product = 0;
		printf("***7***\n");
		for(rct = 0; rct < n/nproc; rct++) {
			right_product += (r[rct] * r[rct]);
		}
		printf("***8***\n");
		MPI_Allreduce(&left_product, &leftproduct_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		printf("***8.1***\n");
		MPI_Allreduce(&right_product, &rightproduct_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		printf("***8.2***\n");
		beta = leftproduct_global / rightproduct_global;
		printf("***9***\n");
		for(j =0; j < n/nproc; j++) {
			p[j] = r_k1[j] + (beta * p[j]);
		}
		
		printf("***10***\n");
		MPI_Allgather(&p, n/nproc, MPI_DOUBLE, &gatherd_p, n/nproc, MPI_DOUBLE, MPI_COMM_WORLD);

		k += 1;
		printf("iterations: %d", k);
	}
	for (i=0; i < n/nproc; i++)
		answer[i] = x[i];

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
    readFile(argv[1],&n,&value,&colind,&rbegin,&b);
    scatterData(&n,&m,&value,&colind,&rbegin,&b,rank,nproc);
  }
  else{
    scatterData(&n,&m,&value,&colind,&rbegin,&b,rank,nproc);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  //printf("process %d on %s gets m=%d\n", rank, processor_name, m);

  double tv1,tv2;
  tv1=MPI_Wtime();
  answer=cg(n,value,colind,rbegin,b,rank,nproc);
  tv2=MPI_Wtime();
  if (rank==MPI_RANK_ROOT){
    printf("Process %d takes %.10f seconds\n",rank,tv2-tv1);
  }
  if (value!=NULL) {free(value);}
  if (colind!=NULL) {free(colind);}
  if (rbegin!=NULL) {free(rbegin);}
  if (b!=NULL) {free(b);}
  if (rank==MPI_RANK_ROOT) {
    writeFile(n,answer);
  }
  if (answer!=NULL) {free(answer);}
  MPI_Finalize();
  return 1;  
}
