#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include "util.h"


double** matrix_mult(	// Return the the local block of C=A*B computed by Fox's algorithm
  int np,		// size of local blocks
  int row_rank, 	// rank in row_comm, represents which COLUMN the proc is
  MPI_Comm row_comm, 	// the communicator of the row
  int col_rank, 	// rank in col_comm, represents which ROW the proc is
  MPI_Comm col_comm, 	// the communicator of the column
  int nproc_row, 	// number of proc per row
  double **mA, 		// local block of input A
  double **mB) 		// local block of input B
{
  MPI_Status status;
  double **C;
  //double **temp;
  double test = 8.0;
  int i, j, k, l;
  double temp[np];
  double newB[np];
  allocateMatrix(np,&C);		
  //allocateMatrix(np,&temp);		
/*
  Fill in your code
*/
	// Initialize C to 0
	for(i = 0; i < np; i++) {
		for(j = 0; j < np; j++) {
			C[i][j] = 0;
		}
	}
	
	for(k = 0; k < np * nproc_row; k++) {
		for(i = 0; i < np; i++) {
			temp[i] = mA[i][(i+k) % np];

		}
		for(i = 0; i < np; i++) {
			MPI_Bcast(&(temp[i]), 1 , MPI_DOUBLE, ((((col_rank * np) + i + k) % (np * nproc_row)) / np), row_comm);
		}
		for(i = 0; i < np; i++) {
			for(j = 0; j < np; j++) {
				C[i][j] += temp[i] * mB[i][j];
			}
		}
		if(col_rank - 1 < 0) {
			for(j = 0; j < np; j++) {
				newB[j] = mB[0][j];
			}
			for(j = 0; j < np - 1; j++) {
				for(l = 0; l < np; l++) {
					mB[j][l] = mB[j+1][l];		
				}
			}
			MPI_Sendrecv_replace(newB, np, MPI_DOUBLE, nproc_row - 1, 0, (col_rank + 1) % nproc_row, 0, col_comm, &status);
			for(j = 0; j < np; j++) {
				mB[np - 1][j] = newB[j];
			}
		}
		else { 
			for(j = 0; j < np; j++) {
				newB[j] = mB[0][j];
			}
			for(j = 0; j < np - 1; j++) {
				for(l = 0; l < np; l++) {
					mB[j][l] = mB[j+1][l];		
				}
			}
			MPI_Sendrecv_replace(newB, np, MPI_DOUBLE, (col_rank - 1) % nproc_row , 0, (col_rank + 1) % nproc_row, 0, col_comm, &status);
			for(j = 0; j < np; j++) {
				mB[np - 1][j] = newB[j];
			}
		}

	}
	
  return C;
}

int main(int argc, char **argv)
{
  int	rank, nproc;
  int	nproc_row;	// number of processes per row;
  int	n;
  int	np;
  double **mA=NULL;
  double **mB=NULL;
  double **mC=NULL;
  double time;
  char	procName[200];
  int	procNameLen;
  int	row_rank;	//rank in row_comm, represent which COLUMN the proc is
  int	col_rank;	//rank in col_comm, represent which ROW the proc is
  MPI_Comm row_comm;	//the communicator of the row
  MPI_Comm col_comm;	//the communicator of the column
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&nproc);
  MPI_Get_processor_name(procName,&procNameLen);
  if (nproc==1){
    nproc_row=1;
  }
  else if(nproc==4){
    nproc_row=2;
  }
  else if (nproc==9){
    nproc_row=3;
  }
  else if (nproc==16){
    nproc_row=4;
  }
  else{
    printf("-np: should be either 1, 4, 9 or 16\n");
    exit(-1);
  }//ugly but simple
  if (argc!=2){
    if (rank==0){
      printf("Usage: ./mmmpi Size_of_Matrix\n");
    }
    np=128;
    n=np*nproc_row;
  }
  else{
    n=atoi(argv[1]);
    np=n/nproc_row;
  }
  printf("Proc %d of %d is running on %s\n",rank,nproc,procName);


//Generate row_comm & col_comm, get row_rank & col_rank.
/*
Fill in your code
*/
MPI_Comm_split(MPI_COMM_WORLD, rank / nproc_row, rank, &row_comm);
MPI_Comm_split(MPI_COMM_WORLD, rank % nproc_row, rank, &col_comm);

MPI_Comm_rank(row_comm, &row_rank);
MPI_Comm_rank(col_comm, &col_rank);

printf("Proc %d row_rank %d, col_rank %d\n",rank, row_rank, col_rank);

//Allocate the local bloacks of input A&B
//The allocation is guaranteed to be contiguous in memory
//So you can  use &A[0][0] as the starting address to transfer the entire block.
  allocateMatrix(np,&mA);
  allocateMatrix(np,&mB);

//Initialize the local A&B
  generateLocalMatrix(np,row_rank, col_rank,nproc_row,mA,mB);

//Compute and time  C=A*B via Fox's algorithm
  MPI_Barrier(MPI_COMM_WORLD);
  time=MPI_Wtime();
  mC=matrix_mult(np,row_rank,row_comm,col_rank,col_comm,nproc_row,mA,mB);
  MPI_Barrier(MPI_COMM_WORLD);
  time=MPI_Wtime()-time;

//You can verify your answer via the following two functions
//verifyAnswerFast(np,rank,row_rank, col_rank, nproc_row,mC); 
verifyAnswerComplete(np,rank,row_rank,col_rank,nproc_row,mC);
  if (rank==0){
    printf("Time=%19.15f\n",time);
  }
  freeMatrix(n,mA);
  freeMatrix(n,mB);
  freeMatrix(n,mC);
  MPI_Finalize();
  return(0);
}
