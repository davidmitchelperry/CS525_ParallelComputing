#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "util.h"
#include <math.h>

int* sort(int *data, int np, int rank, int nproc){

	//Quick/Merge Sort local data
	localMergeSort(data, np);
	int splitters[nproc - 1];
	int global_splitters[nproc - 1];
	int i, j, ct;

	//Select&Bcast splitters
	for(i = 0; i < nproc - 1; i++) {
		splitters[i] = data[(i + 1) * (int)floor(np / (nproc))];
	}

	int recv_splitters[nproc * (nproc - 1)];
	MPI_Gather(splitters, nproc - 1, MPI_INT, recv_splitters, nproc - 1, MPI_INT, 0, MPI_COMM_WORLD);

	if(rank == 0) {
		localMergeSort(recv_splitters, nproc * (nproc - 1));
		for(i = 0; i < nproc - 1; i++) {
			global_splitters[i] = recv_splitters[(nproc - 1) * (i + 1)];  
		}

	}

	MPI_Bcast(global_splitters, nproc - 1, MPI_INT, 0, MPI_COMM_WORLD);

	//Split local array
	int sendcounts[nproc];
	int all_sendcounts[nproc * nproc];
	int rcounts[nproc];
	int recvcounts[nproc];
	
	int rdispls[nproc];
	int sdispls[nproc];
	for(i = 0; i < nproc; i++) {
		rdispls[i] = i * nproc;
		rcounts[i] = nproc;
	}

	j = 0;
	sdispls[0] = 0;
	for(i = 0; i < nproc - 1; i++) {
		ct = 0;
		sdispls[i] = j;
		while(data[j] <= global_splitters[i]) {
			j++;
			ct++;
		}
		sendcounts[i] = ct;
	}
	sendcounts[nproc - 1] = np - j;
	sdispls[nproc - 1] = j;

	MPI_Allgatherv(&sendcounts, nproc, MPI_INT, &all_sendcounts, rcounts, rdispls, MPI_INT, MPI_COMM_WORLD);

	int recvcounts_total[nproc]; 
	for(i = 0; i < nproc; i++) {
		recvcounts[i] = all_sendcounts[rank +(i * nproc)];
		recvcounts_total[i] = 0;
		for(j =0; j < nproc; j++) {
			recvcounts_total[i] += all_sendcounts[i +(nproc * j)];
		}
	}

	rdispls[0] = 0;
	for( i = 1; i < nproc; i++) {
		rdispls[i] = rdispls[i - 1] + recvcounts[i - 1];
	}
			
	//All-to-all personalized communcation
	int recv_total = 0;
	for(i = 0; i < nproc; i++) {
		recv_total += recvcounts[i];
	}
	int *recv_buffer = malloc(sizeof(int) * recv_total);
		
	MPI_Alltoallv(data, sendcounts, sdispls, MPI_INT, recv_buffer, recvcounts, rdispls, MPI_INT, MPI_COMM_WORLD);

	//Local Merge Sort again
	localMergeSort(recv_buffer, recv_total);

	//Gather the result to root
	int *sortedData = malloc(sizeof(int) * (np * nproc));
	rdispls[0] = 0;
	for( i = 1; i < nproc; i++) {
		rdispls[i] = rdispls[i - 1] + recvcounts_total[i - 1];
	}
	MPI_Gatherv(recv_buffer, recv_total, MPI_INT, sortedData, recvcounts_total, rdispls, MPI_INT, 0, MPI_COMM_WORLD);
	return sortedData;
		
}

int main(int argc, char *argv[]){
    int n;      //length of the global int array
    int np;     //length of the local array
    int rank;
    int nproc;
    int *t=NULL;
    int *data=NULL;  //where the data is at
    double time;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&nproc);
    n=atoi(argv[1]);
    np=n/nproc;
    data=(int*)malloc(sizeof(int)*np);
    if (rank==0){
        t=generateData(n);
    }
    int i = 0;
    MPI_Scatter(t,np,MPI_INT,data,np,MPI_INT,0,MPI_COMM_WORLD);
    time=MPI_Wtime();
    t=sort(data,np,rank,nproc);
    time=MPI_Wtime()-time;
    if (rank==0){
        outputFile(t,n);
        printf("Time=%19.15f\n",time);
    }
    if (t!=NULL) {free(t);t=NULL;}
    if (data!=NULL) {free(data);data=NULL;}
    MPI_Finalize();
    return 0;
}
