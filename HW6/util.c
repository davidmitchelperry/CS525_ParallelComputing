#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int* generateData(int size){
    fprintf(stdout,"Generating %d numbers in[0,%d]\n",size, RAND_MAX);
    int *t;
    int i;
    t=(int*)malloc(sizeof(int)*size);
    srand(12345);
    for(i=0;i<size;i++){
        t[i]=rand();
    }
    return t;
}
void outputFile(int *data, int n){
    FILE *fp=fopen("output.dat","w");
    int i;
    for(i=0;i<n;i++){
        fprintf(fp,"%d\n",data[i]);
    }
    fclose(fp);
}

void mergeSort(int *d, int *t, int l, int r){
    int mid=(l+r)/2;
//printf("l=%d,r=%d,m=%d\n",l,r,mid);
    if (l+1>=r){
        return;
    }
    mergeSort(d,t,l,mid);
    mergeSort(d,t,mid,r);
    int p=l;
    int q=mid;
    int k=l;
    while((p<mid)&&(q<r)){
        if (d[p]<d[q]){
            t[k++]=d[p++];
        }
        else{
            t[k++]=d[q++];
        }
    }
    if (p==mid){
        while(q<r){ t[k++]=d[q++];}
    }
    else{
        while(p<mid){t[k++]=d[p++];}
    }
    for(k=l;k<r;k++){
        d[k]=t[k];
    }
}
void localMergeSort(int *data, int n){
    int *t=(int*)malloc(sizeof(int)*n);
    mergeSort(data,t,0,n);
    free(t);
}
