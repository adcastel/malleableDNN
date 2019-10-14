



#include "omp.h"
#include <stdio.h>
#include <stdlib.h>


void test(int t,int * A,int N){

A = (int *) malloc(sizeof(int)*N);
//omp_set_num_threads(4-t);
#pragma omp parallel for num_threads(4-t)
for(int i = 0; i< N; i++){
	printf("thread padre %d Iteraciion %d thread %d/%d\n", t, i, omp_get_thread_num(),omp_get_num_threads());
	A[i] = t;
}
/*for(int i = 0; i< N; i++){

	printf("thread %d => A[%d] = %d\n",omp_get_thread_num(), i, A[i]);
}*/
}


int main(int argc, char * argv[]){

int N = 6;
int *A = (int *) malloc(sizeof(int)*N);
#pragma omp parallel num_threads(2)
{
#pragma omp for 
for(int i = 0; i< N; i++){
	printf("Iteraciion %d thread %d/%d\n", i, omp_get_thread_num(), omp_get_num_threads());
	test(omp_get_thread_num(),A,N);

}
}

/*#pragma omp parallel num_threads(2) shared(A) 
for(int i = 0; i< N; i++){

	printf("thread %d => A[%d] = %d\n",omp_get_thread_num(), i, A[i]);
}*/

}
