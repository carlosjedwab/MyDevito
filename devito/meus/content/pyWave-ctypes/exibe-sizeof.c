#include<stdio.h>
int main (){
	printf(" Lista tamanho (em bytes) dos tipos de dados \n");

	printf(" int         -> %lu \n", sizeof(int));
	printf(" long        -> %lu \n", sizeof(long));
	if (sizeof(int)!=sizeof(long))
		printf("    *** long e int tem tamanhos diferentes \n");
	printf(" float       -> %lu \n", sizeof(float));
	printf(" double      -> %lu \n", sizeof(double));
	printf(" long double -> %lu \n", sizeof(long double));
}
