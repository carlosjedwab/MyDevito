CC=gcc
TARGET=iso2d
SRC=iso2d.c
CFLAGS=-Wall -O3

CLANG=clang
CLANGFLAGS=-fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_75 -Wall -lm -O3

TARGET1=iso2d-omp1
SRC1=iso2d-omp1.c

TARGET2=iso2d-omp2
SRC2=iso2d-omp2.c

all: seq omp1 omp2

seq:
	$(CC) $(SRC) -o $(TARGET) $(CFLAGS)

omp1:
	$(CLANG) $(SRC1) -o $(TARGET1) $(CLANGFLAGS)

omp2:
	$(CLANG) $(SRC2) -o $(TARGET2) $(CLANGFLAGS)

clean:
	rm -f $(TARGET) $(TARGET1) $(TARGET2)
