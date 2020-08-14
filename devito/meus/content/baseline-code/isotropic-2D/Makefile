CC=gcc

TARGET_CD=iso2d_cd
SRC_CD=iso2d_cd.c

TARGET_VD=iso2d_vd
SRC_VD=iso2d_vd.c

2DARRAY_TARGET=iso2d-array2d
2DARRAY_SRC=iso2d-array2d.c

CFLAGS=-Wall -O3

all: constant_density variable_density

constant_density:
	$(CC) $(SRC_CD) -o $(TARGET_CD) $(CFLAGS)

variable_density:
	$(CC) $(SRC_VD) -o $(TARGET_VD) $(CFLAGS)

array2d:
	$(CC) $(2DARRAY_SRC) -o $(2DARRAY_TARGET) $(CFLAGS)

clean:
	rm -f $(TARGET_CD) $(TARGET_VD) $(2DARRAY_TARGET)
