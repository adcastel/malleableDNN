#tintorrum
#/home/adcastel/opt/mpich/bin/mpicc -O3 dnn.c -o dnn_vgg16_12th  -I/state/partition1/soft/intel/compilers_and_libraries/linux/mkl/include/  /state/partition1/soft//intel/compilers_and_libraries/linux/mkl/lib/intel64/libmkl_intel_lp64.a -L/state/partition1/soft/intel/compilers_and_libraries/linux/mkl/lib/intel64 -L /state/partition1/soft/intel/compilers_and_libraries_2019/linux/lib/intel64_lin/ -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -DTIMER -DVGG16 -fopenmp -DSTATIC -DSUMMARY -DPROGRESS


	#/opt/intel/compilers_and_libraries/linux/bin/intel64/icc -O3 malleable_dnn.c -o dnn  -I/opt/intel/compilers_and_libraries/linux/mkl/include/  /opt/intel/compilers_and_libraries/linux/mkl/lib/intel64/libmkl_intel_lp64.a -L/opt/intel/compilers_and_libraries/linux/mkl/lib/intel64 -L/opt/intel/compilers_and_libraries_2019.3.199/linux/compiler/lib/intel64_lin/ -lmkl_intel_thread -lmkl_core -liomp5 -lpthread  -lmkl_scalapack_lp64 -lm -DTIMER -DVGG16 -fopenmp -DSTATIC -DSUMMARY
#	gcc -O3 malleable_dnn.c -o dnn  -I/opt/intel/compilers_and_libraries/linux/mkl/include/  /opt/intel/compilers_and_libraries/linux/mkl/lib/intel64/libmkl_intel_lp64.a -L/opt/intel/compilers_and_libraries/linux/mkl/lib/intel64 -L/opt/intel/compilers_and_libraries_2019.3.199/linux/compiler/lib/intel64_lin/ -lmkl_intel_thread -lmkl_core -liomp5 -lpthread  -lmkl_scalapack_lp64 -lm -DTIMER -DVGG16 -fopenmp -DSTATIC -DSUMMARY


all: test malleableDNN 

BLISFLAGS := -I/home/adcastel/opt/blis-0-5-1/include/blis/ -L/home/adcastel/opt/blis-0-5-1/lib/ -lblis -lm
OMPFLAGS := -I/opt/intel/compilers_and_libraries_2017.1.132/linux/compiler/include/ -L/opt/intel/compilers_and_libraries_2017.1.132/linux/compiler/lib/intel64_lin/ -liomp5 -fopenmp
OMPFLAGS := -fopenmp
CC := gcc
#CC := /opt/intel/bin/icc
CFLAGS := #-Wall#-O3 #-Wall  

OBJECTS := 


ifeq ($(DEBUG), 1)
    CFLAGS = -g -Wall
endif
ifeq ($(DOUBLE), 1)
    CFLAGS += -DDOUBLE
endif

ifeq ($(NOIM2COL), 1)
    CFLAGS += -DNOIMTOCOL
endif
ifeq ($(NOGEMM), 1)
    CFLAGS += -DNOGEMM
endif

ifeq ($(PROGRESS), 1)
    CFLAGS += -DPROGRESS
endif
ifeq ($(TIMER), 1)
    CFLAGS += -DTIMER
endif
ifeq ($(SUMMARY), 1)
    CFLAGS += -DSUMMARY
endif
ifeq ($(TRAZA), 1)
    CFLAGS += -DTRAZA
    CFLAGS += -g -O0
    EXTRAEFLAGS := -L/home/adcastel/opt/extrae/lib/ -lomptrace -I/home/adcastel/opt/extrae/include
else
    CFLAGS += -O3
endif
test:
	$(CC) test.c -o test $(BLISFLAGS) $(OMPFLAGS) 
	
malleableDNN:
	$(CC) $(CFLAGS) malleable_dnn.c -o malleable_dnn $(BLISFLAGS) $(OMPFLAGS) $(EXTRAEFLAGS)
clean:
	rm *.so *.o test malleable_dnn


