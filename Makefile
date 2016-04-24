
COMMON_FLAGS=-O3 -arch=compute_30 -code=sm_30

CC=nvcc
CFLAGS=${COMMON_FLAGS}
NVCCFLAGS=${COMMON_FLAGS}
LIBS=

# Update OUTPUT_PATTERNS later to match any kind of output from running the application
OUTPUT_PATTERNS= *.stdout

TARGETS=gpu

# These can stay the same
all:	${TARGETS}

clean:
	rm -f *.o ${TARGETS} ${OUTPUT_PATTERNS}

# Generation of any binaries
gpu: gpu.o
	${CC} -o $@ ${NVCCLIBS} gpu.o

common: common.o
	${CC} -o $@ ${NVCCLIBS} common.o

# Generation of any object files
gpu.o: gpu.cu common.h
	${CC} -c ${NVCCFLAGS} gpu.cu
common.o: common.cu common.h
	${CC} -c ${NVCCFLAGS} common.cu
