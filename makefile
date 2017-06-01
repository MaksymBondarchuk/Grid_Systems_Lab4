EXECS=main
MPICC?=mpicc

all: ${EXECS}

main: main.c
	${MPICC} -o ${EXECS} ${EXECS}.c linalg.c -lm

run:
	mpirun -np 4 ./${EXECS}

clean:
	rm ${EXECS}