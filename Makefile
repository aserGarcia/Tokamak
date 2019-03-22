CC = nvcc

GLFLAGS = -lGL -lGLU -lm -lglut

output: main.o particle.o
	$(CC) -o output src/main.cu include/particle.cu $(GLFLAGS)

main.o: src/main.cu
	$(CC) -c src/main.cu $(GLFLAGS)

particle.o: include/particle.cu  include/particle.cuh
	$(CC) -c include/particle.cu

clean:
	rm *.o output