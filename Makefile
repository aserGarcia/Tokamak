CC = nvcc

GLFLAGS = -lGL -lGLU -lm -lglut

output: main.o particle.o blender_obj.o
	$(CC) -o output src/main.cu include/particle.cu include/blender_object.cu $(GLFLAGS)

main.o: src/main.cu
	$(CC) -c src/main.cu $(GLFLAGS)

particle.o: include/particle.cu  include/particle.cuh
	$(CC) -c include/particle.cu

blender_obj.o: include/blender_object.cu  include/blender_object.cuh
	$(CC) -c include/blender_object.cu

clean:
	rm *.o output