// Optimized using shared memory and on chip memory
// nvcc nBodySimulation.cu -o nBody -lglut -lm -lGLU -lGL; ./nBody
//To stop hit "control c" in the window you launched it from.
#include <GL/glut.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "../cudaErrCheck.cuh"

#define NR_NEUTRONS 8 
#define NR_ELECTRONS 8 
#define NR_PROTONS 8 

//atomic mass (u)
#define MASS_PROTON 1.007276 
#define MASS_NEUTRON 1.008664
#define MASS_ELECTRON 5.485799e-4

#define BLOCK 256

#define XWindowSize 2500
#define YWindowSize 2500

#define DRAW 10
#define DAMP 1.0

#define DT 0.001
#define STOP_TIME 10.0

#define G 1.0
#define H 1.0

#define EYE 10.0
#define FAR 90.0

#define SHAPE_CT 24
#define SHAPE_SIZE 256
#define PATH "./objects/Tokamak_256.obj" //256 vertices-shape (for array simplicity)
#define N 8*8*8


//***********************
// TODO: 
//		insert reactor array into GPU (flatten?)
//***********************

// Globals
float4 *p;
float3 *v, *f, *reactor,*r_GPU0, *r_GPU1;
float4 *p_GPU0, *p_GPU1;

//DeviceStruct is to work on each GPU//
struct DeviceStruct {
	int deviceID;
	int size;
	int offset;
	float4 *pos;
	float3 *vel;
	float3 *force;
};

void read_obj(){
	FILE *fp = fopen(PATH, "r");
	char c, line[256];
	memset(line, 0, 256);
	reactor = (float3*)malloc(SHAPE_CT*SHAPE_SIZE*sizeof(float3));

    int j =0;
    while(fgets(line, sizeof(line), fp) != 0){
		c = line[0];
        if(c=='v'){
			sscanf(line, "%c %f %f %f\n", &c, &reactor[j].x, &reactor[j].y, &reactor[j].z);
			j++;
		}
    }
	fclose(fp);
}

void set_initial_conditions(){
	p = (float4*)malloc(N*sizeof(float4));
	v = (float3*)malloc(N*sizeof(float3));
	f = (float3*)malloc(N*sizeof(float3));

	int num,particles_per_side;
    float position_start, temp;
    float initail_seperation;

	temp = pow((float)N,1.0/3.0) + 0.99999;
	particles_per_side = temp;
	printf("\n cube root of N = %d \n", particles_per_side);
    position_start = -(particles_per_side -1.0)/2.0;
	initail_seperation = 2.0;
	for(int i=0; i<N; i++){
		p[i].w = 1.0;
	}
	num = 0;
	for(int i=-particles_per_side/2; i<particles_per_side/2; i++){
		for(int j=-particles_per_side/2; j<particles_per_side/2; j++){
			for(int k=-particles_per_side/2; k<particles_per_side/2; k++){
			    if(N <= num) break;
				p[num].x = position_start + i*initail_seperation;
				p[num].y = position_start + j*initail_seperation;
				p[num].z = position_start + k*initail_seperation;
				v[num].x = 0.0;
				v[num].y = 2.0;
				v[num].z = 0.0;
				num++;		
			}
		}
	}	
}

void draw_picture(){

	//glClear(GL_COLOR_BUFFER_BIT);
	//glClear(GL_DEPTH_BUFFER_BIT);
	glColor3d(1.0,1.0,1.0);
	glPointSize(3.0);
	glBegin(GL_POINTS);
	for(int i=0; i<N; i++)
	{
		glVertex3f(p[i].x, p[i].y, p[i].z);
	}
	glEnd();
	glutSwapBuffers();
}

void draw_obj(){

	//glClear(GL_COLOR_BUFFER_BIT);
	//glClear(GL_DEPTH_BUFFER_BIT);
	glColor3d(1.0,0.0,0.0);
	glPointSize(10.0);
	glBegin(GL_POINTS);
	for(int i=0; i<SHAPE_SIZE*SHAPE_CT; i++)
	{
		glVertex3f(reactor[i].x, reactor[i].y, reactor[i].z);
	}
	glEnd();
	glutSwapBuffers();
}

__device__ float3 getBodyBodyForce(float4 p0, float4 p1){
    float3 f;
    float dx = p1.x - p0.x;
    float dy = p1.y - p0.y;
    float dz = p1.z - p0.z;
    float r2 = dx*dx + dy*dy + dz*dz;
	float r = sqrt(r2);
	
    float force  = (G*p0.w*p1.w)/(r2) - (H*p0.w*p1.w)/(r2*r2);
    
    f.x = force*dx/r;
    f.y = force*dy/r;
    f.z = force*dz/r;
    
    return(f);
}

__host__ __device__ float3 getMagForce(float4 p0, float3 v0, float3 dl_tail, float3 dl_head, float I){
	//dl is the section of wire
	float3 f, B;
	float3 dl = {dl_head.x-dl_tail.x, dl_head.y-dl_tail.y, dl_head.z-dl_tail.z};

	float rx = p0.x-dl_tail.x;
    float ry = p0.y-dl_tail.y;
	float rz = p0.z-dl_tail.z;

	float r2 = rx*rx+ry*ry+rz*rz;
	float r = sqrtf(r2);
	float3 rhat = {rx/r, ry/r, rz/r};

	//(dl cross rhat)/r2 = force
	B.x = (dl.y*rhat.z-dl.z*rhat.y)/r2;
	B.y = (dl.z*rhat.x-dl.x*rhat.z)/r2;
	B.z = (dl.x*rhat.y-dl.y*rhat.x)/r2;

	f.x = (v0.y*B.z-v0.z*B.y);
	f.y = (v0.z*B.x-v0.x*B.z);
	f.z = (v0.x*B.y-v0.y*B.x);

	return f;
}

__global__ void getForces(float4 *g_pos, float3 *g_vel, float3 *force, float3 *reactor, int offset){
	int ii;
    float3 force_b2b, forceSum;
    float4 posMe;
    __shared__ float4 shPos[BLOCK];
    int id = threadIdx.x + blockDim.x*blockIdx.x;
    
    forceSum.x = 0.0;
	forceSum.y = 0.0;
	forceSum.z = 0.0;
		
	posMe.x = g_pos[id+offset].x;
	posMe.y = g_pos[id+offset].y;
	posMe.z = g_pos[id+offset].z;
	posMe.w = g_pos[id+offset].w;
	    
    for(int j=0; j < gridDim.x*2; j++)
    {
    	shPos[threadIdx.x] = g_pos[threadIdx.x + blockDim.x*j];
    	__syncthreads();
   
		#pragma unroll 32
        for(int i=0; i < blockDim.x; i++)	
        {
        	ii = i + blockDim.x*j;
		    if(ii != id+offset && ii < N) 
		    {
		    	force_b2b = getBodyBodyForce(posMe, shPos[i]);
			    forceSum.x += force_b2b.x;
			    forceSum.y += force_b2b.y;
			    forceSum.z += force_b2b.z;
		    }
	   	}
	}

	if(id <N){
	    force[id].x = forceSum.x;
	    force[id].y = forceSum.y;
	    force[id].z = forceSum.z;
    }
}

__global__ void moveBodies(float4 *g_pos, float4 *d_pos, float3 *vel, float3 * force, int offset){
    int id = threadIdx.x + blockDim.x*blockIdx.x;
    if(id < N){
	    vel[id].x += ((force[id].x-DAMP*vel[id].x)/d_pos[id].w)*DT;
	    vel[id].y += ((force[id].y-DAMP*vel[id].y)/d_pos[id].w)*DT;
	    vel[id].z += ((force[id].z-DAMP*vel[id].z)/d_pos[id].w)*DT;

		d_pos[id].x += vel[id].x*DT;
	    d_pos[id].y += vel[id].y*DT;
		d_pos[id].z += vel[id].z*DT;
		
		g_pos[id+offset].x = d_pos[id].x;
		g_pos[id+offset].y = d_pos[id].y;
		g_pos[id+offset].z = d_pos[id].z;
    }
}

void n_body(){
	int deviceCount;
	ERROR_CHECK( cudaGetDeviceCount ( &deviceCount ) );
	DeviceStruct* dev = (DeviceStruct*)malloc(deviceCount*sizeof(DeviceStruct));
	
	for(int i = 0; i<deviceCount; i++){
		cudaSetDevice(i);
		if(i==0){
			ERROR_CHECK( cudaMalloc(&p_GPU0, N*sizeof(float4)) );
			ERROR_CHECK( cudaMemcpy(p_GPU0, p, N*sizeof(float4), cudaMemcpyHostToDevice) );
			
			ERROR_CHECK( cudaMalloc(&r_GPU0, SHAPE_CT*SHAPE_SIZE*sizeof(float3)) );
			ERROR_CHECK( cudaMemcpy(r_GPU0, reactor, SHAPE_CT*SHAPE_SIZE*sizeof(float3), cudaMemcpyHostToDevice) );
		}
		if(i==1){
			ERROR_CHECK( cudaMalloc(&p_GPU1, N*sizeof(float4)) );
			ERROR_CHECK( cudaMemcpy(p_GPU1, p, N*sizeof(float4), cudaMemcpyHostToDevice) );

			ERROR_CHECK( cudaMalloc(&r_GPU1, SHAPE_CT*SHAPE_SIZE*sizeof(float3)) );
			ERROR_CHECK( cudaMemcpy(r_GPU0, reactor, SHAPE_CT*SHAPE_SIZE*sizeof(float3), cudaMemcpyHostToDevice) );
		}

		dev[i].deviceID = i;
		dev[i].size = N/deviceCount;
		dev[i].offset = i*N/deviceCount;
		ERROR_CHECK( cudaMalloc(&dev[i].pos, dev[i].size*sizeof(float4)) );
		ERROR_CHECK( cudaMalloc(&dev[i].vel, dev[i].size*sizeof(float3)) );
		ERROR_CHECK( cudaMalloc(&dev[i].force, dev[i].size*sizeof(float3)) );
		
		ERROR_CHECK( cudaMemcpy(dev[i].pos, p+dev[i].offset, dev[i].size*sizeof(float4), cudaMemcpyHostToDevice) );
		ERROR_CHECK( cudaMemcpy(dev[i].vel, v+dev[i].offset, dev[i].size*sizeof(float3), cudaMemcpyHostToDevice) );
		ERROR_CHECK( cudaMemcpy(dev[i].force, f+dev[i].offset, dev[i].size*sizeof(float3), cudaMemcpyHostToDevice) );
	}

	dim3 block(BLOCK);
	dim3 grid((N/deviceCount - 1)/BLOCK + 1);
	
	float dt;
	int   tdraw = 0; 
	float time = 0.0;
	float elapsedTime;
	
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	
	dt = DT;
    
	while(time < STOP_TIME){	
		for(int i = 0; i < deviceCount; i++){
			float4 *temp;
			float3 *rtemp;
			temp = i?p_GPU1:p_GPU0;
			rtemp = i?r_GPU0:r_GPU1;
			cudaSetDevice( dev[i].deviceID );
			getForces<<<grid, block>>>(temp, dev[i].vel, dev[i].force, rtemp, dev[i].offset);
			
			ERROR_CHECK( cudaPeekAtLastError() );
			ERROR_CHECK( cudaMemcpy(f+dev[i].offset, dev[i].force,dev[i].size*sizeof(float3), cudaMemcpyDeviceToHost) );
			ERROR_CHECK( cudaMemcpy(p+dev[i].offset, dev[i].pos, dev[i].size*sizeof(float4), cudaMemcpyDeviceToHost) );
			ERROR_CHECK( cudaMemcpy(v+dev[i].offset, dev[i].vel, dev[i].size*sizeof(float3), cudaMemcpyDeviceToHost) );
			for(int me=dev[i].offset; me<(dev[i].size+dev[i].offset);me++){
				
				float3 forceSum = make_float3(0.0,0.0,0.0);
				float3 force_mag;
				float4 posMe = p[me];
				float3 velMe = v[me];
				for(int k=0;k<SHAPE_CT;k++){
					for(int j = 1; j<=SHAPE_SIZE; j++){
						float3 dl_tail = reactor[(j-1)+SHAPE_SIZE*k];
						float3 dl_head = reactor[(j%SHAPE_SIZE)+SHAPE_SIZE*k];
						
						force_mag = getMagForce(posMe, velMe, dl_tail, dl_head, 1.0); //current[i] =1
						forceSum.x += force_mag.x;
						forceSum.y += force_mag.y;
						forceSum.z += force_mag.z;
					}
				}
				f[me].x += forceSum.x;
				f[me].y += forceSum.x;
				f[me].z += forceSum.z;
			}
			ERROR_CHECK( cudaMemcpy(dev[i].vel, v+dev[i].offset, dev[i].size*sizeof(float3), cudaMemcpyHostToDevice) );
			ERROR_CHECK( cudaMemcpy(dev[i].force, f+dev[i].offset, dev[i].size*sizeof(float3), cudaMemcpyHostToDevice) );
			
			moveBodies<<<grid, block>>>(temp, dev[i].pos, dev[i].vel, dev[i].force, dev[i].offset);
			ERROR_CHECK( cudaPeekAtLastError() );
		}

		cudaDeviceSynchronize();
		
		if(deviceCount > 1){
			cudaSetDevice( 0 );
			ERROR_CHECK( cudaMemcpy(p_GPU1+dev[0].offset, dev[0].pos, dev[1].size*sizeof(float4), cudaMemcpyDeviceToDevice) );
			cudaSetDevice( 1 );
			ERROR_CHECK( cudaMemcpy(p_GPU0+dev[1].offset, dev[1].pos, dev[0].size*sizeof(float4), cudaMemcpyDeviceToDevice) );
		}

		cudaDeviceSynchronize();


		//To kill the draw comment out the next 7 lines.
		if(tdraw == DRAW){
			cudaSetDevice(1);
			ERROR_CHECK( cudaMemcpy(p, p_GPU1, N*sizeof(float4), cudaMemcpyDeviceToHost) );
			draw_obj();
			draw_picture();
			tdraw = 0;
		}
		tdraw++;
		time += dt;
	}
	
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("\n\nGPU time = %3.1f milliseconds\n", elapsedTime);


	cudaFree(p_GPU0);
	cudaFree(p_GPU1);
	cudaFree(r_GPU0);
	cudaFree(r_GPU1);
	for(int i = 0; i<deviceCount; i++){
		cudaFree(dev[i].pos);
		cudaFree(dev[i].vel);
		cudaFree(dev[i].force);
	}
	free(p);
	free(v);
	free(f);
	free(reactor);
}

void control(){	
	read_obj();
	set_initial_conditions();
	draw_obj();
	draw_picture();
    n_body();
	
	printf("\n DONE \n");
}

void Display(void){
	gluLookAt(EYE, EYE, EYE, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	control();
}

void reshape(int w, int h){
	glViewport(0, 0, (GLsizei) w, (GLsizei) h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glFrustum(-0.2, 0.2, -0.2, 0.2, 0.2, FAR);
	glMatrixMode(GL_MODELVIEW);
}

int main(int argc, char** argv){
	glutInit(&argc,argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGB);
	glutInitWindowSize(XWindowSize,YWindowSize);
	glutInitWindowPosition(0,0);
	glutCreateWindow("2 Body 3D");
	GLfloat light_position[] = {1.0, 1.0, 1.0, 0.0};
	GLfloat light_ambient[]  = {0.0, 0.0, 0.0, 1.0};
	GLfloat light_diffuse[]  = {1.0, 1.0, 1.0, 1.0};
	GLfloat light_specular[] = {1.0, 1.0, 1.0, 1.0};
	GLfloat lmodel_ambient[] = {0.2, 0.2, 0.2, 1.0};
	GLfloat mat_specular[]   = {1.0, 1.0, 1.0, 1.0};
	GLfloat mat_shininess[]  = {10.0};
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glShadeModel(GL_SMOOTH);
	glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);
	glLightfv(GL_LIGHT0, GL_POSITION, light_position);
	glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
	glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
	glLightModelfv(GL_LIGHT_MODEL_AMBIENT, lmodel_ambient);
	glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
	glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess);
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glEnable(GL_COLOR_MATERIAL);
	glEnable(GL_DEPTH_TEST);
	glutDisplayFunc(Display);
	glutReshapeFunc(reshape);
	glutMainLoop();
	return 0;
}
