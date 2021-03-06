// Optimized using shared memory and on chip memory
// Compile source: $- nvcc src/TokamakSimulation.cu -o nBody -lglut -lm -lGLU -lGL
// Run Executable: $- ./nBody
//To stop hit "control c" in the window you launched it from.
//Make movies https://gist.github.com/JPEGtheDev/db078e1b066543ce40580060eee9c1bf
#include <GL/freeglut.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "../includes/cudaErrCheck.cuh"

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

#define G 6.67408E-11
#define H 1.0

#define EYE 8.5
#define FAR 80.0

#define SHAPE_CT 24
#define SHAPE_SIZE 256
#define PATH "./objects/Tokamak_256.obj" //256 vertices-shape (for array simplicity)
#define N 16*16*16

//***********************
// TODO: 
//		Check units velocity calculation mag
//		ಠ_ಠ
//***********************

// Globals
float4 *p;
float3 *v, *f, *reactor,*r_GPU0, *r_GPU1;
float4 *p_GPU0, *p_GPU1;

//DeviceStruct stores GPU(s) info//
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

	float numc = 1.0;
	int separation = 360*8/N;
	int nr_circles = N/16;
	float r = 5.0;
	for(int num=0;num<N;num++){
		p[num].x = r*cos(separation*num);
		p[num].y = numc;
		p[num].z = r*sin(separation*num);
		p[num].w = MASS_PROTON;
		
		v[num].x = -1.5*p[num].x;
		v[num].y = 0.0;
		v[num].z = 1.5*sqrtf(r*r-p[num].x*p[num].x);

		f[num].x = 0.0;
		f[num].y = 0.0;
		f[num].z = 0.0;
		
		if(num%nr_circles==0){
			numc += 0.2;
		}
	}
}

void draw_picture(){
	glClear(GL_COLOR_BUFFER_BIT);
	glClear(GL_DEPTH_BUFFER_BIT);
	glColor3d(0.6,0.8,1.0);
	glPointSize(3.0);
	glBegin(GL_POINTS);
	for(int i=0; i<N; i++)
	{
		glVertex3f(p[i].x, p[i].y, p[i].z);
	}
	glEnd();
	
	//drawing ractor
	glColor3d(1.0,0.0,0.0);
	glPointSize(5.0);
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
	float inv_r = 1/sqrt(r2);
	
    float force  = (G*p0.w*p1.w)/(r2);// - (H*p0.w*p1.w)/(r2*r2);
    
    f.x = force*dx*inv_r;
    f.y = force*dy*inv_r;
    f.z = force*dz*inv_r;
    
    return(f);
}

__device__ float3 getMagForce(float4 p0, float3 v0, float3 dl_tail, float3 dl_head, float I){
	//dl is the section of wire
	float3 dB, dl;
	dl.x = dl_head.x-dl_tail.x;
	dl.y = dl_head.y-dl_tail.y;
	dl.z = dl_head.z-dl_tail.z;

	float rx = p0.x-dl_tail.x;
    float ry = p0.y-dl_tail.y;
	float rz = p0.z-dl_tail.z;

	float r2 = rx*rx+ry*ry+rz*rz;
	float inv_r2 = 1/r2;
	float inv_r = 1/sqrtf(r2);
	float3 rhat = {rx*inv_r, ry*inv_r, rz*inv_r};

	//(dl cross rhat)/r2 = force
	//gamma is mu0*I/4Pi which simplifies to Ie-7
	float gamma = I;
	dB.x = gamma*(dl.y*rhat.z-dl.z*rhat.y)*inv_r2;
	dB.y = gamma*(dl.z*rhat.x-dl.x*rhat.z)*inv_r2;
	dB.z = gamma*(dl.x*rhat.y-dl.y*rhat.x)*inv_r2;

	return (dB);
}

__global__ void getForcesMag(float4 *g_pos, float3 *vel, float3 *force, int offset, float3 *g_reactor){
	
	int id = threadIdx.x + blockDim.x*blockIdx.x;
	float3 total_force, B, dB, dl_tail, dl_head, velMe;
	float4 posMe;
	__shared__ float3 shared_r[BLOCK];

	total_force.x = B.x = 0.0;
	total_force.y = B.y = 0.0;
	total_force.z = B.z = 0.0;

	posMe.x = g_pos[id+offset].x;
	posMe.y = g_pos[id+offset].y;
	posMe.z = g_pos[id+offset].z;
	posMe.w = g_pos[id+offset].w;

	velMe.x = vel[id].x;
	velMe.y = vel[id].y;
	velMe.z = vel[id].z;
	
	for(int k=0;k<SHAPE_CT;k++){
		shared_r[threadIdx.x] = g_reactor[threadIdx.x + blockDim.x*k];
		__syncthreads();
		
		for(int j = 1; j<=SHAPE_SIZE; j++){
			dl_tail = shared_r[(j-1)];
			dl_head = shared_r[(j%SHAPE_SIZE)];
			dB = getMagForce(posMe, velMe, dl_tail, dl_head, 1.0); //current[i] =1
			
			B.x += dB.x;
			B.y += dB.y;
			B.z += dB.z;
		}
	}

	total_force.x = (velMe.y*B.z-velMe.z*B.y);
	total_force.y = (velMe.z*B.x-velMe.x*B.z);
	total_force.z = (velMe.x*B.y-velMe.y*B.x);

	if(id<N){
		force[id].x += total_force.x;
		force[id].y += total_force.y;
		force[id].z += total_force.z;
	}
}

__global__ void getForces(float4 *g_pos, float3 *force, int offset, int device_ct){
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
	    
    for(int j=0; j < gridDim.x*device_ct; j++)
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
			ERROR_CHECK( cudaMemcpy(r_GPU1, reactor, SHAPE_CT*SHAPE_SIZE*sizeof(float3), cudaMemcpyHostToDevice) );
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
			rtemp = i?r_GPU1:r_GPU0;
			cudaSetDevice( i );
			getForces<<<grid, block>>>(temp, dev[i].force, dev[i].offset, deviceCount);
			ERROR_CHECK( cudaPeekAtLastError() );

			getForcesMag<<<grid,block>>>(temp, dev[i].vel, dev[i].force,dev[i].offset,rtemp);
			ERROR_CHECK( cudaPeekAtLastError() );

			moveBodies<<<grid, block>>>(temp, dev[i].pos, dev[i].vel, dev[i].force, dev[i].offset);
			ERROR_CHECK( cudaPeekAtLastError() );
		}

		cudaDeviceSynchronize();

		if(deviceCount > 1){
			//cudaDeviceSynchronize();
			cudaSetDevice( 0 );
			ERROR_CHECK( cudaMemcpy(p_GPU1+dev[0].offset, dev[0].pos, dev[1].size*sizeof(float4), cudaMemcpyDeviceToDevice) );
			cudaSetDevice( 1 );
			ERROR_CHECK( cudaMemcpy(p_GPU0+dev[1].offset, dev[1].pos, dev[0].size*sizeof(float4), cudaMemcpyDeviceToDevice) );
			//cudaDeviceSynchronize();
		}
		cudaDeviceSynchronize();

		//To kill the draw comment out the next 7 lines.
		if(tdraw == DRAW){
			//cudaDeviceSynchronize();
			cudaSetDevice(0);
			ERROR_CHECK( cudaMemcpy(p, p_GPU0, N * sizeof(float4), cudaMemcpyDeviceToHost) );
			draw_picture();
			//break the for loop by closing window
			glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);
			glutMainLoopEvent();
			if(!glutGetWindow()){ break; }

			tdraw = 0;
		}
		tdraw++;
		time += dt;
	}
	
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("\n\nGPU time = %3.1f seconds\n", elapsedTime/1000.0);

	cudaDeviceSynchronize();
	cudaFree(p_GPU0);
	cudaFree(r_GPU0);
	cudaFree(dev[0].pos);
	cudaFree(dev[0].vel);
	cudaFree(dev[0].force);

	if(deviceCount>1){
		cudaFree(p_GPU1);
		cudaFree(r_GPU1);
		cudaFree(dev[1].pos);
		cudaFree(dev[1].vel);
		cudaFree(dev[1].force);
	}
	free(p);
	free(v);
	free(f);
	free(reactor);
}

void control(){	
	read_obj();
	set_initial_conditions();
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
	GLfloat light_ambient[]  = {0.2, 0.2, 0.2, 1.0};
	GLfloat light_diffuse[]  = {0.8, 0.8, 0.8, 1.0};
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
	glutMainLoopEvent();
	return 0;
}
