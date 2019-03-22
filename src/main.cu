#include <GL/freeglut.h>
#include <GL/glu.h>
#include <GL/gl.h>
#include <random>
#include <cuda.h>
#include "../include/particle.cuh"
#include "../include/blender_object.cuh"

#define N_PARTICLES 20

#define BLOCK 256

#define XWindowSize 1024
#define YWindowSize 1024

#define DRAW 10
#define DAMP 1.0

#define DT 0.001
#define STOP_TIME 10.0

#define EYE 8.0
#define FAR 90.0

#define TOKAMAK_PATH "./objects/OBJ_tokamak.obj"

/*--------------------
|   TO DO:
|       1. Contniue "simulate" function
|			- getForces
--------------------*/


//Globals
float3 SHAPE_SEPARATOR = {1234.0f, 0.0f, 0.0f};

int window;
float3 color = {1.0, 1.0, 1.0};
float3 vel = make_float3(0.0,0.0,0.0);

//global arrays
std::vector<float3> obj_glob; //vertices contain separator for shapes
std::vector<float4> p_glob; //particles + mass
std::vector<float3> v_glob;
std::vector<float4> p_gpu0, p_gpu1;

//for devices
struct DeviceStruct {
	int deviceID;
	int size;
	int offset;
	std::vector<std::vector<float3> > v; //vertices
	std::vector<std::vector<float4> > p;
};


//------------------------------//
//    Creates charged particles   
//------------------------------//
void particleInit(){
	std::default_random_engine generator;
	std::uniform_real_distribution<float> distribution(0.0,1.0);

	p_glob.resize(N_PARTICLES);
	for(auto &p: p_glob){
		p = make_float4(2.0f, 0.0f,1.0f, 1.0f);
	}

	v_glob.resize(N_PARTICLES);
	for(auto &v: v_glob){
		v.x = distribution(generator);
		v.y = distribution(generator);
		v.z = distribution(generator);
	}
}

//---------------------------//
//    Displays the Particles   
//---------------------------//
void draw_object(){
	glClear(GL_COLOR_BUFFER_BIT);
	glClear(GL_DEPTH_BUFFER_BIT);
	
	glBegin(GL_POINTS);
	glColor3f(color.x, color.y, color.z);
	for(auto &vertex:obj_glob){
		if(vertex.x == SHAPE_SEPARATOR.x)
			continue;
		else
			glVertex3f(vertex.x,vertex.y,vertex.z);
	}
	glEnd();
	glutSwapBuffers();
}

void draw_picture(){
	
	glColor3f(1.0f, 0.0f,0.0f);
	
	for(auto &p: p_glob){
		glPushMatrix();
		glTranslatef(p.x,p.y,p.z);
		glutSolidSphere(0.1,10,10);
		glPopMatrix();
		glutSwapBuffers();
	}
}

void simulate(){
	float t = 0.0;
	int tdraw = 0;
	
	while(t<STOP_TIME){
		for(int v=1;v<obj_glob.size();v++){
			if(obj_glob[v].x == SHAPE_SEPARATOR.x)
				continue;
			else
				std::cout<<obj_glob[v-1].x<<obj_glob[v].x<<std::endl;
		}
		//particle.getB(vertices[0]);
		//particle.move();
		if(tdraw == DRAW-9){
			draw_picture();
			tdraw = 0;
		}
		tdraw++;
		t += DT;
	}
	std::cout<<"-----------FINISHED SIMULATING-----------\n";
}

//---------------------------------//
//   Set, Draw, Simulate, CleanUp   
//---------------------------------//
void control(){	
	const std::string path = TOKAMAK_PATH;
	bool loaded = loadOBJ(path, obj_glob, SHAPE_SEPARATOR);
	if(!loaded){
		std::cout<<"Could not load object, exiting\n";
		exit(1);
	}
	particleInit();
	//obj_glob call
	draw_object();
	
    simulate();
    draw_picture();
	
	std::cout<<"DONE, press SPACE then ENTER to exit"<<std::endl;
	std::cin.ignore(256, ' ');
	
}

//---------------------//
//    Sets Camera View   
//---------------------//
void Display(void){
	gluLookAt(EYE, EYE, EYE, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	control();
}

//-----------------//
//    Sets Window   
//-----------------//
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
	window = glutCreateWindow("Tokamak");
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