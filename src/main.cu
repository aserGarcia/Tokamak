#include <GL/freeglut.h>
#include <GL/glu.h>
#include <GL/gl.h>
#include <cuda.h>
#include "../include/particle.cuh"
#include "../include/blender_object.cuh"

#define N 1

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

//Globals
int window;
float3 color = {1.0, 1.0, 1.0};
float3 *particle_ptr;

//------------------------------//
//    Creates charged particles   
//------------------------------//
void particleInit(){
	particle_ptr = new float3[N];
	printf("\nNumber of particles = %d \n", N);

}

void simulate(BlenderOBJ & obj){}

//-----------------------------//
//    Destroys any remaining
//    objects before exit
//-----------------------------//
void cleanUp(){ 
	glutDestroyWindow(window);
	delete particle_ptr;
	std::cout<<"cleaned"<<std::endl;
}

//---------------------------//
//    Displays the Particles   
//---------------------------//
void draw_picture(BlenderOBJ& obj){
	std::vector<std::vector<float3> > vertices = obj.getVertices();

	glClear(GL_COLOR_BUFFER_BIT);
	glClear(GL_DEPTH_BUFFER_BIT);
	
	glBegin(GL_POINTS);
	glColor3f(color.x, color.y, color.z);
	for(auto shape:vertices){
		for(auto vertex:shape){
			glVertex3f(vertex.x,vertex.y,vertex.z);
		}
	}
	glEnd();
	glutSwapBuffers();
}

//---------------------------------//
//   Set, Draw, Simulate, CleanUp   
//---------------------------------//
void control(){	
	const std::string path = TOKAMAK_PATH;
	BlenderOBJ T(path, "Tokamak");
	
	draw_picture(T);
    simulate(T);
    //draw_picture();
	
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
	cleanUp();
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