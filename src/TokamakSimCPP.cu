//Run with:
//nvcc TokamakSimCPP.cu glad.c -lglfw -lGLU -lGL -lglut -lX11 -lpthread -lXrandr -lXi -ldl; ./a.out

//     Visualization Includes
//-----------------------------
#include <glad/glad.h>
#include <GLFW/glfw3.h>

//    C++ Includes
//-----------------------------
#include <iostream>

//--------------------------------------
//          GLOBAL VARS
//--------------------------------------
class Structure{

private:
    float3* structure;    
public:
    void read_obj(std::string path);

};

void Structure::read_obj(std::string path){
    
}

//--------------------------------------
//          FUNCTION PROTOTYPES
//--------------------------------------
void framebuffer_size_callback(GLFWwindow* widnow, int width, int height);
void processInput(GLFWwindow* window);


int main() {
    int WIDTH = 800; int HEIGHT = 600;
    glfwInit();


    //------------------------------------------
    //          CONFIGURING OPENGL
    // -----------------------------------------
    //ensuring version 3.3
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,3);

    //smaller subset of features with core profile
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);


    //-----------------------------------------
    //         WINDOW OBJECT    
    //-----------------------------------------
    GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "Simulation", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create window... Exiting" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    //-----------------------------------------
    //          WINDOW MANAGER
    //-----------------------------------------
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to init GLAD" << std::endl;
        return -1;
    }

    //------------------------------------------
    //          VIEWING WINDOW  
    //------------------------------------------
    //setting lower left corner at the origin
    //size 800 by 600
    glViewport(0,0, WIDTH,HEIGHT);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);


    //-----------------------------------------
    //              RENDER LOOP
    //-----------------------------------------
    while(!glfwWindowShouldClose(window))
    {
        processInput(window);

        //make render commands on window here

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;

}


//-------------------------------------------
//         FUNCTION DEFINITIONS
//-------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    glViewport(0,0, width, height);
}

void processInput(GLFWwindow* window)
{
    if(glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
    {
        glfwSetWindowShouldClose(window, true);
    }
}