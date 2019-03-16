//Header Includes
#include <fstream>
#include <limits>
#include <cuda.h>
#include "blender_object.cuh"

/*--------------------
|   TO DO:
|       1. Parse uv and normal data
--------------------*/

//--------------------
// Load Blender Object
//--------------------
bool BlenderOBJ::loadOBJ(const std::string path){
    //file variables
    std::ifstream obj_read(path);

    //temporary vars
    std::string ID;
    int v_line = 0;

    //Check file opening
    if(obj_read.fail()){
        std::cout << "Cannot read file " << path <<".\nTerminating...\n";
        return false;
    }

    //Executes only if stream is successful
    
    while(!obj_read.eof()){
        obj_read >> ID;
        //Header
        if(ID == "#"){
            obj_read.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        }
            
        //Object
        if(ID == "o"){
            std::string name, nr;
            std::getline(obj_read, name, '.');
            std::getline(obj_read, nr, '\n');
            id[name] = std::stoi(nr);
        }

        //Vertices
        else if(ID == "v"){
            float3 v;
            obj_read >> v.x >> v.y >> v.z;
            vertices.push_back(std::vector<float3>());
            vertices[v_line++].push_back(v);
        }
        /*    
        //UVs
        else if(ID == "vt"){
            float2 vt;
            obj_read >> vt.x >> vt.y;
            uvs.push_back(std::vector<float2>());
            uvs[shape_count].push_back(vt);
        }

        //Normals
        else if(ID == "vn"){
            float3 n;
            obj_read >> n.x >> n.y >> n.z;
            normals.push_back(std::vector<float3>());
            normals[v_line++].push_back(n);
        }
        */
    }
    return true;
}


//--------------
// Constructor
//--------------
BlenderOBJ::BlenderOBJ(const std::string p, std::string s): path(p), name(s) {
    bool loaded = loadOBJ(path);
    if(loaded){
        std::cout << "Created object: " << name << std::endl;
    } else {
        std::cout << "Could not create object...\n";
    }
}

//--------------
// Name Setter
//--------------
void BlenderOBJ::setName(std::string n){
    std::cout<< "Name set from " << name;
    name = n;
    std::cout << " to " << name;
}

//--------------
// Name Setter
//--------------
std::vector<std::vector<float3> > BlenderOBJ::getVertices(){
    return vertices;
}


//--------------
// Destructor
//--------------
BlenderOBJ::~BlenderOBJ(){
    std::cout << "Destroyed object: " << name << std::endl;
}