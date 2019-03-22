#pragma once
/*---------------------------------------------------
| Author: Aser Garcia
| Date: March 10, 2019
| Description: Parser to load object into environment
|   of .obj file extension
|
---------------------------------------------------*/

#include <iostream>
#include <fstream>
#include <limits>
#include <cuda.h>
#include "blender_object.cuh"

/*--------------------
|   TO DO:
|       1. Parse uv and normal data
--------------------*/

bool loadOBJ(const std::string path, std::vector<float3> &vertices, float3 SHAPE_SEPARATOR){
    std::ifstream obj_read(path);

    std::string ID;
    int v_line = 0;
	float3 v;

    if(obj_read.fail()){
        std::cout << "Cannot read file " << path <<".\nTerminating...\n";
        return false;
    }
    
    while(!obj_read.eof()){
		obj_read >> ID;
		
        //Header
        if(ID == "#"){
			//skipline
            obj_read.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        }
        //Object
        if(ID == "o"){
            vertices.push_back(SHAPE_SEPARATOR);
        }

        //Vertices
        else if(ID == "v"){
            v = {0.0,0.0,0.0};
            obj_read >> v.x >> v.y >> v.z;
            vertices.push_back(v);
        }
    }
    return true;
}