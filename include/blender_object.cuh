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

bool loadOBJ(const std::string path, std::vector<int>&obj_indx, std::vector<float3> &vertices){
    std::ifstream obj_read(path);

    std::string ID;
    float3 v;
    int line = 0;
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
            continue;
        }

        //Shape
        if(ID == "o"){
            obj_indx.push_back(line);
        }

        //Vertices
        else if(ID == "v"){
            line++;
            obj_read >> v.x >> v.y >> v.z;
            vertices.push_back(v);
        }
    }
    
    return true;
}