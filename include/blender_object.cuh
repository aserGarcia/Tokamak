#pragma once
/*---------------------------------------------------
| Author: Aser Garcia
| Date: March 10, 2019
| Description: Parser to load object into environment
|   of .obj file extension
|
---------------------------------------------------*/

/*---------------------------------------------------
| Class: BlenderOBJ
| Description: Creates a blender object for render
| Variables:
|	1. path - File path to read
|	2. vector<vector float3> name 
|		- Similar to an ID
|	3. vector<vector float3> vertices 
|		- Stores vertex points, diff shapes
|	4. vector<vector float2> uvs 
|		- coordinates of 2D wrap onto 3D object, diff shapes
|	5. normals - coordinates for normals, diff shapes
|		- name.z for indexing vectors
| Function:
| 	1. loadOBJ - reads the file (used in constructor)
---------------------------------------------------*/
#include <iostream>
#include <vector>
#include <string>
#include <map>

class BlenderOBJ {
	private:
		const std::string path;
		std::string name;
		std::map<std::string,int> id;
		std::vector<std::vector<float3> > vertices;
		std::vector<std::vector<float3> > normals;
		std::vector<std::vector<float2> > uvs;

		bool loadOBJ(const std::string);
	public:
		BlenderOBJ(const std::string, std::string);
		void setName(std::string);
		std::vector<std::vector<float3> > getVertices();
		~BlenderOBJ();
};