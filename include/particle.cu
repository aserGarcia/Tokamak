#include <iostream>
#include "particle.cuh"
#include "vec_helpers.cuh"

Particle::Particle(float3 p, float3 v, int c){pos = p; vel = v; charge = c;}
Particle::~Particle(){}

float3 Particle::getPosition(){ return pos; }
float3 Particle::getVelocity(){ return vel; }

//magnetic force calculation
void Particle::getB(std::vector<float3> vertices, std::vector<int>obj_indx){
    float3 B, crossv, dl, rv; //B (magnetic field) force

    float r;
    int k = 1;
    for (int i = 1; i<vertices.size(); i++){
        /* (if-else) Since all the shapes are in the same vector
        | with no separation, there is a vector of indices
        | that separates the shapes.
        */

        if(i == obj_indx[k]){ 
            dl = makeVector(vertices[i],vertices[obj_indx[k-1]]);
            rv = makeVector(vertices[i], pos);
            i++;
            k++;
        }
        else{
            rv = makeVector(vertices[i], pos);
            dl = makeVector(vertices[i-1],vertices[i]);
        }

        r = length(rv);

        float3 unit_r = unit(rv);
        crossv = cross(dl, unit_r);

        scale(mu_0/r*r, crossv);
        B.x += crossv.x;
        B.y += crossv.y;        
        B.z += crossv.z;

    }

    float3 new_vel = cross(vel, B);
    scale(charge*dt, new_vel);    
    vel = new_vel;
}

void Particle::move(){
    std::cout<<"vel";
    printVector(vel);
    scale(dt, vel);
    std::cout<<"scaled vel";
    printVector(vel);
    pos = vel;
}

