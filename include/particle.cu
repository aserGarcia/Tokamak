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
            //dl = makeVector(vertices[i],vertices[obj_indx[k-1]]);
            dl.x = vertices[i].x-vertices[obj_indx[k-1]].x;
            dl.y = vertices[i].y-vertices[obj_indx[k-1]].y;
            dl.z = vertices[i].z-vertices[obj_indx[k-1]].z;
            
            rv.x = vertices[i].x-pos.x;
            rv.y = vertices[i].y-pos.y;
            rv.z = vertices[i].z-pos.z;
            
            i++;
            k++;
        }
        else{
            //rv = makeVector(vertices[i], pos);
            rv.x = vertices[i].x-pos.x;
            rv.y = vertices[i].y-pos.y;
            rv.z = vertices[i].z-pos.z;
            //dl = makeVector(vertices[i-1],vertices[i]);
            dl.x = vertices[i-1].x-vertices[i].x;
            dl.y = vertices[i-1].y-vertices[i].y;
            dl.z = vertices[i-1].z-vertices[i].z;
        }

        r = length(rv);
        float3 unit_r = {rv.x/r,rv.y/r,rv.z/r};
            //crossv = cross(dl, unit_r);
        crossv.x = dl.y*unit_r.z-dl.z*unit_r.y;
        crossv.y = dl.z*unit_r.x-dl.x*unit_r.z;
        crossv.z = dl.x*unit_r.y-dl.y*unit_r.x;

        //scale(mu_0/r*r, crossv);
        crossv.x = crossv.x*(mu_0/(r*r));
        crossv.y = crossv.y*(mu_0/(r*r));
        crossv.z = crossv.z*(mu_0/(r*r));

        B.x += crossv.x;
        B.y += crossv.y;        
        B.z += crossv.z;
    }
    //float3 force = cross(vel, B);
    printVector(vel, "vel");
    float3 force;
    force.x = vel.y*B.z-vel.z*B.y;
    force.y = vel.z*B.x-vel.x*B.z;
    force.z = vel.x*B.y-vel.y*B.x;
    
    //scaling
    force.x = force.x*(charge/mass);
    force.y = force.y*(charge/mass);
    force.z = force.z*(charge/mass);

    float3 acc;
    acc.x = force.x;
    acc.y = force.y;
    acc.z = force.z;

    vel.x = vel.x+acc.x*dt;
    vel.y = vel.y+acc.y*dt;
    vel.z = vel.z+acc.z*dt;
    printVector(vel,"vel");
}

void Particle::move(){
    //scale(dt, vel);
    vel.x = vel.x*dt;
    vel.y = vel.y*dt;
    vel.z = vel.z*dt;
    //printVector(vel,"vel");
    pos.x = pos.x+vel.x;
    pos.y = pos.y+vel.y;
    pos.z = pos.z+vel.z;
    printVector(pos, "pos");
}

