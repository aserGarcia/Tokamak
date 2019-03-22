#include <iostream>
#include "particle.cuh"
#include "vec_helpers.cuh"


/*-------------------------------
|    Constructors/Destructors   |
-------------------------------*/
Particle::Particle(float3 p, float3 v, int c){pos = p; vel = v; charge = c;}
Particle::~Particle(){}

/*-----------------------
|    Getter Functions   |
-----------------------*/
float3 Particle::getPosition(){ return pos; }

float3 Particle::getVelocity(){ return vel; }

void Particle::getB(std::vector<float3> vertices){
    float3 B, crossv, dl; //B (magnetic field) force

    float distance;
    float BSLscalar = mu_0;
    int size_vect = vertices.size();
    for (int i = 1; i<size_vect; i++){

        std::cout<<"in second loop";
        float3 now_pt = vertices[i%size_vect];
        float3 prev_pt = vertices[i-1%size_vect];
        //calculate distance
        float3 r = makeVector(prev_pt, pos);
        distance = length(r);
        //distance is 3m

        dl = makeVector(prev_pt, now_pt);
        float3 unit_r = unit(r);
        crossv = cross(dl, unit_r);
        std::cout<<crossv.x<<std::endl;
        BSLscalar /= distance*distance;
        B.x += BSLscalar*crossv.x;
        B.y += BSLscalar*crossv.y;        
        B.z += BSLscalar*crossv.z;

        BSLscalar *= distance*distance; //undoes the division
        }


    float3 old_vel = cross(vel, B);    
    vel.x = charge*old_vel.x*dt;
    vel.y = charge*old_vel.y*dt;
    vel.z = charge*old_vel.z*dt;
}

void Particle::move(){
    pos.x = vel.x*dt;
    pos.y = vel.y*dt;
    pos.z = vel.z*dt;
}

