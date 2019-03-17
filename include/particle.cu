#include <iostream>
#include "particle.cuh"
#include "vec_helpers.hpp"


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

void Particle::getB(std::vector<std::vector<float3> > & vertices){
    float3 B, cross, dl; //B (magnetic field) force
    float tol = 0.001f;
    float distance;
    float BSLscalar = mu_0;
    //iterate through vectors //start one vector ahead
    std::vector<float3>::iterator prev_pt;
    for (auto shape: vertices){
        prev_pt = shape.begin();
        for(std::vector<float3>::iterator now_pt = shape.begin()+1; now_pt!= shape.end(); ++now_pt){
            //calculate distance
            float3 r = makeVector(*prev_pt, pos);
            distance = length(r);
            //distance is 3m
            if(abs(distance-3.0f)<tol){
                dl = makeVector(*prev_pt, *now_pt);
                float3 unit_r = normalize(r);
                cross = vectCross(dl, unit_r);

                BSLscalar /= distance*distance;
                B.x += BSLscalar*cross.x;
                B.y += BSLscalar*cross.y;        
                B.z += BSLscalar*cross.z;

                BSLscalar *= distance*distance; //undoes the division
            }
            *prev_pt = *now_pt; //move along the wire section
        }
    }


    float3 old_vel = vectCross(vel, B);    
    vel.x = charge*old_vel.x*dt;
    vel.y = charge*old_vel.y*dt;
    vel.z = charge*old_vel.z*dt;
}

void Particle::move(){
    pos.x = vel.x*dt;
    pos.y = vel.y*dt;
    pos.z = vel.z*dt;
}

