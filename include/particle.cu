#include <iostream>
#include "particle.cuh"
#include "vec_helpers.cuh"

Particle::Particle(float3 p, float3 v, int c){pos = p; vel = v; charge = c;}
Particle::~Particle(){}

float3 Particle::getPosition(){ return pos; }
float3 Particle::getVelocity(){ return vel; }

//magnetic force calculation
void Particle::getB(std::vector<float3> vertices, float3 SHAPE_SEPARATOR){
    float3 B, crossv, dl; //B (magnetic field) force

    float r;
    float mu_0;
    for (int i = 1; i<vertices.size(); i++){
        if(vertices[i].x != SHAPE_SEPARATOR.x){
            float3 now_pt = vertices[i];
            float3 prev_pt = vertices[i-1];
            //calculate distance
            float3 rv = makeVector(prev_pt, pos);
            r = length(rv);
            //distance is 3m
            dl = makeVector(prev_pt, now_pt);
            float3 unit_r = unit(rv);
            crossv = cross(dl, unit_r);

            mu_0 /= r*r;
        
            scale(mu_0, crossv);
            B.x += crossv.x;
            B.y += crossv.y;        
            B.z += crossv.z;

            mu_0 *= r*r; //undoes the division
        }
    }

    float3 old_vel = cross(vel, B);
    scale(charge*dt, old_vel);    
    vel.x = old_vel.x;
    vel.y = old_vel.y;
    vel.z = old_vel.z;
}

void Particle::move(){
    pos.x = vel.x*dt;
    pos.y = vel.y*dt;
    pos.z = vel.z*dt;
}

