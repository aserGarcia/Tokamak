#include <iostream>
#include "particle.cuh"

/*-------------------------------
|    Constructors/Destructors   |
-------------------------------*/
Particle::Particle(float4 p, float3 v, float3 f){pos = p; vel = v; force=f;}
Particle::~Particle(){}

/*-----------------------
|    Setter Functions   |
-----------------------*/
__device__ void Particle::setPosition(float4 p) {pos = p;}

__device__ void Particle::setVelocity(float3 v) {vel =v;}

__device__ void Particle::setForce(float3 f) {force = f;}


/*-----------------------
|    Getter Functions   |
-----------------------*/
__device__ float4 Particle::getPosition(){ return pos; }

__device__ float3 Particle::getVelocity(){ return vel; }

__device__ float3 Particle::getForce(){ return force; }
