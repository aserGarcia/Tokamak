#pragma once
/*
|Author: Aser Garcia
|Date: Spring 2019   
|Course: Math Topics (CUDA)
|
|Class: Particle
|
|Description: The Class Particle creates an object with positions,
|             velocity, and force suing float4 and float3 respectively
|               
*/

class Particle{
    private:
        float4 pos; //(x, y, z, w)  w = mass
        float3 vel; //(x, y, z)
        float3 force; //(x, y, z)

    public:
        //Default Constructor
        Particle(float4, float3, float3);
        ~Particle();
        //setter functions
        __inline__ __device__ void setPosition(float4 p);
        __inline__ __device__ void setVelocity(float3 v);
        __inline__ __device__ void setForce(float3 f);

        //getter functions
        __inline__ __device__ float4 getPosition();
        __inline__ __device__ float3 getVelocity();
        __inline__ __device__ float3 getForce();

};
