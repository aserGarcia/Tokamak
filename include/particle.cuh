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

#include <vector>
#include <cuda.h>

class Particle{
    private:
        float3 pos; //(x, y, z)
        float3 vel; //(x, y, z)
        int charge;
        float dt = 0.001;
        float mu_0 = 1e-7; //mu_0 = (4*PI)e-7 and biot savart divides bu_0 by 4*PI... (Newt/Amps^2)

    public:
        //Default Constructor
        Particle(float3, float3, int c = 1);
        ~Particle();

        //getter functions
        float3 getPosition();
        float3 getVelocity();
        void getB(std::vector<float3>, std::vector<int>obj_indx);

        void move();
};


