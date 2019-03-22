#include <cuda.h>

float3 makeVector(float3 point1, float3 point2){
    return make_float3(point2.x-point1.x, point2.y-point1.y, point2.z-point1.z);
}

void scale(float s, float3 &v){
    v.x *= s;
    v.y *= s;
    v.z *= s;
}

float length(float3 v){
    return sqrtf(v.x*v.x+v.y*v.y+v.z*v.z);
}

float3 unit(float3 v){
    float magV = length(v);
    return make_float3(v.x/magV, v.y/magV, v.z/magV);
}

float3 cross(float3 v1, float3 v2){
    return make_float3(v1.y*v2.z-v1.z*v2.y, v1.z*v2.x-v1.x*v2.z, v1.x*v2.y-v1.y*v2.x);
}