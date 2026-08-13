#pragma once

#include <iostream>
#define DIM 3

export class MortonKeys{
    public:
    uint32_t* keys;
    int* indices;
    MortonKeys(){
        
    }
};

friend minmax* GlobalBounding(std::unique_ptr<Prticles> bodies){
    
}