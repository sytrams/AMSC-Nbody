module;

#include <string>
#include <iostream>
#include <array>
#include <cstring>
#include <utility>
#include <fstream>
#include <memory>

export module particle;

export inline constexpr double G = 6.67e-11;  // gravitational constant
export inline constexpr double K = 8.99e-9;   // Coulomb constant

export class Particle{
    public:
        uint64_t num_particles;
        std::unique_ptr<double[]> x, y, z;
        std::unique_ptr<double[]> vx, vy, vz;
        std::unique_ptr<double[]> mass;
    explicit Particle(std::ifstream& inFile)
    {
        if (!inFile.is_open() || !inFile.good()){
            throw std::runtime_error("Invalid or unreadable file stream provided.");
        }

        inFile.read(reinterpret_cast<char*>(&num_particles), sizeof(uint64_t));
        if (num_particles == 0){
            throw std::runtime_error("Particle count is zero.");
        }
        x = std::make_unique<double[]>(num_particles);
        y = std::make_unique<double[]>(num_particles);
        z = std::make_unique<double[]>(num_particles);
        vx = std::make_unique<double[]>(num_particles);
        vy = std::make_unique<double[]>(num_particles);
        vz = std::make_unique<double[]>(num_particles);
        mass = std::make_unique<double[]>(num_particles);
        
        size_t bytes_to_read = num_particles * sizeof(double);
        
        inFile.read(reinterpret_cast<char*>(x.get()), bytes_to_read);
        inFile.read(reinterpret_cast<char*>(y.get()), bytes_to_read);
        inFile.read(reinterpret_cast<char*>(z.get()), bytes_to_read);
        inFile.read(reinterpret_cast<char*>(vx.get()), bytes_to_read);
        inFile.read(reinterpret_cast<char*>(vy.get()), bytes_to_read);
        inFile.read(reinterpret_cast<char*>(vz.get()), bytes_to_read);
        inFile.read(reinterpret_cast<char*>(mass.get()), bytes_to_read);

        // 5. Final validation to ensure the file didn't end prematurely
        if (!inFile) {
            throw std::runtime_error("Failed to read all expected particle data. File may be truncated.");
        }
    }
};
