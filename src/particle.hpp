#ifndef PARTICLE_HPP
#define PARTICLE_HPP
#include "vector.hpp"
#include <vector>
#define G 6.67e-11  //gravitational constant
#define K 8.99e-9 //Coulomb constant

/*Define the Particle class template
Each particle has a name, mass, an initial position, and an initial velocity
Position and velocity are then updated according to the forces applied on the particle, using Verlet's method*/

template <int DIM>
class Particle{
    private:
        std::string name;
        bool is_setup;
        double mass_;
        Vector<DIM> position_;
        Vector<DIM> old_position_;
        Vector<DIM> velocity_;
    public:

        Particle(){};

        Particle (std::string name, double mass, Vector<DIM> initial_position, Vector<DIM> initial_velocity):
            name(name),
            mass_(mass),
            position_(initial_position),
            velocity_(initial_velocity),
            is_setup(false)
        {};

        double mass(){
            return mass_;
        }
        //Update of position (and velocity) with Verlet integration
        void update(Vector<DIM> &acceleration, double delta_t){
            if (!is_setup){
                is_setup = true;
                old_position_=position_;
                position_+=velocity_*delta_t + 0.5*acceleration*delta_t*delta_t;
            }
            else{
                Vector<DIM> temp = position_;
                position_= 2*position_ - old_position_ +acceleration*delta_t*delta_t;
                old_position_=temp;
            }
            // Euler method update (simple but less accurate)
            // velocity_+= acceleration* delta_t;
            // position_+= velocity_ * delta_t;
        }

        //Compute gravitational acceleration exerted by another particle
        Vector<DIM> acceleration(Particle& other) const{
            Vector<DIM> direction = other.position_ - this->position_;
            double distance = direction.norm();
            Vector<DIM> a = ((G*other.mass_)/(distance*distance*distance)) * direction;
            return a;
        }

        //Print particle information
        inline friend std::ostream &operator<<(std::ostream& stream, Particle<DIM> particle){
            stream << particle.name
                << ": mass: " << particle.mass_
                << ", position: " << particle.position_
                << " velocity: " << particle.velocity_ << ";";
            return stream;
        };

        const Vector<DIM>& get_position() const{
            return position_;
        };
        
};

#endif
