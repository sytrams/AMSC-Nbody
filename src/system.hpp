#include "particle.hpp"
#include "vector.hpp"
#include <vector>
#include <omp.h>

//Define the gravitation_system class template, which manages a collection of particles (i.e. celestial bodies)
template<int DIM>
class gravitation_system{
	private:
		std::vector<Particle<DIM>> particles;

	public:

		gravitation_system(const gravitation_system<DIM>& other){
			particles = other.particles;
		}

		gravitation_system(){}

		//Add a particle to the system
		void add_particle(const Particle<DIM> &particle){
			this->particles.push_back(Particle<DIM>(particle));
		};

		//Update all particles' positions based on gravitational interactions
		void update(double delta_t){
			Vector<DIM> acceleration;
			#pragma omp parallel for private(acceleration)
			for (int i = 0; i < this->particles.size(); ++i){
				auto &particle_a = this->particles[i];
				acceleration = 0;

				for (auto &particle_b : this->particles)
					//assume that if the the particle a and b have the same address, they are the same
					if (&particle_a != &particle_b){
						auto contribute = particle_a.acceleration(particle_b);
						acceleration += contribute;
					}
				
				particle_a.update(acceleration, delta_t);
			}
		};

		//Retrieve the positions of all particles in a flat vector
		std::vector<double> get_positions() const{
			std::vector<double> positions;
			for(auto &particle : particles){
				auto &position_particle = particle.get_position();
				for (int i = 0; i < DIM; ++i)
					positions.push_back(position_particle[i]);
			}
			return positions;
		}

		inline gravitation_system<DIM>& operator= (gravitation_system<DIM> &other){
			particles = other.particles;
			return *this;
		}

		inline gravitation_system<DIM>& operator= (gravitation_system<DIM> &&other){
			particles.swap(other.particles);
			return *this;
		}

		//Print system information
		inline friend std::ostream &operator<<(std::ostream &stream, gravitation_system<DIM> system){
			for (auto particle : system.particles)
				stream << particle << std::endl;

			return stream;
		}
};

