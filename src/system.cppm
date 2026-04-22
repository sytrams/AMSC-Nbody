module;

#include <vector>
#include <iostream>
#include <cstddef>
#include <omp.h>

export module system;

import particle;
import vector;

//Define the gravitation_system class template, which manages a collection of particles (i.e. celestial bodies)
export template<int DIM>
class gravitation_system{
	private:
		std::vector<Particle<DIM>> particles;

	public:

		gravitation_system() = default;
		gravitation_system(const gravitation_system<DIM>& other) = default;
		gravitation_system(gravitation_system<DIM>&& other) noexcept = default;

		//Add a particle to the system
		void add_particle(const Particle<DIM> &particle){
			this->particles.push_back(particle);
		}

		//Update all particles' positions based on gravitational interactions
		void update(double delta_t){
			std::vector<Vector<DIM>> accelerations(this->particles.size());

			#pragma omp parallel for
			for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(this->particles.size()); ++i){
				const auto &particle_a = this->particles[static_cast<std::size_t>(i)];
				Vector<DIM> acceleration{};

				for (std::size_t j = 0; j < this->particles.size(); ++j) {
					if (static_cast<std::size_t>(i) == j)
						continue;

					acceleration += particle_a.acceleration(this->particles[j]);
				}

				accelerations[static_cast<std::size_t>(i)] = acceleration;
			}

			#pragma omp parallel for
			for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(this->particles.size()); ++i)
				this->particles[static_cast<std::size_t>(i)].update(
					accelerations[static_cast<std::size_t>(i)],
					delta_t
				);
		}

		//Retrieve the positions of all particles in a flat vector
		std::vector<double> get_positions() const{
			std::vector<double> positions;
			positions.reserve(particles.size() * static_cast<std::size_t>(DIM));
			for(const auto &particle : particles){
				const auto &position_particle = particle.get_position();
				for (int i = 0; i < DIM; ++i)
					positions.push_back(position_particle[i]);
			}
			return positions;
		}

		inline gravitation_system<DIM>& operator= (const gravitation_system<DIM> &other) = default;
		inline gravitation_system<DIM>& operator= (gravitation_system<DIM> &&other) noexcept = default;

		//Print system information
		inline friend std::ostream &operator<<(std::ostream &stream, const gravitation_system<DIM> &system){
			for (const auto &particle : system.particles)
				stream << particle << std::endl;

			return stream;
		}
};
