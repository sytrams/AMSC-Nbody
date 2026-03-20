//solar system simulation
#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <map>
#include <string>
#include <cassert>
#include <functional>

#include "particle.hpp"
#include "vector.hpp"
#include "system.hpp"



//this function creates a 2D gravitation_system from file data
auto sys2(std::ifstream *file) {
	gravitation_system<2> sys;
	std::string name;
	double mass, pos_x, pos_y, vel_x, vel_y;

	while (*file >> name >> mass >> pos_x >> pos_y >> vel_x >> vel_y) {
		Vector<2> position = {pos_x, pos_y};
		Vector<2> velocity = {vel_x, vel_y};
		Particle<2> p(name, mass, position, velocity);
		sys.add_particle(p);
	}
	return sys;
}	

 //this function creates a 3D gravitation_system from file data
auto sys3(std::ifstream *file) {
	gravitation_system<3> sys;
	std::string name;
	double mass, pos_x, pos_y, pos_z, vel_x, vel_y, vel_z;

	while (*file >> name >> mass >> pos_x >> pos_y >> pos_z >> vel_x >> vel_y >> vel_z) {
		Vector<3> position = {pos_x, pos_y, pos_z};
		Vector<3> velocity = {vel_x, vel_y, vel_z};
		Particle<3> p(name, mass, position, velocity);
		sys.add_particle(p);
	}
	return sys;
}

class command_iterator{
	private:
		int argc;
		char** args;
		size_t i;

		std::ifstream *file = nullptr;
		unsigned int seed = 0;
		unsigned int number_particle = 0;
		unsigned int dim = 0;
		double delta_t = -INFINITY;
		double total_period = -INFINITY;
			
		bool has_next(){ return i < argc; }

		char* next(){
			assert (has_next());
			return args[i++];
		}

		void help(){
			std::cout <<
				"Usage:" << std::endl <<
				"	" << args[0] << " [OPTIONS]" << std::endl <<
				"Description:" << std::endl << 
				"	Program that finds the solution to an n-body problem by simulating the behavior of the particles" <<std::endl << 
				"Options:" << std::endl <<
				"-h --help	    Shows this message and exit." <<std::endl <<
				"-f	<file>      File where are stored the particles parameters." << std::endl <<
				"-s	<int>       Sets the seed to create a random simulation." << std::endl <<
				"-d	<int>       Sets the number of degree of freedom of the random simulation." << std::endl <<
				"-n	<int>       Sets the number of particle in the random simulation." << std::endl <<
				"-t	<double>    Sets the delta_t of the random simulation." << std::endl <<
				"-p	<file>      Sets the duration of the simulation." << std::endl;
				exit(0);
		}

		void set_file(){
			char *file_path = next();
			file = new std::ifstream(file_path);
			if (!file->is_open()) {
				std::cerr << "Cannot open " << file_path << std::endl;
				exit(1);
			}

			*file >> dim >> delta_t >> total_period;   //read simulation parameters from file
			std::cout << "DIM: " << dim
									<< ", delta_t: " << delta_t
									<< "s, total_period: " << total_period << "s\n" << std::endl;
		}

		void set_seed(){
			if (seed != 0){
				std::cerr << "Seed is setted twice" << std::endl;
				exit(1);
			}
			seed = atoi(next());
		}

		void set_number(){
			if (number_particle != 0){
				std::cerr << "The number of particles is setted twice" << std::endl;
				exit(1);
			}
			number_particle = atoi(next());
		}

		void set_dimention(){
			if (dim != 0){
				std::cerr << "The space dimention is setted twice" << std::endl;
				exit(1);
			}
			dim = atoi(next());
			if (dim != 2 && dim != 3){
				std::cerr << "The space dimention is not supported" << std::endl;
				exit(1);
			}
		}

		void set_delta_t(){
			if (delta_t > 0){
				std::cerr << "The delta_t value is setted twice" << std::endl;
				exit(1);
			}
			delta_t = atof(next());
		}

		void set_period(){
			if (total_period > 0){
				std::cerr << "The period value is setted twice" << std::endl;
				exit(1);
			}
			total_period = atof(next());
		}

		const std::map<std::string, std::function<void()>> commands {
			{"-h", [this](){this->help();}},
			{"--help", [this](){this->help();}},
			{"-f", [this](){this->set_file();}},
			{"-s", [this](){this->set_seed();}},
			{"-d", [this](){this->set_dimention();}},
			{"-n", [this](){this->set_number();}},
			{"-t", [this](){this->set_delta_t();}},
			{"-p", [this](){this->set_period();}}
		};
	
	public:
		command_iterator(int argc, char** args) : argc(argc), args(args), i(1){};

		void parse(){
			while(has_next())
				commands.at(next())();
		}

		unsigned int get_dim(){
			return dim;
		}

		template<unsigned int DIM>
		gravitation_system<DIM> generate_system(){
			gravitation_system<DIM> sys;
			if (file != nullptr){
				if constexpr(DIM == 2) {
					sys = sys2(file);
				}
				else if constexpr(DIM == 3) {
					sys = sys3(file);
				}

				if(dim != 2 && dim != 3)
					std::cerr << "Dimension not supported!" << std::endl;

				file->close();
				file = nullptr;
			} else {
				if (seed == 0){
					std::cerr << "Neither file neither seed is seted" << std::endl;
					exit(1);
				}
				srand(seed);
				Particle<DIM> p;
				Vector<DIM> pos;
				Vector<DIM> v;
				// std::cout << number_particle << std::endl;
				for (size_t i = 0; i < number_particle; ++i){
					if constexpr (DIM == 2){
						pos = Vector<2>(rand() - RAND_MAX/2, rand() - RAND_MAX/2);
						v = Vector<2>((rand() - RAND_MAX/2)*1.0e-5, (rand() - RAND_MAX/2)*1.0e-5);
						p = Particle<2>("random_particle", rand() * 1.0e24 , pos, v);
					} else if constexpr(DIM == 3){
						pos = Vector<3>(rand() - RAND_MAX/2, rand() - RAND_MAX/2, rand() - RAND_MAX/2);
						v = Vector<3>(rand() - RAND_MAX/2, rand() - RAND_MAX/2, rand() - RAND_MAX/2);
						p = Particle<3>("random_particle", rand(), pos, v);
					}
					// std::cout << p << std::endl;
					sys.add_particle(p);
				}
			}
			return sys;
		 }

		double get_delta_t(){
			return delta_t;
		}

		double get_period(){
			return total_period;
		}
};

//gravitation_system needs DIM in compile time, use template to achieve this
template<int N> //N is the space dimension
void simulate(gravitation_system<N>& sys, double delta_t, double total_period) {

    std::cout << sys << std::endl; //print initial state of the system

    std::vector<std::vector<double>> position_matrix;

		position_matrix.push_back(sys.get_positions());
    for(int i = 0; i * delta_t < total_period; ++i){
        sys.update(delta_t); //update system by delta_t time step
		if (i % 20 == 0 || (i+1)*delta_t >= total_period) //store position every 10 steps and the last step to avoid huge files
			position_matrix.push_back(sys.get_positions());
    }

    std::ofstream output("positions.txt");
    for (auto &row: position_matrix){ //write positions to file positions.txt
        if (!row.empty()) {
            output << row[0];
            for (int j = 1; j < row.size(); j++)
                output << ", " << row[j];
        }
        output << "\n";
    }
	std::cout << sys << std::endl; //print final state of the system
}

int main(int argc, char **args){
	command_iterator arg(argc, args);
	arg.parse();
	

	unsigned int dim = arg.get_dim();
	if (dim == 2){
		gravitation_system<2> sys = arg.generate_system<2>();
		simulate<2>(sys, arg.get_delta_t(), arg.get_period());
	} else if(dim == 3){
		gravitation_system<3> sys = arg.generate_system<3>();
		simulate<3>(sys, arg.get_delta_t(), arg.get_period());
	}

	return 0;
}

