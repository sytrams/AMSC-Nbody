module;

export module core;

template <typename Backend> 
struct GnuOps{
export void createMortonKeys(Backend backend);

void loadParticleOnGPU(Backend backend);

void normalise(Backend backend);

void writeKeys(Backend backend);

void writeIndices(Backend backend);
};