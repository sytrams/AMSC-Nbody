#ifndef VECTOR_HPP
#define VECTOR_HPP

#include <array>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <cstddef>
#include <cassert>

//template class VectorBase that provides storage and basic operations for vectors of arbitrary dimension
template <std::size_t DIM>
class VectorBase {
protected:
    std::array<double, DIM> components_;

public:
    // constructors
    VectorBase() : components_{} {}
    explicit VectorBase(const std::array<double, DIM>& tmp) : components_{tmp} {}
    VectorBase(const VectorBase& other) = default;
    VectorBase(VectorBase&& other) noexcept = default;
    ~VectorBase() = default;

    // helpers
    double norm() const {
        double square_sum = 0.0;
        for (std::size_t i = 0; i < DIM; ++i)
            square_sum += components_[i] * components_[i];
        return std::sqrt(square_sum);
    }

    // fill all components with a value
    void fill(double v) { components_.fill(v); }

    // element access for derived classes and free functions
    double& at(std::size_t i) { return components_[i]; }
    const double& at(std::size_t i) const { return components_[i]; }


    //operators
    VectorBase &operator=(const nullptr_t){ // set all components to zero
        for (auto &component : components_)
            component = 0;
        return *this;
    };

    VectorBase &operator=(const VectorBase &other){ // copy all values to this
        for (int i = 0; i < DIM; ++i)
            components_[i] = other.components_[i];
        return *this;
    };

    VectorBase &operator=(const VectorBase &&other){ // move all values to this
        for (int i = 0; i < DIM; ++i)
            components_[i] = std::move(other.components_[i]);
        return *this;
    }

    // access
    double& operator[](std::size_t i) { return this->components_[i]; }
    const double& operator[](std::size_t i) const { return this->components_[i]; }

    // arithmetic assignments
    VectorBase& operator+=(const VectorBase& other) { // sum component-wise
        for (std::size_t i = 0; i < DIM; ++i) this->components_[i] += other.components_[i];
        return *this;
    }
    VectorBase& operator-=(const VectorBase& other) { //subtract component-wise
        for (std::size_t i = 0; i < DIM; ++i) this->components_[i] -= other.components_[i];
        return *this;
    }
    VectorBase& operator*=(double a) { // multiply all components by a scalar
        for (std::size_t i = 0; i < DIM; ++i) this->components_[i] *= a;
        return *this;
    }
    VectorBase& operator/=(double a) { // divide all components by a scalar
        for (std::size_t i = 0; i < DIM; ++i) this->components_[i] /= a;
        return *this;
    }

    inline friend std::ostream &operator<< (std::ostream &stream, VectorBase vector){ // print vector to output stream
        stream << "{";

        if (DIM > 0)
            stream << vector.components_[0];

        for (size_t i = 1; i < DIM; ++i)
            stream << ", " << vector.components_[i];
        
        stream << "}";
        return stream;
    }
};

// Primary template: Vector<DIM>
template <std::size_t DIM>
class Vector : public VectorBase<DIM> {
public:
    using VectorBase<DIM>::VectorBase; // inherit base constructors
    using VectorBase<DIM>::operator=;

    // defaulted special members (good enough for this example)
    Vector() = default;
    Vector(const Vector&) = default;
    Vector(Vector&&) noexcept = default;
    Vector& operator=(const Vector&) = default;
    Vector& operator=(Vector&&) noexcept = default;
    ~Vector() = default;
};

// Specialization for 2D with convenient constructors and accessors
template <>
class Vector<2> : public VectorBase<2> {
public:
    using VectorBase<2>::VectorBase;
    using VectorBase<2>::operator=;

    Vector() = default;
    Vector(double x, double y) : VectorBase(std::array<double,2>{x, y}) {}

    Vector(const Vector&) = default;
    Vector(Vector&&) noexcept = default;
    Vector& operator=(const Vector&) = default;
    Vector& operator=(Vector&&) noexcept = default;
    ~Vector() = default;

    double& x() { return this->components_[0]; }
    double& y() { return this->components_[1]; }
    const double& x() const { return this->components_[0]; }
    const double& y() const { return this->components_[1]; }
};

// Specialization for 3D (simple)
template <>
class Vector<3> : public VectorBase<3> {
public:
    using VectorBase<3>::VectorBase;
    using VectorBase<3>::operator=;

    Vector() = default;
    Vector(double x, double y, double z) : VectorBase(std::array<double,3>{x, y, z}) {}

    Vector(const Vector&) = default;
    Vector(Vector&&) noexcept = default;
    Vector& operator=(const Vector&) = default;
    Vector& operator=(Vector&&) noexcept = default;
    ~Vector() = default;

    double& x() { return this->components_[0]; }
    double& y() { return this->components_[1]; }
    double& z() { return this->components_[2]; }
    const double& x() const { return this->components_[0]; }
    const double& y() const { return this->components_[1]; }
    const double& z() const { return this->components_[2]; }
};

// free operators (generic)

template <std::size_t DIM>
inline Vector<DIM> operator+(Vector<DIM> a, const Vector<DIM>& b) { a += b; return a; } // addition
template <std::size_t DIM>
inline Vector<DIM> operator-(Vector<DIM> a, const Vector<DIM>& b) { a -= b; return a; } // subtraction
template <std::size_t DIM>
inline Vector<DIM> operator*(Vector<DIM> a, double s) { a *= s; return a; } // scalar multiplication
template <std::size_t DIM>
inline Vector<DIM> operator*(double s, Vector<DIM> a) { a *= s; return a; } // scalar multiplication
template <std::size_t DIM>
inline Vector<DIM> operator/(Vector<DIM> a, double s) { a /= s; return a; } // scalar division

extern template class Vector<2>;
extern template class Vector<3>;
#endif 
