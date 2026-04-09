#pragma once
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <stdexcept>


namespace swarm_algorithm {

    template <size_t DIM> class hvector {
    public:
        using fp_t = double;

        hvector() {
            for (size_t i = 0; i < DIM; i++) {
                data_[i] = {};
            }
        }

        hvector(std::initializer_list<fp_t> data) {
            if (data.size() != DIM) {
                throw std::runtime_error("sizes doesn't match.");
            }

            size_t i = 0;
            for (fp_t v : data) {
                data_[i] = v;
            }
        }

        hvector(const hvector& other) {
            for (size_t i = 0; i < DIM; i++) {
                data_[i] = other.data_[i];
            }
        }

        size_t dims() const { return DIM; }
        const fp_t* data() const { return &data_[0]; }

        fp_t operator[](size_t index) const { return data_[index]; }

        fp_t& operator[](size_t index) { return data_[index]; }

        hvector& operator+=(const hvector& other) {
            for (size_t i = 0; i < DIM; i++) {
                data_[i] += other.data_[i];
            }
            return *this;
        }

        hvector& operator-=(const hvector& other) {
            for (size_t i = 0; i < DIM; i++) {
                data_[i] -= other.data_[i];
            }
            return *this;
        }

        hvector& operator*=(fp_t scalar) {
            for (size_t i = 0; i < DIM; i++) {
                data_[i] *= scalar;
            }
            return *this;
        }

        hvector& operator/=(fp_t scalar) {
            for (size_t i = 0; i < DIM; i++) {
                data_[i] /= scalar;
            }
            return *this;
        }

        fp_t length2() const {
            fp_t res = {};
            for (size_t i = 0; i < DIM; i++) {
                res += data_[i] * data_[i];
            }
            return res;
        }

        fp_t length() const { return std::sqrt(length2()); }

    private:
        double data_[DIM];
    };

    template <size_t DIM>
    hvector<DIM> operator+(const hvector<DIM>& a, const hvector<DIM>& b) {
        hvector<DIM> res(a);
        res += b;
        return res;
    }

    template <size_t DIM>
    hvector<DIM> operator-(const hvector<DIM>& a, const hvector<DIM>& b) {
        hvector<DIM> res(a);
        res -= b;
        return res;
    }

} // namespace swarm_algorithm