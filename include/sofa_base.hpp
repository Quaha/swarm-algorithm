#pragma once
#include <vector>
#include <stdexcept>

#include "hrect.hpp"

namespace swarm_algorithm {
    class sofa_base final {
    public:
        using function_ptr_type = double (*)(const std::vector<double>&);

        sofa_base(function_ptr_type func, const hrect& search_area) : func_(func), search_area_(search_area) {
            if (func_ == nullptr) {
                throw std::invalid_argument(__FUNCTION__ ": func was nullptr.");
            }
        }

        function_ptr_type func() const { return func_; }
        const hrect& search_area() const { return search_area_; }

        double result(size_t iter_count, bool from_start = true);

    private:
        function_ptr_type func_;
        hrect search_area_;

        void initialize();
        double make_step(size_t k);
    };
}