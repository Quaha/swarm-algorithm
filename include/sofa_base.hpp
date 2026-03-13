#pragma once
#include <vector>
#include <stdexcept>
#include <random>

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

        std::pair<std::vector<double>, double> result(size_t iter_count, bool from_start = true);

    private:
        function_ptr_type func_;
        hrect search_area_;
        std::mt19937_64 gen_;
        size_t start_population_size_ = 100;
        double gamma_ = 0.001;

        double ans_ = 0.0;
        std::vector<double> ans_point_;

        std::vector<std::vector<double>> points_;
        std::vector<double> values_;

        void initialize();
        double make_step();

        std::vector<double> gen_point_uniform();
    };
}