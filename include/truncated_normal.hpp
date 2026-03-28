#pragma once

#include "hrect.hpp"
#include "xoshiro.hpp"
#include <cmath>
#include <random>
#include <vector>


namespace swarm_algorithm {
    template <typename TGen> class truncated_normal {
    public:
        truncated_normal(const hrect& search_area, TGen& gen, size_t dim)
            : search_area_(search_area), gen_(gen), dim_(dim) {

            build_standard_table();

            cached_bounds_.resize(dim);
            for (size_t i = 0; i < dim; ++i) {
                auto [l, r] = search_area_.get(i);
                cached_bounds_[i] = { l, r };
            }
        }

        // Генерация одной координаты
        double generate(double mean, double stddev, size_t coord) {
            auto [lower, upper] = cached_bounds_[coord];

            double z_lower = (lower - mean) / stddev;
            double z_upper = (upper - mean) / stddev;

            if (z_lower <= -6.0 && z_upper >= 6.0) {
                std::normal_distribution<double> normal(mean, stddev);
                return normal(gen_);
            }

            double cdf_lower = get_cdf_fast(z_lower);
            double cdf_upper = get_cdf_fast(z_upper);

            std::uniform_real_distribution<double> uniform(cdf_lower, cdf_upper);
            double u = uniform(gen_);

            double z = get_z_from_cdf_fast(u);

            return mean + stddev * z;
        }

    private:
        void build_standard_table(double z_min = -8.0, double z_max = 8.0,
            size_t n = 10000) {
            z_table_.resize(n);
            cdf_table_.resize(n);

            double h = (z_max - z_min) / (n - 1);
            double total = 0.0;

            for (size_t i = 0; i < n; ++i) {
                z_table_[i] = z_min + i * h;
                double pdf = normal_pdf(z_table_[i]);

                if (i == 0) {
                    cdf_table_[0] = 0.0;
                }
                else {
                    total += (normal_pdf(z_table_[i - 1]) + pdf) * h * 0.5;
                    cdf_table_[i] = total;
                }
            }

            double inv_total = 1.0 / total;
            for (auto& c : cdf_table_)
                c *= inv_total;

            last_z_idx_ = 0;
            last_cdf_idx_ = 0;
        }

        double get_cdf_fast(double z) {
            if (z <= z_table_[0])
                return 0.0;
            if (z >= z_table_.back())
                return 1.0;

            size_t left = 0, right = z_table_.size() - 1;

            if (z >= z_table_[last_z_idx_]) {
                left = last_z_idx_;
            }
            else {
                right = last_z_idx_;
            }

            while (left < right) {
                size_t mid = (left + right) >> 1;
                if (z_table_[mid] < z) {
                    left = mid + 1;
                }
                else {
                    right = mid;
                }
            }

            size_t idx = left;
            last_z_idx_ = idx;

            if (idx == 0)
                return 0.0;

            double t = (z - z_table_[idx - 1]) / (z_table_[idx] - z_table_[idx - 1]);
            return cdf_table_[idx - 1] + t * (cdf_table_[idx] - cdf_table_[idx - 1]);
        }

        double get_z_from_cdf_fast(double cdf) {
            if (cdf <= cdf_table_[0])
                return z_table_[0];
            if (cdf >= cdf_table_.back())
                return z_table_.back();

            size_t left = 0, right = cdf_table_.size() - 1;

            if (cdf >= cdf_table_[last_cdf_idx_]) {
                left = last_cdf_idx_;
            }
            else {
                right = last_cdf_idx_;
            }

            while (left < right) {
                size_t mid = (left + right) >> 1;
                if (cdf_table_[mid] < cdf) {
                    left = mid + 1;
                }
                else {
                    right = mid;
                }
            }

            size_t idx = left;
            last_cdf_idx_ = idx;

            if (idx == 0)
                return z_table_[0];

            double t =
                (cdf - cdf_table_[idx - 1]) / (cdf_table_[idx] - cdf_table_[idx - 1]);
            return z_table_[idx - 1] + t * (z_table_[idx] - z_table_[idx - 1]);
        }

        static double normal_pdf(double x) {
            constexpr double DPI = 6.283185307179586476925286766559;
            return std::exp(-0.5 * x * x) / std::sqrt(DPI);
        }

        const hrect& search_area_;
        TGen& gen_;
        size_t dim_;
        std::vector<double> z_table_;
        std::vector<double> cdf_table_;
        std::vector<std::pair<double, double>> cached_bounds_;
        size_t last_z_idx_;
        size_t last_cdf_idx_;
    };
} // namespace swarm_algorithm