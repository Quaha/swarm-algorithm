#pragma once
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>


#include "hrect.hpp"
#include "hvector.hpp"
#include "rand_precalc.hpp"
#include "truncated_normal.hpp"
#include "xoshiro.hpp"


namespace swarm_algorithm {

    template <size_t DIM> class sofa_base final {
    public:
        using function_ptr_type = double (*)(const hvector<DIM>&);

        sofa_base(function_ptr_type func, const hrect& search_area, uint64_t seed)
            : seed_(seed), func_(func), search_area_(search_area),
            normal_distr_(search_area, gen_, DIM), bad_points_(0) {
            if (func_ == nullptr) {
                throw std::invalid_argument("func was nullptr.");
            }

            if (search_area_.dimensions_cnt() != DIM) {
                throw std::runtime_error("invalid search area.");
            }
        }

        function_ptr_type func() const { return func_; }
        const hrect& search_area() const { return search_area_; }
        size_t bad_points() const { return bad_points_; }

        std::pair<hvector<DIM>, double> result(size_t iter_count,
            bool from_start = true) {
            if (from_start) {
                auto initialize_begin = std::chrono::steady_clock::now();
                initialize();
                auto initialize_end = std::chrono::steady_clock::now();
                std::chrono::duration<double> initialize_seconds{ initialize_end -
                                                                 initialize_begin };
                std::cout << "Initialize: " << initialize_seconds << "s\n";
            }

            auto steps_begin = std::chrono::steady_clock::now();

            for (size_t iter = 0; iter < iter_count; iter++) {
                double res = make_step();

                if (res > ans_) {
                    ans_ = values_.back();
                    ans_point_ = points_.back();
                    best_upds_.push_back(values_.size() - 1);
                }
            }

            auto steps_end = std::chrono::steady_clock::now();
            std::chrono::duration<double> steps_seconds{ steps_end - steps_begin };
            std::cout << "Steps: " << steps_seconds << "s\n";

            return { ans_point_, ans_ };
        }

        std::vector<std::tuple<size_t, hvector<DIM>, double>> dump_bests() {
            std::vector<std::tuple<size_t, hvector<DIM>, double>> res;

            for (auto v : best_upds_) {
                res.push_back({ v, points_[v], values_[v] });
            }

            return res;
        }

        void reserve_buffers(size_t size) {
            f_psi_.resize(std::max(f_psi_.size(), size));
            rho_.resize(std::max(rho_.size(), size));
            prob_.resize(std::max(prob_.size(), size));
        }

    private:
        uint64_t seed_;
            
        function_ptr_type func_;
        hrect search_area_;
        rand_precalc<Xoshiro::Xoshiro256PP> gen_;
        size_t start_population_size_ = 100;
        double gamma_ = 0.01;

        double ans_ = 0.0;
        hvector<DIM> ans_point_;

        truncated_normal<decltype(gen_)> normal_distr_;
        std::vector<hvector<DIM>> points_;
        std::vector<double> logs_;
        std::vector<double> values_;
        std::vector<size_t> best_upds_;

        size_t bad_points_;

        void initialize() {
            gen_.seed(seed_);
            gen_.set_batch_size(40000);
            ans_ = 0.0;
            bad_points_ = 0;

            points_.clear();
            logs_.clear();
            values_.clear();
            best_upds_.clear();

            for (size_t i = 0; i < start_population_size_; i++) {
                points_.emplace_back(gen_point_uniform());
                values_.push_back(func_(points_.back()));
                logs_.push_back(std::log(values_.back()));

                if (values_.back() > ans_) {
                    ans_ = values_.back();
                    ans_point_ = points_.back();
                    best_upds_.push_back(values_.size() - 1);
                }
            }
        }

        std::vector<double> f_psi_;
        std::vector<double> rho_;
        std::vector<double> prob_;

        double make_step() {

            const auto psi_ = [](size_t k) -> double {
                return std::sqrt((static_cast<double>(k) * 0.001 + 1.0));
                };

            const size_t k = points_.size() + 1;
            reserve_buffers(k - 1);

            // calc f_psi
            {
                double psi_k = psi_(k);
                for (size_t i = 0; i < k - 1; i++) {
                    f_psi_[i] = std::exp(logs_[i] * psi_k);
                }
            }

            // calc rho
            {
                double sum = 0.0;
                for (size_t i = 0; i < k - 1; i++) {
                    rho_[i] = f_psi_[i];
                    sum += f_psi_[i];
                }

                double inv_sum = 1.0 / sum;
                for (size_t i = 0; i < k - 1; i++) {
                    rho_[i] *= inv_sum;
                }
            }

            // pivot selection
            {
                double sum = 0.0;
                for (size_t i = 0; i < k - 1; i++) {
                    double psi = rho_[i] < gamma_ ? 0.0 : f_psi_[i];

                    prob_[i] = psi;
                    sum += psi;
                }

                double inv_sum = 1.0 / sum;
                for (size_t i = 0; i < k - 1; i++) {
                    prob_[i] *= inv_sum;
                }
            }

            // choose pivot
            size_t pivot_index = 0;
            {
                std::uniform_real_distribution pivot_distr(0.0, 1.0);
                double pivot_val = pivot_distr(gen_);
                double sum = 0.0;
                for (size_t i = 0; i < k - 1; i++) {
                    sum += prob_[i];
                    if (sum >= pivot_val) {
                        pivot_index = i;
                        break;
                    }
                }
            }

            // gen new point
            {
                double stddev = search_area_.max_dim() * sqrt(2.0 * search_area_.dimensions_cnt() / log(k));

                hvector<DIM> new_point;
                for (size_t i = 0; i < DIM; i++) {

                    double x = normal_distr_.generate(points_[pivot_index][i], stddev, i);

                    new_point[i] = x;
                }

                points_.emplace_back(new_point);
                values_.push_back(func_(points_.back()));
                logs_.push_back(std::log(values_.back()));
            }

            return values_.back();
        }

        hvector<DIM> gen_point_uniform() {
            hvector<DIM> res;
            for (size_t i = 0; i < DIM; i++) {
                const auto [l, r] = search_area_.get(i);
                std::uniform_real_distribution distr(l, r);
                res[i] = distr(gen_);
            }

            return res;
        }
    };
} // namespace swarm_algorithm