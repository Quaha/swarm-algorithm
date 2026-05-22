#pragma once
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include "hrect.hpp"
#include "hvector.hpp"
#include "rand_precalc.hpp"
#include "xoshiro.hpp"


namespace swarm_algorithm {

    template <size_t DIM> class mingo_base final {
    public:
        using function_ptr_type = double (*)(const hvector<DIM>&);

        mingo_base(function_ptr_type func, const hrect& search_area, uint64_t seed)
            : seed_(seed), func_(func), search_area_(search_area),
            step_(0) {
            if (func_ == nullptr) {
                throw std::invalid_argument("func was nullptr.");
            }

            if (search_area_.dimensions_cnt() != DIM) {
                throw std::runtime_error("invalid search area.");
            }
        }

        function_ptr_type func() const { return func_; }
        const hrect& search_area() const { return search_area_; }

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

            for (size_t iter = 0; iter < iter_count; ++iter) {
                if (phase_ == 0) {
                    phase_1_step(current_i_);
                }
                else {
                    phase_2_step(current_i_);
                }

                ++current_i_;
                if (current_i_ >= points_.size()) {
                    current_i_ = 0;
                    phase_ = 1 - phase_;
                    if (phase_ == 0) {
                        recompute_best();
                    }
                }
            }

            auto steps_end = std::chrono::steady_clock::now();
            std::chrono::duration<double> steps_seconds{ steps_end - steps_begin };
            std::cout << "Steps: " << steps_seconds << "s\n";

            return { ans_point_, ans_ };
        }

        std::vector<std::tuple<size_t, hvector<DIM>, double>> dump_bests() const {
            return best_upds_;
        }

        void set_population_size(size_t sz) {
            if (sz < 2) {
                throw std::invalid_argument("population size must be >= 2.");
            }
            population_size_ = sz;
        }

    private:
        uint64_t seed_;

        function_ptr_type func_;
        hrect search_area_;
        rand_precalc<Xoshiro::Xoshiro256PP> gen_;

        // Paper uses NP = 50 across all CEC-2017/2022 experiments.
        size_t population_size_ = 50;

        double ans_ = -std::numeric_limits<double>::infinity();
        hvector<DIM> ans_point_;

        size_t step_;
        size_t current_i_ = 0;
        int phase_ = 0;  // 0 = exploration, 1 = exploitation
        size_t best_idx_ = 0;

        std::vector<hvector<DIM>> points_;
        std::vector<double> values_;
        std::vector<std::tuple<size_t, hvector<DIM>, double>> best_upds_;

        void initialize() {
            gen_.seed(seed_);
            gen_.set_batch_size(40000);
            step_ = 0;
            ans_ = -std::numeric_limits<double>::infinity();
            current_i_ = 0;
            phase_ = 0;

            points_.clear();
            values_.clear();
            best_upds_.clear();
            points_.reserve(population_size_);
            values_.reserve(population_size_);

            for (size_t i = 0; i < population_size_; ++i) {
                points_.emplace_back(gen_point_uniform());
                values_.push_back(func_(points_.back()));
                ++step_;

                if (values_.back() > ans_) {
                    ans_ = values_.back();
                    ans_point_ = points_.back();
                    best_upds_.emplace_back(step_, ans_point_, ans_);
                }
            }

            recompute_best();
        }

        void recompute_best() {
            best_idx_ = 0;
            for (size_t i = 1; i < values_.size(); ++i) {
                if (values_[i] > values_[best_idx_]) best_idx_ = i;
            }
        }

        void phase_1_step(size_t i) {
            const size_t N = points_.size();
            std::uniform_int_distribution<size_t> idx_distr(0, N - 1);
            size_t p_idx;
            do { p_idx = idx_distr(gen_); } while (p_idx == i);

            const auto& x_i    = points_[i];
            const auto& prey   = points_[p_idx];
            const double f_i   = values_[i];
            const double f_p   = values_[p_idx];

            hvector<DIM> candidate;

            if (f_p > f_i) {
                const auto& x_best = points_[best_idx_];
                std::uniform_int_distribution<int> i_distr(1, 2);
                const double I = static_cast<double>(i_distr(gen_));
                for (size_t k = 0; k < DIM; ++k) {
                    double rl = 0.5 * levy_sample();
                    candidate[k] = x_best[k] + rl * (x_best[k] - I * x_i[k]);
                }
            }
            else {
                std::normal_distribution<double> nrm(0.0, 1.0);
                for (size_t k = 0; k < DIM; ++k) {
                    double rb = nrm(gen_);
                    candidate[k] = x_i[k] + rb * (x_i[k] - prey[k]);
                }
            }

            repair_bounds(candidate);

            double fval = func_(candidate);
            ++step_;

            greedy_select(i, candidate, fval);
        }

        void phase_2_step(size_t i) {
            const size_t N = points_.size();
            std::uniform_real_distribution<double> u01(0.0, 1.0);
            std::normal_distribution<double> nrm(0.0, 1.0);

            const auto& x_i = points_[i];
            hvector<DIM> candidate;

            if (u01(gen_) < 0.5) {
                for (size_t k = 0; k < DIM; ++k) {
                    double R = u01(gen_);
                    double RB = nrm(gen_);
                    constexpr double PI = 3.14159265358979323846;
                    double cauchy = 1.0 + std::tan(0.5 * PI * (u01(gen_) - 0.5));
                    candidate[k] = x_i[k] + (-R + 2.0 * RB) * x_i[k] * cauchy;
                }
            }
            else {
                std::uniform_int_distribution<size_t> idx_distr(0, N - 1);
                size_t p_idx;
                do { p_idx = idx_distr(gen_); } while (p_idx == i);
                const auto& prey = points_[p_idx];

                std::uniform_int_distribution<int> i_distr(1, 2);
                const double I = static_cast<double>(i_distr(gen_));
                for (size_t k = 0; k < DIM; ++k) {
                    double RB = nrm(gen_);
                    candidate[k] = x_i[k] + RB * (prey[k] - I * x_i[k]);
                }
            }

            repair_bounds(candidate);

            double fval = func_(candidate);
            ++step_;

            greedy_select(i, candidate, fval);
        }

        double levy_sample() {
            constexpr double eta = 1.5;
            constexpr double s = 0.01;
            // Precomputed sigma for eta = 1.5:
            //   sigma = ( Gamma(1+eta) * sin(pi*eta/2)
            //             / ( Gamma((1+eta)/2) * eta * 2^((eta-1)/2) ) )^(1/eta)
            // Evaluates to approximately 0.6966 for eta = 1.5.
            constexpr double sigma = 0.69657251;

            std::normal_distribution<double> u_distr(0.0, sigma);
            std::normal_distribution<double> v_distr(0.0, 1.0);
            double u = u_distr(gen_);
            double v = v_distr(gen_);

            double denom = std::pow(std::abs(v), 1.0 / eta);
            if (denom < 1e-30) denom = 1e-30;
            return s * u / denom;
        }

        void repair_bounds(hvector<DIM>& v) {
            for (size_t k = 0; k < DIM; ++k) {
                const auto [l, r] = search_area_.get(k);
                if (v[k] < l || v[k] > r || !std::isfinite(v[k])) {
                    std::uniform_real_distribution<double> box(l, r);
                    v[k] = box(gen_);
                }
            }
        }

        void greedy_select(size_t i, const hvector<DIM>& candidate, double fval) {
            if (fval > values_[i]) {
                points_[i] = candidate;
                values_[i] = fval;

                if (fval > values_[best_idx_]) best_idx_ = i;

                if (fval > ans_) {
                    ans_ = fval;
                    ans_point_ = candidate;
                    best_upds_.emplace_back(step_, ans_point_, ans_);
                }
            }
        }

        hvector<DIM> gen_point_uniform() {
            hvector<DIM> res;
            for (size_t k = 0; k < DIM; ++k) {
                const auto [l, r] = search_area_.get(k);
                std::uniform_real_distribution<double> distr(l, r);
                res[k] = distr(gen_);
            }
            return res;
        }
    };
} // namespace swarm_algorithm