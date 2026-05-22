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
    template <size_t DIM> class de_base final {
    public:
        using function_ptr_type = double (*)(const hvector<DIM>&);

        de_base(function_ptr_type func, const hrect& search_area, uint64_t seed)
            : seed_(seed), func_(func), search_area_(search_area),
            step_(0), accepted_(0) {
            if (func_ == nullptr) {
                throw std::invalid_argument("func was nullptr.");
            }

            if (search_area_.dimensions_cnt() != DIM) {
                throw std::runtime_error("invalid search area.");
            }
        }

        function_ptr_type func() const { return func_; }
        const hrect& search_area() const { return search_area_; }
        size_t accepted() const { return accepted_; }

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
                process_one_target();

                if (++current_i_ >= points_.size()) {
                    current_i_ = 0;
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
            if (sz < 4) {
                throw std::invalid_argument("population size must be >= 4.");
            }
            population_size_ = sz;
        }

        void set_F(double F) {
            if (F <= 0.0) throw std::invalid_argument("F must be positive.");
            F_ = F;
        }

        void set_CR(double CR) {
            if (CR < 0.0 || CR > 1.0)
                throw std::invalid_argument("CR must lie in [0, 1].");
            CR_ = CR;
        }

    private:
        uint64_t seed_;

        function_ptr_type func_;
        hrect search_area_;
        rand_precalc<Xoshiro::Xoshiro256PP> gen_;

        size_t population_size_ = 10 * DIM;
        double F_ = 0.5;
        double CR_ = 0.9;

        double ans_ = -std::numeric_limits<double>::infinity();
        hvector<DIM> ans_point_;

        size_t step_;
        size_t accepted_;
        size_t current_i_ = 0;
        size_t best_idx_ = 0;

        std::vector<hvector<DIM>> points_;
        std::vector<double> values_;
        std::vector<std::tuple<size_t, hvector<DIM>, double>> best_upds_;

        hvector<DIM> trial_;

        void initialize() {
            gen_.seed(seed_);
            gen_.set_batch_size(40000);
            step_ = 0;
            accepted_ = 0;
            ans_ = -std::numeric_limits<double>::infinity();
            current_i_ = 0;

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

        size_t random_index_excluding(
            size_t N,
            std::initializer_list<size_t> exclude) {
            std::uniform_int_distribution<size_t> distr(0, N - 1);
            while (true) {
                size_t idx = distr(gen_);
                bool bad = false;
                for (size_t e : exclude) {
                    if (idx == e) { bad = true; break; }
                }
                if (!bad) return idx;
            }
        }

        void process_one_target() {
            const size_t N = points_.size();
            const size_t i = current_i_;

            size_t r1 = random_index_excluding(N, { i });
            size_t r2 = random_index_excluding(N, { i, r1 });

            const auto& x_i     = points_[i];
            const auto& x_best  = points_[best_idx_];
            const auto& x_r1    = points_[r1];
            const auto& x_r2    = points_[r2];

            hvector<DIM> v;
            for (size_t k = 0; k < DIM; ++k) {
                v[k] = x_i[k]
                     + F_ * (x_best[k] - x_i[k])
                     + F_ * (x_r1[k]   - x_r2[k]);
            }

            std::uniform_real_distribution<double> u01(0.0, 1.0);
            for (size_t k = 0; k < DIM; ++k) {
                const auto [l, r] = search_area_.get(k);
                if (v[k] < l || v[k] > r) {
                    std::uniform_real_distribution<double> box(l, r);
                    v[k] = box(gen_);
                }
            }

            std::uniform_int_distribution<size_t> dim_distr(0, DIM - 1);
            const size_t j_rand = dim_distr(gen_);
            for (size_t k = 0; k < DIM; ++k) {
                if (u01(gen_) <= CR_ || k == j_rand) {
                    trial_[k] = v[k];
                }
                else {
                    trial_[k] = x_i[k];
                }
            }

            // Evaluate and greedy-select (maximisation).
            double fval = func_(trial_);
            ++step_;

            if (fval >= values_[i]) {
                points_[i] = trial_;
                values_[i] = fval;
                ++accepted_;

                if (i == best_idx_) {
                    // The incumbent best improved in place -- still best.
                }
                else if (fval > values_[best_idx_]) {
                    best_idx_ = i;
                }

                if (fval > ans_) {
                    ans_ = fval;
                    ans_point_ = trial_;
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