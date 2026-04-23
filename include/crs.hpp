#pragma once
#include <algorithm>
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

    // Controlled Random Search 2 with Local Mutation (CRS2-LM).
    // Reference:
    //   P. Kaelo, M. M. Ali, "Some variants of the controlled random
    //   search algorithm for global optimization," J. Optim. Theory Appl.
    //   130 (2), 253-264 (2006).
    //
    // Formulated for MAXIMIZATION to match sofa_base API:
    //   'best'  = argmax_i f(x_i)
    //   'worst' = argmin_i f(x_i)
    // Each main-loop iteration consumes exactly one evaluation of f,
    // so `iter_count` in result() has the same meaning as in sofa_base.

    template <size_t DIM> class crs_base final {
    public:
        using function_ptr_type = double (*)(const hvector<DIM>&);

        crs_base(function_ptr_type func, const hrect& search_area, uint64_t seed)
            : seed_(seed), func_(func), search_area_(search_area),
            step_(0), failed_trials_(0) {
            if (func_ == nullptr) {
                throw std::invalid_argument("func was nullptr.");
            }

            if (search_area_.dimensions_cnt() != DIM) {
                throw std::runtime_error("invalid search area.");
            }
        }

        function_ptr_type func() const { return func_; }
        const hrect& search_area() const { return search_area_; }
        size_t failed_trials() const { return failed_trials_; }

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

            // Trial-generation state-machine for the main loop:
            //   need_new_trial         -> build a fresh reflection trial
            //   !need_new_trial && !after_local_mutation
            //                          -> previous trial failed; build local mutation
            //   after_local_mutation   -> previous local mutation also failed;
            //                             next iteration must build a fresh trial
            bool need_new_trial = true;
            bool after_local_mutation = false;

            for (size_t iter = 0; iter < iter_count; ++iter) {
                if (need_new_trial) {
                    build_trial_reflection();
                    need_new_trial = false;
                    after_local_mutation = false;
                }

                double fval = func_(trial_);
                ++step_;

                if (fval > values_[worst_idx_]) {
                    // Trial accepted: replaces worst.
                    points_[worst_idx_] = trial_;
                    values_[worst_idx_] = fval;
                    recompute_best_worst();

                    if (fval > ans_) {
                        ans_ = fval;
                        ans_point_ = trial_;
                        best_upds_.emplace_back(step_, ans_point_, ans_);
                    }

                    need_new_trial = true;
                }
                else {
                    ++failed_trials_;
                    if (!after_local_mutation) {
                        build_trial_local_mutation();
                        after_local_mutation = true;
                    }
                    else {
                        need_new_trial = true;
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
            // Kaelo & Ali default is 10*(n+1); minimum must be n+1
            // (otherwise we cannot form a simplex).
            if (sz < DIM + 1) {
                throw std::invalid_argument(
                    "population size must be >= DIM+1.");
            }
            population_size_ = sz;
        }

    private:
        uint64_t seed_;

        function_ptr_type func_;
        hrect search_area_;
        rand_precalc<Xoshiro::Xoshiro256PP> gen_;

        size_t population_size_ = 10 * (DIM + 1);

        double ans_ = -std::numeric_limits<double>::infinity();
        hvector<DIM> ans_point_;

        size_t step_;
        size_t failed_trials_;

        std::vector<hvector<DIM>> points_;
        std::vector<double> values_;
        std::vector<std::tuple<size_t, hvector<DIM>, double>> best_upds_;

        // Scratch buffers reused between iterations to avoid allocations.
        std::vector<size_t> shuffled_indices_;
        hvector<DIM> trial_;
        hvector<DIM> centroid_;

        size_t best_idx_ = 0;
        size_t worst_idx_ = 0;

        void initialize() {
            gen_.seed(seed_);
            gen_.set_batch_size(40000);
            step_ = 0;
            ans_ = -std::numeric_limits<double>::infinity();
            failed_trials_ = 0;

            points_.clear();
            values_.clear();
            best_upds_.clear();
            points_.reserve(population_size_);
            values_.reserve(population_size_);
            shuffled_indices_.reserve(population_size_);

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

            recompute_best_worst();
        }

        void recompute_best_worst() {
            best_idx_ = 0;
            worst_idx_ = 0;
            const size_t N = values_.size();
            for (size_t i = 1; i < N; ++i) {
                if (values_[i] > values_[best_idx_]) best_idx_ = i;
                if (values_[i] < values_[worst_idx_]) worst_idx_ = i;
            }
        }

        // Build a reflection trial point (CRS2 core step):
        //   1. Pick n distinct random indices from population \ {best_idx_}.
        //   2. Let x_n be the last picked point; remaining n-1 plus best
        //      form the simplex used for the centroid.
        //   3. Centroid G = (1/n) * (best + sum of n-1 others).
        //   4. Trial = 2*G - x_n, clipped to bounds.
        void build_trial_reflection() {
            const size_t N = points_.size();
            const size_t n = DIM;

            // Fill pool with all indices except best_idx_.
            shuffled_indices_.clear();
            for (size_t i = 0; i < N; ++i) {
                if (i != best_idx_) shuffled_indices_.push_back(i);
            }

            // Partial Fisher-Yates: randomize the first n positions only.
            const size_t pool = shuffled_indices_.size(); // = N - 1
            for (size_t i = 0; i < n; ++i) {
                std::uniform_int_distribution<size_t> distr(i, pool - 1);
                size_t j = distr(gen_);
                std::swap(shuffled_indices_[i], shuffled_indices_[j]);
            }

            // Centroid over { best, indices[0..n-2] }.
            for (size_t k = 0; k < DIM; ++k) {
                centroid_[k] = points_[best_idx_][k];
            }
            for (size_t i = 0; i + 1 < n; ++i) {
                const auto& pt = points_[shuffled_indices_[i]];
                for (size_t k = 0; k < DIM; ++k) centroid_[k] += pt[k];
            }
            const double inv_n = 1.0 / static_cast<double>(n);
            for (size_t k = 0; k < DIM; ++k) centroid_[k] *= inv_n;

            // Trial = 2*centroid - x_n, clipped to bounds.
            const auto& x_n = points_[shuffled_indices_[n - 1]];
            for (size_t k = 0; k < DIM; ++k) {
                double v = 2.0 * centroid_[k] - x_n[k];
                const auto [l, r] = search_area_.get(k);
                if (v < l) v = l;
                else if (v > r) v = r;
                trial_[k] = v;
            }
        }

        // Local mutation (Kaelo & Ali 2006):
        //   x'_k = (1+w_k)*best_k - w_k*trial_k,   w_k ~ U[0,1]   (fresh per coord)
        // Applied to the trial that just failed to improve the worst point.
        void build_trial_local_mutation() {
            std::uniform_real_distribution<double> wdist(0.0, 1.0);
            const auto& best = points_[best_idx_];
            for (size_t k = 0; k < DIM; ++k) {
                double w = wdist(gen_);
                double v = best[k] * (1.0 + w) - w * trial_[k];
                const auto [l, r] = search_area_.get(k);
                if (v < l) v = l;
                else if (v > r) v = r;
                trial_[k] = v;
            }
        }

        hvector<DIM> gen_point_uniform() {
            hvector<DIM> res;
            for (size_t i = 0; i < DIM; ++i) {
                const auto [l, r] = search_area_.get(i);
                std::uniform_real_distribution<double> distr(l, r);
                res[i] = distr(gen_);
            }

            return res;
        }
    };
} // namespace swarm_algorithm