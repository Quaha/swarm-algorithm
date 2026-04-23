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

    // Multi-Strategy Integrated Northern Goshawk Optimizer (MINGO).
    // Reference:
    //   F. Yang, H. Jiang, L. Lyu, "Multi-strategy fusion improved Northern
    //   Goshawk optimizer is used for engineering problems and UAV path
    //   planning," Scientific Reports 14:23300 (2024).
    //
    // The paper layers four enhancements on the original NGO (Dehghani
    // et al. 2021):
    //   -- Levy flight exploration (Eq. 1-4)
    //   -- Brownian motion (Eq. 5) as an alternative exploration rule
    //   -- Cauchy mutation (Eq. 8-9) in the exploitation phase
    //   -- A reinforced exploitation formula (Eq. 10)
    //
    // Pseudocode (Table 1 in the paper):
    //   Phase 1 (exploration)   per agent i:
    //       p = random prey; if f(p) > f(x_i) use Eq. (1), else Eq. (5).
    //   Phase 2 (exploitation)  per agent i:
    //       if rand < 0.5 use Eq. (9); else Eq. (10).
    //
    // Formulated for MAXIMIZATION to match sofa_base/crs_base/de_base.
    //
    // One outer "generation" in the paper performs two position updates
    // per agent, hence 2*N fitness evaluations. To keep the same "iter
    // == one eval" semantics used by sofa_base, crs_base, and de_base,
    // we unroll the generation into a phase-aware state machine driven
    // by a single per-iteration counter.
    //
    // Ambiguities in the paper and choices made here:
    //   * "prey p_{i,j}" in Eq. (5), (10): identified with a random
    //     population member j != i, matching the original NGO.
    //   * "R" in Eq. (9): not defined in the paper. Interpreted as
    //     R ~ U[0, 1], re-drawn per iteration as is customary for NGO
    //     variants.
    //   * "RB (Brownian)": N(0, 1), the standard interpretation in the
    //     broader literature (e.g., Marine Predators Algorithm).
    //   * Out-of-bounds components after an update: re-sampled uniformly
    //     within bounds (same scheme as de_base).

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

                // Advance the (agent, phase) counter. One outer
                // generation touches every agent twice: once in phase 1,
                // then once in phase 2.
                ++current_i_;
                if (current_i_ >= points_.size()) {
                    current_i_ = 0;
                    phase_ = 1 - phase_;
                    if (phase_ == 0) {
                        // Starting a new generation -- best may have
                        // shifted during phase 2.
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
            // Sub-phase-1 needs at least one other agent as "prey", so
            // NP >= 2. Paper uses NP = 50.
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

        // Phase 1: prey identification (exploration).
        //   pick random prey p, p != i.
        //   if f(p) > f(x_i): apply Levy flight update (Eq. 1)
        //     x_new = x_best + RL * (x_best - I * x_i),  RL = 0.5*Levy(D)
        //   else:                apply Brownian update (Eq. 5)
        //     x_new = x_i + RB * (x_i - p)
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
                // Eq. (1): Levy-guided pull toward x_best.
                const auto& x_best = points_[best_idx_];
                // I in {1, 2}, single draw for the whole vector.
                std::uniform_int_distribution<int> i_distr(1, 2);
                const double I = static_cast<double>(i_distr(gen_));
                for (size_t k = 0; k < DIM; ++k) {
                    double rl = 0.5 * levy_sample();
                    candidate[k] = x_best[k] + rl * (x_best[k] - I * x_i[k]);
                }
            }
            else {
                // Eq. (5): Brownian-driven step away from (inferior) prey.
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

        // Phase 2: chase prey (exploitation).
        //   with prob 0.5:     Cauchy-mutation update (Eq. 9)
        //     x_new = x_i + (-R + 2*RB) * x_i * cauchy,
        //     where cauchy = 1 + tan(0.5*pi*(rand - 0.5))  [Eq. 8]
        //   else:              reinforced exploitation update (Eq. 10)
        //     x_new = x_i + RB * (p - I * x_i)
        // R in Eq. 9 is not defined in the paper; interpreted as U[0,1].
        void phase_2_step(size_t i) {
            const size_t N = points_.size();
            std::uniform_real_distribution<double> u01(0.0, 1.0);
            std::normal_distribution<double> nrm(0.0, 1.0);

            const auto& x_i = points_[i];
            hvector<DIM> candidate;

            if (u01(gen_) < 0.5) {
                // Eq. (9): Cauchy-mutation perturbation of the current
                // agent. Uses a fresh R, RB, and Cauchy draw per coord.
                for (size_t k = 0; k < DIM; ++k) {
                    double R = u01(gen_);
                    double RB = nrm(gen_);
                    // Eq. (8): Cauchy operator.
                    constexpr double PI = 3.14159265358979323846;
                    double cauchy = 1.0 + std::tan(0.5 * PI * (u01(gen_) - 0.5));
                    candidate[k] = x_i[k] + (-R + 2.0 * RB) * x_i[k] * cauchy;
                }
            }
            else {
                // Eq. (10): reinforced exploitation pull toward prey.
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

        // Mantegna's algorithm for Levy(eta=1.5) samples.
        // Section 2.2 of the paper:
        //   Levy(D) = s * (u * sigma) / |v|^(1/eta),
        //   s = 0.01, eta = 1.5, u, v ~ "random numbers in [0,1]".
        // A literal reading of the paper would use u, v ~ U[0,1] but this
        // gives a degenerate, non-symmetric distribution with only
        // positive values. Every common Levy implementation in the
        // nature-inspired optimisation literature (Mantegna 1994, and
        // the mealpy/pymoo codebases) uses u ~ N(0, sigma^2), v ~ N(0,1),
        // which yields the proper stable distribution. We follow that
        // convention here; the paper's text is almost certainly an
        // imprecise description of the same scheme.
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

            // Guard against v very close to 0.
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

        // Greedy selection: accept the candidate if it improves fitness.
        // The paper's pseudocode lines 12 and 20 say "Update i-th
        // positions" after each phase -- NGO is a greedy algorithm that
        // only replaces the agent when the candidate is strictly better.
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