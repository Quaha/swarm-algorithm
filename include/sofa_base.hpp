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
            normal_distr_(search_area, gen_, DIM), step_(0) {
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

            for (size_t iter = 0; iter < iter_count; iter++) {
                double res = make_step(step_);

                if (res > ans_) {
                    ans_ = values_.back();
                    ans_point_ = points_.back();
                    best_upds_.emplace_back(step_, ans_point_, ans_);
                }

                step_++;
            }

            auto steps_end = std::chrono::steady_clock::now();
            std::chrono::duration<double> steps_seconds{ steps_end - steps_begin };
            std::cout << "Steps: " << steps_seconds << "s\n";

            return { ans_point_, ans_ };
        }

        std::vector<std::tuple<size_t, hvector<DIM>, double>> dump_bests() const {
            return best_upds_;
        }

        void set_start_population_size(size_t sz) {
            start_population_size_ = sz;
        }

        void set_gamma(double val) {
            gamma_ = val;
        }

        void set_phi(double (*fun)(size_t)) {
            phi_ = fun;
        }

        void reserve_buffers(size_t size) {
            f_phi_.resize(std::max(f_phi_.size(), size));
            rho_.resize(std::max(rho_.size(), size));
            probs_.resize(std::max(probs_.size(), size));
        }

    private:
        double (*phi_)(size_t) = [](size_t k) -> double {
            return std::sqrt(static_cast<double>(k) * 0.001 + 1.0);
            };

        uint64_t seed_;

        function_ptr_type func_;
        hrect search_area_;
        rand_precalc<Xoshiro::Xoshiro256PP> gen_;
        size_t start_population_size_ = 100;
        double gamma_ = 0.01;

        double ans_ = 0.0;
        hvector<DIM> ans_point_;

        truncated_normal<decltype(gen_)> normal_distr_;
        size_t step_;
        std::vector<hvector<DIM>> points_;
        std::vector<double> logs_;
        std::vector<double> scaled_logs_;
        std::vector<double> values_;
        std::vector<std::tuple<size_t, hvector<DIM>, double>> best_upds_;

        void initialize() {
            gen_.seed(seed_);
            gen_.set_batch_size(40000);
            step_ = 1;
            ans_ = 0.0;

            points_.clear();
            values_.clear();
            logs_.clear();
            best_upds_.clear();

            for (size_t i = 0; i < start_population_size_; ++i) {
                points_.emplace_back(gen_point_uniform());
                values_.push_back(func_(points_.back()));
                logs_.push_back(std::log(values_.back()));

                if (values_.back() > ans_) {
                    ans_ = values_.back();
                    ans_point_ = points_.back();
                    best_upds_.emplace_back(step_, ans_point_, ans_);
                }
                ++step_;
            }
        }

        std::vector<double> f_phi_;
        std::vector<double> rho_;
        std::vector<double> probs_;

        size_t select() {
            std::uniform_real_distribution<double> u01(0.0, 1.0);
            const double r = u01(gen_);
            double cum = 0.0;
            const size_t np = probs_.size();
            for (size_t i = 0; i < np; ++i) {
                cum += probs_[i];
                if (cum >= r) return i;
            }
            return np - 1;
        }

        void compute_probs(size_t k) {
            const size_t np = points_.size();
            const double phi_k = phi_(k);

            scaled_logs_.resize(np);
            probs_.resize(np);

            double max_sl = -std::numeric_limits<double>::infinity();
            for (size_t i = 0; i < np; ++i) {
                scaled_logs_[i] = scaled_logs_[i] * phi_k;
                if (scaled_logs_[i] > max_sl) max_sl = scaled_logs_[i];
            }

            double sum = 0.0;
            for (size_t i = 0; i < np; ++i) {
                probs_[i] = std::exp(scaled_logs_[i] - max_sl);
                sum += probs_[i];
            }
            const double inv_sum = 1.0 / sum;
            for (size_t i = 0; i < np; ++i) probs_[i] *= inv_sum;
        }

        hvector<DIM> mutate(size_t k,
            const hvector<DIM>& pivot) {

            hvector<DIM> new_point;

            double r2 = 0.0;
            for (int i = 0; i < search_area_.dimensions_cnt(); i++) {
                double d = search_area_.get(i).second - search_area_.get(i).first;
                r2 += d * d;
            }

            double stddev = sqrt(2 * r2 / k);

            for (size_t i = 0; i < DIM; i++) {

                std::normal_distribution distr(pivot[i], stddev);
                const auto [l, r] = search_area_.get(i);

                bool accept = false;
                double x = 0.0;
                while (!accept) {
                    double candidate = distr(gen_);
                    if (l <= candidate && candidate <= r) {
                        x = candidate;
                        accept = true;
                    }
                }

                new_point[i] = x;
            }
           
            return new_point;
        }

        double make_step(size_t k) {
            const size_t np = points_.size();

            compute_probs(k);

            const size_t pivot_idx = select();

            size_t max_idx = 0;
            for (size_t i = 1; i < values_.size(); ++i) {
                if (values_[i] > values_[max_idx]) max_idx = i;
            }
            const hvector<DIM> max_pop = points_[max_idx];

            // gen new point
            hvector<DIM> mutant = mutate(k, points_[pivot_idx]);

            // (5) Evaluate the mutant. We do this BEFORE appending so
            // that we can return the true mutant value even if the
            // repopulation step later filters it out.
            const double mutant_val = func_(mutant);
            const double mutant_log = std::log(mutant_val);

            // Update the best-so-far tracker here, based on the mutant
            // we just evaluated. (The caller result() also looks at
            // values_.back(), but the repopulation step can filter the
            // mutant out of the population, so we handle the book-
            // keeping locally to keep the semantics clean.)
            if (mutant_val > ans_) {
                ans_ = mutant_val;
                ans_point_ = mutant;
                best_upds_.emplace_back(step_, ans_point_, ans_);
            }

            // Append the mutant to the population, then repopulate.
            // The slide runs "NP := NP + 1; population[NP] <- mutation"
            // and *then* computes SumF over all NP, so we do the same.
            points_.emplace_back(mutant);
            values_.push_back(mutant_val);
            logs_.push_back(mutant_log);

            // Recompute probabilities with the mutant included.
            compute_probs(k);

            // Keep indices with prob >= gamma (slide uses ">=", base
            // sofa code uses ">"; we follow the slide).
            size_t new_np = 0;
            const size_t old_np = points_.size();
            for (size_t i = 0; i < old_np; ++i) {
                if (probs_[i] >= gamma_) {
                    if (new_np != i) {
                        points_[new_np] = points_[i];
                        values_[new_np] = values_[i];
                        logs_[new_np] = logs_[i];
                    }
                    ++new_np;
                }
            }
            // Safety: never let the population go empty -- otherwise
            // the next roulette operates on an empty range. If gamma
            // happens to kill everyone, keep the single best agent.
            if (new_np == 0) {
                // Index `max_idx` is stale after the filter loop above,
                // but the argmax over the original population is still
                // a valid surviving agent for our fallback.
                points_[0] = points_[max_idx];
                values_[0] = values_[max_idx];
                logs_[0] = logs_[max_idx];
                new_np = 1;
            }
            points_.resize(new_np);
            values_.resize(new_np);
            logs_.resize(new_np);

            return mutant_val;
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