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
            normal_distr_(search_area, gen_, DIM), step_(0), bad_points_(0) {
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

        void set_psi(double (*fun)(size_t)) {
            psi_ = fun;
        }

        void reserve_buffers(size_t size) {
            f_psi_.resize(std::max(f_psi_.size(), size));
            rho_.resize(std::max(rho_.size(), size));
            prob_.resize(std::max(prob_.size(), size));
        }

    private:
        double (*psi_)(size_t) = [](size_t k) -> double {
            return std::sqrt((static_cast<double>(k) * 0.001 + 1.0));
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
        std::vector<double> values_;
        std::vector<std::tuple<size_t, hvector<DIM>, double>> best_upds_;

        size_t bad_points_;

        void initialize() {
            gen_.seed(seed_);
            gen_.set_batch_size(40000);
            step_ = 1;
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
                    best_upds_.emplace_back(step_, ans_point_, ans_);
                }

                step_++;
            }
        }

        std::vector<double> f_psi_;
        std::vector<double> rho_;
        std::vector<double> prob_;

        double make_step(size_t k) {
            const size_t np = points_.size();

            reserve_buffers(np + 1);

            // calc f_psi
            {
                double psi_k = psi_(k);
                for (size_t i = 0; i < np; i++) {
                    f_psi_[i] = std::exp(logs_[i] * psi_k);
                }
            }

            // calc prob
            {
                double sum = 0.0;
                for (size_t i = 0; i < np; i++) {
                    prob_[i] = f_psi_[i];
                    sum += f_psi_[i];
                }

                double inv_sum = 1.0 / sum;
                for (size_t i = 0; i < np; i++) {
                    prob_[i] *= inv_sum;
                }
            }

            // pivot selection
            size_t pivot_index = 0;
            {
                std::uniform_real_distribution pivot_distr(0.0, 1.0);
                double pivot_val = pivot_distr(gen_);
                double sum = 0.0;
                for (size_t i = 0; i < np; i++) {
                    sum += prob_[i];
                    if (sum >= pivot_val) {
                        pivot_index = i;
                        break;
                    }
                }
            }

            // gen new point
            hvector<DIM> new_point;
            {
                double stddev = search_area_.max_dim() * sqrt(search_area_.dimensions_cnt() / log(k));

                for (size_t i = 0; i < DIM; i++) {

                    std::normal_distribution distr(points_[pivot_index][i], stddev);
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

                    //double x = normal_distr_.generate(points_[pivot_index][i], stddev, i);

                    new_point[i] = x;
                }
            }

            // repopulate
            {
                size_t new_np = 0;
                for (size_t i = 0; i < np; i++) {
                    if (prob_[i] > gamma_) {
                        points_[new_np] = points_[i];
                        values_[new_np] = values_[i];
                        logs_[new_np] = values_[i];
                        new_np++;
                    }
                }
                points_.resize(new_np);
                values_.resize(new_np);
                logs_.resize(new_np);
            }

            points_.emplace_back(new_point);
            values_.push_back(func_(points_.back()));
            logs_.push_back(std::log(values_.back()));
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