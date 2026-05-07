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
#include <Eigen/Dense>


namespace swarm_algorithm {

    template <size_t DIM> class sofamg_base final {
    public:
        using function_ptr_type = double (*)(const hvector<DIM>&);

        sofamg_base(function_ptr_type func, const hrect& search_area, uint64_t seed)
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
                make_step(step_);
                ++step_;
            }

            auto steps_end = std::chrono::steady_clock::now();
            std::chrono::duration<double> steps_seconds{ steps_end - steps_begin };
            std::cout << "Steps: " << steps_seconds << "s\n";

            return { ans_point_, ans_ };
        }

        std::vector<std::tuple<size_t, hvector<DIM>, double>> dump_bests() const {
            return best_upds_;
        }

        void set_start_population_size(size_t sz) { start_population_size_ = sz; }
        void set_gamma(double v) { gamma_ = v; }
        void set_Q(double v) {
            if (v < 0.0 || v > 1.0)
                throw std::invalid_argument("Q must lie in [0, 1].");
            Q_ = v;
        }
        void set_H(double v) {
            if (v < 0.0 || v > 1.0)
                throw std::invalid_argument("H must lie in [0, 1].");
            H_ = v;
        }
        void set_N(size_t n) { N_ = n; }

        void set_varsigma(double (*fun)(size_t)) { varsigma_ = fun; }

        void set_phi(double (*fun)(size_t)) { phi_ = fun; }

        void enable_gradient(bool on) { gradient_enabled_ = on; }
        void set_gradient_prob(double p) {
            if (p < 0.0 || p > 1.0)
                throw std::invalid_argument("gradient_prob must lie in [0, 1].");
            gradient_prob_ = p;
        }
        void set_gradient_step_size(double s) { gradient_step_size_ = s; }
        void set_gradient_neighbours(size_t n) {
            if (n < DIM)
                throw std::invalid_argument(
                    "gradient_neighbours must be >= DIM for a well-posed system.");
            gradient_neighbours_ = n;
        }

    private:
        uint64_t seed_;
        function_ptr_type func_;
        hrect search_area_;
        rand_precalc<Xoshiro::Xoshiro256PP> gen_;

        size_t start_population_size_ = 100;
        double gamma_ = 0.01;
        double Q_ = 0.5;
        double H_ = 0.5;
        size_t N_ = 10000;

        bool   gradient_enabled_    = true;
        double gradient_prob_       = 0.1;
        double gradient_step_size_  = -1.0;
        size_t gradient_neighbours_ = 2 * DIM;

        double (*varsigma_)(size_t) = [](size_t k) -> double {
            if (k < 1) k = 1;
            return std::pow(1.0 / static_cast<double>(k),
                            1.0 / (2.0 * static_cast<double>(DIM)));
        };
        double (*phi_)(size_t) = [](size_t k) -> double {
            return std::sqrt(static_cast<double>(k) * 0.001 + 1.0);
        };

        double ans_ = 0.0;
        hvector<DIM> ans_point_;

        size_t step_;
        std::vector<hvector<DIM>> points_;
        std::vector<double> values_;
        std::vector<double> logs_;
        std::vector<std::tuple<size_t, hvector<DIM>, double>> best_upds_;

        std::vector<double> scaled_logs_;
        std::vector<double> probs_;

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

        void compute_probs(size_t k) {
            const size_t np = points_.size();
            const double phi_k = phi_(k);

            scaled_logs_.resize(np);
            probs_.resize(np);

            double max_sl = -std::numeric_limits<double>::infinity();
            for (size_t i = 0; i < np; ++i) {
                scaled_logs_[i] = logs_[i] * phi_k;
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

        size_t roulette_select() {
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

        double truncated_cauchy(double centre, double sigma,
                                double l, double r) {
            const double aL = std::atan((l - centre) / sigma);
            const double aR = std::atan((r - centre) / sigma);
            const double B = aR - aL;
            std::uniform_real_distribution<double> u01(0.0, 1.0);
            const double u = u01(gen_);
            if (B < 1e-12) {
                std::uniform_real_distribution<double> box(l, r);
                return box(gen_);
            }
            double x = centre + sigma * std::tan(u * B + aL);
            if (x < l) x = l;
            else if (x > r) x = r;
            return x;
        }

        double mutate_gene(size_t k, size_t i,
                           const hvector<DIM>& reference,
                           const hvector<DIM>& shift1,
                           const hvector<DIM>& shift2,
                           const hvector<DIM>& max_pop) {
            std::uniform_real_distribution<double> u01(0.0, 1.0);
            const auto [aL, aR] = search_area_.get(i);

            if (u01(gen_) < Q_) {
                const double centre = reference[i]
                    + H_ * (max_pop[i] - reference[i])
                    + H_ * (shift1[i]  - shift2[i]);
                const double sigma = varsigma_(k);
                return truncated_cauchy(centre, sigma, aL, aR);
            }
            else {
                const double centre = reference[i];
                const double sigma = varsigma_(k + N_);
                return truncated_cauchy(centre, sigma, aL, aR);
            }
        }

        double effective_step_size() const {
            if (gradient_step_size_ > 0.0)
                return gradient_step_size_;
            double longest = 0.0;
            for (size_t d = 0; d < DIM; ++d) {
                const auto [l, r] = search_area_.get(d);
                const double len = r - l;
                if (len > longest) longest = len;
            }
            return 0.05 * longest;
        }

        hvector<DIM> clamp_to_rect(const hvector<DIM>& p) const {
            hvector<DIM> c;
            for (size_t d = 0; d < DIM; ++d) {
                const auto [l, r] = search_area_.get(d);
                c[d] = std::max(l, std::min(r, p[d]));
            }
            return c;
        }

        double gradient_step(size_t ref_idx) {
            const size_t np = points_.size();

            if (np < DIM + 1)
                return -std::numeric_limits<double>::infinity();

            const hvector<DIM>& z_prime = points_[ref_idx];
            const double        f_prime = values_[ref_idx];

            const size_t k_neigh = std::min(gradient_neighbours_, np - 1);

            std::vector<std::pair<double, size_t>> dist_idx;
            dist_idx.reserve(np - 1);
            for (size_t i = 0; i < np; ++i) {
                if (i == ref_idx) continue;
                double sq = 0.0;
                for (size_t d = 0; d < DIM; ++d) {
                    const double dd = points_[i][d] - z_prime[d];
                    sq += dd * dd;
                }
                dist_idx.emplace_back(sq, i);
            }
            std::nth_element(dist_idx.begin(),
                dist_idx.begin() + static_cast<ptrdiff_t>(k_neigh),
                dist_idx.end());

            Eigen::MatrixXd A(k_neigh, DIM);
            Eigen::VectorXd b(k_neigh);

            size_t row = 0;
            for (size_t ni = 0; ni < k_neigh; ++ni) {
                const size_t idx = dist_idx[ni].second;
                const double dist = std::sqrt(dist_idx[ni].first);

                if (dist < 1e-14) continue;

                const double inv_dist = 1.0 / dist;
                for (size_t d = 0; d < DIM; ++d)
                    A(row,d) = (points_[idx][d] - z_prime[d]) * inv_dist;

                b(row) = (values_[idx] - f_prime) * inv_dist;
                ++row;
            }

            if (row < DIM)
                return -std::numeric_limits<double>::infinity();

            Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(A);
            if (qr.info() != Eigen::Success)
                return -std::numeric_limits<double>::infinity();
            
            Eigen::VectorXd grad = qr.solve(b);
            double g_norm = grad.norm();

            constexpr double eps_grad = 1e-14;
            if (g_norm < eps_grad)
                return -std::numeric_limits<double>::infinity();

            grad.normalize();

            const double step_sz = effective_step_size();
            hvector<DIM> z_new;
            for (size_t d = 0; d < DIM; ++d)
                z_new[d] = z_prime[d] + step_sz * grad(d);

            z_new = clamp_to_rect(z_new);

            const double f_new = func_(z_new);

            if (f_new > f_prime) {
                points_.emplace_back(z_new);
                values_.push_back(f_new);
                logs_.push_back(std::log(f_new));

                if (f_new > ans_) {
                    ans_ = f_new;
                    ans_point_ = z_new;
                    best_upds_.emplace_back(step_, ans_point_, ans_);
                }
                return f_new;
            }

            return -std::numeric_limits<double>::infinity();
        }

        double make_step(size_t k) {
            compute_probs(k);

            const size_t ref_idx  = roulette_select();
            const size_t sh1_idx  = roulette_select();
            const size_t sh2_idx  = roulette_select();
            const hvector<DIM> reference = points_[ref_idx];
            const hvector<DIM> shift1    = points_[sh1_idx];
            const hvector<DIM> shift2    = points_[sh2_idx];

            size_t max_idx = 0;
            for (size_t i = 1; i < values_.size(); ++i) {
                if (values_[i] > values_[max_idx]) max_idx = i;
            }
            const hvector<DIM> max_pop = points_[max_idx];

            hvector<DIM> mutant;
            for (size_t i = 0; i < DIM; ++i) {
                mutant[i] = mutate_gene(k, i, reference, shift1, shift2, max_pop);
            }

            const double mutant_val = func_(mutant);
            const double mutant_log = std::log(mutant_val);

            const double new_ref_idx = mutant_val > ans_ ? points_.size() : ref_idx;

            if (mutant_val > ans_) {
                ans_ = mutant_val;
                ans_point_ = mutant;
                best_upds_.emplace_back(step_, ans_point_, ans_);
            }

            points_.emplace_back(mutant);
            values_.push_back(mutant_val);
            logs_.push_back(mutant_log);

            if (gradient_enabled_) {
                std::uniform_real_distribution<double> u01(0.0, 1.0);
                if (u01(gen_) < gradient_prob_) {
                    gradient_step(ref_idx);
                }
            }

            compute_probs(k);

            size_t new_np = 0;
            const size_t old_np = points_.size();
            for (size_t i = 0; i < old_np; ++i) {
                if (probs_[i] >= gamma_) {
                    if (new_np != i) {
                        points_[new_np] = points_[i];
                        values_[new_np] = values_[i];
                        logs_[new_np]   = logs_[i];
                    }
                    ++new_np;
                }
            }

            if (new_np == 0) {
                points_[0] = points_[max_idx];
                values_[0] = values_[max_idx];
                logs_[0]   = logs_[max_idx];
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