#include "sofa_base.hpp"

namespace swarm_algorithm {

    std::pair<std::vector<double>, double> sofa_base::result(size_t iter_count, bool from_start) {
        if (from_start) {
            initialize();
        }

        for (size_t iter = 0; iter < iter_count; iter++) {
            double res = make_step();

            if (res > ans_) {
                ans_ = values_.back();
                ans_point_ = points_.back();
            }
        }

        return { ans_point_, ans_ };
    }

    void sofa_base::initialize() {
        const size_t D = search_area_.dimensions_cnt();

        gen_.seed(123);
        ans_ = 0.0;

        points_.clear();
        for (size_t i = 0; i < start_population_size_; i++) {
            points_.emplace_back(gen_point_uniform());
            values_.push_back(func_(points_.back()));

            if (values_.back() > ans_) {
                ans_ = values_.back();
                ans_point_ = points_.back();
            }
        }
    }

    double sofa_base::make_step() {

        const auto psi_ = [](size_t k) -> double {
            return (static_cast<double>(k) * 0.001);
            };

        const size_t k = points_.size() + 1;

        std::vector<double> f_psi(k - 1);
        for (size_t i = 0; i < f_psi.size(); i++) {
            f_psi[i] = pow(values_[i], psi_(k));
        }

        std::vector<double> rho(k - 1);

        double sum = 0.0;

        for (size_t i = 0; i < rho.size(); i++) {
            rho[i] = f_psi[i];
            sum += f_psi[i];
        }

        for (size_t i = 0; i < rho.size(); i++) {
            rho[i] /= sum;
        }

        // pivot selection
        std::vector<double> prob(k - 1);
        sum = 0.0;
        for (size_t i = 0; i < prob.size(); i++) {
            if (rho[i] < gamma_) continue;

            prob[i] = f_psi[i];
            sum += f_psi[i];
        }

        for (size_t i = 0; i < prob.size(); i++) {
            prob[i] /= sum;
        }

        std::discrete_distribution<size_t> pivot_distr(prob.begin(), prob.end());
        size_t pivot_index = pivot_distr(gen_);

        double stddev = sqrt(search_area_.max_dim() / log(k));

        std::vector<double> new_point(search_area_.dimensions_cnt());
        for (size_t i = 0; i < new_point.size(); i++) {
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

            new_point[i] = x;
        }

        points_.emplace_back(std::move(new_point));
        values_.push_back(func_(points_.back()));

        return values_.back();
    }

    std::vector<double> sofa_base::gen_point_uniform() {
        const size_t D = search_area_.dimensions_cnt();
        std::vector<double> res(D);
        for (size_t i = 0; i < D; i++) {
            const auto [l, r] = search_area_.get(i);
            std::uniform_real_distribution distr(l, r);
            res[i] = distr(gen_);
        }
        
        return res;
    }

}