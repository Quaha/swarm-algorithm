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

    // Survival of the Fittest with Anisotropic Mutations (SoFAM).
    // Reference:
    //   O. Kuzenkov, "Algorithm: Survival of the Fittest Anisotropic
    //   Mutations," Lobachevsky State University of Nizhny Novgorod,
    //   lecture at Harbin Institute of Technology (2025).
    //   See docs/lectures/extra/Algorithm_Survival_of_the_Fittest_...pdf
    //
    // SoFAM differs from the base SoFA in two main ways:
    //   1. Mutations use a TRUNCATED CAUCHY distribution (with fatter
    //      tails than the normal), which samples large jumps more often
    //      and thereby improves global exploration.
    //   2. Per coordinate, the mutation centre may be SHIFTED away from
    //      the reference point using the current best and two extra
    //      "shift" agents picked by roulette -- this is the anisotropy
    //      and is controlled by the bernoulli(Q) branch ("strong"
    //      mutation) vs. a tighter local step ("weak" mutation).
    //
    // The five sub-algorithms of the lecture are inlined as private
    // methods:
    //   Generator        -> initialize()
    //   Calculator_F     -> implicit in f evaluations
    //   Selector         -> roulette_select()
    //   Mutation_Operator-> mutate_gene() per coordinate
    //   main loop        -> make_step()
    //
    // Formulated for MAXIMIZATION, to match the rest of the suite. One
    // call to result(iter_count) consumes exactly iter_count additional
    // f-evaluations (one per main-loop step), so `iter_count` has the
    // same meaning as in sofa_base/crs_base/de_base/mingo_base and the
    // four algorithms share a common x-axis for convergence plots.
    //
    // IMPORTANT NOTES ON THE MUTATION FORMULA
    // ---------------------------------------
    // The lecture slide for Algorithm 5 gives the mutated-gene formula
    // as
    //   mutated_gene <- reference_point[i]
    //                   + s(k) * tan( rand(0,1) / B
    //                                 + arctan((a^L - centre)/s(k)) )
    // Two things in this expression appear to be typos:
    //   (a) "rand(0,1) / B" should be "rand(0,1) * B". The canonical
    //       inverse CDF of a truncated Cauchy on [a, b] with centre mu
    //       and scale sigma is
    //           x = mu + sigma * tan( u*B + arctan((a-mu)/sigma) ),
    //       where B = arctan((b-mu)/sigma) - arctan((a-mu)/sigma) and
    //       u ~ U[0,1]. Dividing by B instead causes the tangent
    //       argument to blow up whenever B is small (i.e. whenever the
    //       centre is far from the interval), producing near-infinite
    //       gene values that are clipped to the bounds. That would turn
    //       the mutation into "clip to boundary" most of the time,
    //       which contradicts the convergence theorems.
    //   (b) The final expression uses "reference_point[i]" as the base
    //       of the shift, but the Cauchy distribution is centred at
    //       "centre_point" (which can differ from the reference point
    //       in the strong-mutation branch). For the inverse CDF to
    //       produce a sample of the intended distribution, the base
    //       must be centre_point.
    //
    // We implement the canonical inverse CDF (u*B, centre-based). If a
    // verbatim reading of the slide is ever required, only one line in
    // mutate_gene() has to change: replace "centre + sigma * tan(u * B
    // + ...)" with "reference[i] + sigma * tan(u / B + ...)".

    template <size_t DIM> class sofam_base final {
    public:
        using function_ptr_type = double (*)(const hvector<DIM>&);

        sofam_base(function_ptr_type func, const hrect& search_area, uint64_t seed)
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
                // make_step handles its own best-tracker update because
                // the mutant may be filtered out of the population right
                // after evaluation -- so we can't rely on values_.back()
                // here. See the note inside make_step.
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

        // varsigma(k) -- mutation spread; must be positive, monotonically
        // decreasing to 0. Default per Theorem 1: (1/k)^(1/(2n)).
        void set_varsigma(double (*fun)(size_t)) { varsigma_ = fun; }

        // phi(k) -- selection pressure; monotonically increasing to
        // infinity. Default is the same sqrt schedule used in sofa_base
        // so that results are directly comparable.
        void set_phi(double (*fun)(size_t)) { phi_ = fun; }

    private:
        uint64_t seed_;
        function_ptr_type func_;
        hrect search_area_;
        rand_precalc<Xoshiro::Xoshiro256PP> gen_;

        // Defaults mirror sofa_base where possible, with SoFAM-specific
        // parameters taken from sensible values in the lecture.
        size_t start_population_size_ = 100;
        double gamma_ = 0.01;
        double Q_ = 0.5;   // probability of the "strong" mutation branch
        double H_ = 0.5;   // expectation-shift magnitude in strong branch
        size_t N_ = 10000; // time gap for weak-mutation shrinkage

        // Theorem 1: varsigma(k) = (1/k)^(1/(2n)) guarantees the
        // everywhere-dense property of the generated sequence. The cast
        // to double for DIM is deliberate; this is a pointer-to-function
        // so it must close over no state -- we instead read DIM via the
        // class template parameter at the point of use.
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

        // Scratch buffers reused across steps to avoid reallocations.
        std::vector<double> scaled_logs_;  // scaled_logs_[i] * phi(k)
        std::vector<double> probs_;        // normalised probabilities

        void initialize() {
            gen_.seed(seed_);
            gen_.set_batch_size(40000);
            step_ = 1;
            ans_ = 0.0;

            points_.clear();
            values_.clear();
            logs_.clear();
            best_upds_.clear();

            // Algorithm 2 (Generator): draw m uniformly random points
            // from the hyperparallelepiped.
            for (size_t i = 0; i < start_population_size_; ++i) {
                points_.emplace_back(gen_point_uniform());
                // Algorithm 3 (Calculator_F): evaluate f on each.
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

        // Compute the normalised selection probabilities
        //   prob[i] = (F[i])^phi(k) / sum_j (F[j])^phi(k)
        // via the log-sum-exp trick to avoid overflow when phi(k) * log F
        // is large. Cached in probs_ for use by both the selector and
        // the repopulation step.
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

        // Algorithm 4 (Selector): roulette-wheel pick using probs_.
        // Returns the selected population index.
        size_t roulette_select() {
            std::uniform_real_distribution<double> u01(0.0, 1.0);
            const double r = u01(gen_);
            double cum = 0.0;
            const size_t np = probs_.size();
            for (size_t i = 0; i < np; ++i) {
                cum += probs_[i];
                if (cum >= r) return i;
            }
            return np - 1;  // numerical safety
        }

        // Sample one coordinate from a truncated Cauchy on [l, r] with
        // centre `centre` and scale `sigma`, using inverse-CDF transform.
        // If B comes out numerically 0 (centre far from the interval on
        // the scale sigma), fall back to uniform on [l, r].
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

        // Algorithm 5 (Mutation Operator) -- generate one coordinate
        // of the new point. See the long comment at the top of the file
        // for the typos in the lecture formula and how we resolve them.
        double mutate_gene(size_t k, size_t i,
                           const hvector<DIM>& reference,
                           const hvector<DIM>& shift1,
                           const hvector<DIM>& shift2,
                           const hvector<DIM>& max_pop) {
            std::uniform_real_distribution<double> u01(0.0, 1.0);
            const auto [aL, aR] = search_area_.get(i);

            if (u01(gen_) < Q_) {
                // Strong mutation: centre shifted toward the population
                // best and along the difference of two roulette picks.
                const double centre = reference[i]
                    + H_ * (max_pop[i] - reference[i])
                    + H_ * (shift1[i]  - shift2[i]);
                const double sigma = varsigma_(k);
                return truncated_cauchy(centre, sigma, aL, aR);
            }
            else {
                // Weak mutation: tight local step around the reference
                // point, with sigma = varsigma(k + N) (smaller than in
                // the strong branch, yielding a finer-grained search).
                const double centre = reference[i];
                const double sigma = varsigma_(k + N_);
                return truncated_cauchy(centre, sigma, aL, aR);
            }
        }

        double make_step(size_t k) {
            // (1) Update the selection probabilities for this k.
            compute_probs(k);

            // (2) Three roulette draws for reference and shift points.
            const size_t ref_idx  = roulette_select();
            const size_t sh1_idx  = roulette_select();
            const size_t sh2_idx  = roulette_select();
            const hvector<DIM> reference = points_[ref_idx];
            const hvector<DIM> shift1    = points_[sh1_idx];
            const hvector<DIM> shift2    = points_[sh2_idx];

            // (3) Find the current best as the mutation-shift anchor.
            size_t max_idx = 0;
            for (size_t i = 1; i < values_.size(); ++i) {
                if (values_[i] > values_[max_idx]) max_idx = i;
            }
            const hvector<DIM> max_pop = points_[max_idx];

            // (4) Anisotropic mutation, coordinate by coordinate.
            hvector<DIM> mutant;
            for (size_t i = 0; i < DIM; ++i) {
                mutant[i] = mutate_gene(k, i, reference, shift1, shift2, max_pop);
            }

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
                        logs_[new_np]   = logs_[i];
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
