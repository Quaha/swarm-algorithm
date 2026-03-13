#pragma once
#include <vector>
#include <utility>
#include <initializer_list>
#include <stdexcept>
#include <string>

namespace swarm_algorithm {
    class hrect {
    public:
        using bounds_type = std::pair<double, double>;
        hrect(std::initializer_list<bounds_type> bounds) : bounds_(bounds) {
            if (bounds_.empty()) {
                throw std::invalid_argument(__FUNCTION__ ": hrect must have at least one dimension.");
            }

            max_dim_ = 0.0;
            for (size_t i = 0; i < bounds_.size(); i++) {
                const auto& [l, r] = bounds_[i];
                if (l >= r) {
                    throw std::invalid_argument(__FUNCTION__ ": invalid bounds at dimension " + std::to_string(i) + ".");
                }

                if (r - l > max_dim_) {
                    max_dim_ = r - l;
                }
            }
        }

        size_t dimensions_cnt() const {
            return bounds_.size();
        }

        bounds_type get(size_t index) const {
            return bounds_[index];
        }

        double max_dim() const {
            return max_dim_;
        }
    private:
        std::vector<bounds_type> bounds_;
        double max_dim_;
    };
};