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

            for (size_t i = 0; i < bounds_.size(); i++) {
                const auto& [l, r] = bounds_[i];
                if (l >= r) {
                    throw std::invalid_argument(__FUNCTION__ ": invalid bounds at dimension " + std::to_string(i) + ".");
                }
            }
        }

        size_t dimensions_cnt() const {
            return bounds_.size();
        }

        bounds_type get(size_t index) const {
            return bounds_[index];
        }
    private:
        std::vector<bounds_type> bounds_;
    };
};