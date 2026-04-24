#pragma once

#include <cstdint>
#include <iostream>

namespace swarm_algorithm {
    template <typename T> class rand_precalc {
    public:
        using result_type = T::result_type;

        rand_precalc(size_t batch_size = 1000) : batch_size_(batch_size), calls_(0) {}

        constexpr static result_type min() { return T::min(); }

        constexpr static result_type max() { return T::max(); }

        void seed(uint64_t seed) { gen_.seed(seed); }

        size_t batch_size() const { return batch_size_; }
        void set_batch_size(size_t batch_size) {
            if (batch_size == 0) {
                throw std::runtime_error("batch size must be positive.");
            }

            batch_size_ = batch_size;
            calc(batch_size);
        }

        result_type operator()() {
            calls_++;

            if (pos_ >= cache_.size()) {
                calc(batch_size_);
            }

            return cache_[pos_++];
        }

        size_t calls() const { return calls_; }

    private:
        size_t calls_;
        size_t batch_size_;
        T gen_;
        std::vector<result_type> cache_;
        size_t pos_;

        void calc(size_t cnt) {
            cache_.clear();
            cache_.reserve(cnt);
            pos_ = 0;
            for (size_t i = 0; i < cnt; i++) {
                cache_.push_back(gen_());
            }
        }
    };
} // namespace swarm_algorithm