#pragma once
#include <cstddef>
#include <functional>
#include <new>
#include <stack>
#include <vector>

#include "hvector.hpp"

namespace swarm_algorithm {
    template <size_t DIM> class hvector_allocator {
    public:
        hvector_allocator(const hvector_allocator&) = delete;
        hvector_allocator& operator=(const hvector_allocator&) = delete;

        hvector_allocator(size_t cnt) : data_(cnt), allocated_(0) {}

        hvector<DIM>& alloc() {
            if (free_.empty() && allocated_ == data_.size()) {
                throw std::bad_alloc();
            }

            if (!free_.empty()) {
                auto ref = free_.top();
                free_.pop();
                return ref;
            }

            return data_[allocated_++];
        }

        void free(const hvector<DIM>& vec) { free_.push(vec); }

    private:
        std::vector<hvector<DIM>> data_;
        size_t allocated_;
        std::stack<std::reference_wrapper<hvector<DIM>>> free_;
    };
} // namespace swarm_algorithm