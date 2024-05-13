#pragma once
#include "ContinuousIterator.hpp"

/// @brief Iterator for datastructures with continuous memory allocation block in reversed order
/// @tparam Container is type of the container with continuous memory allocation block
template <class Container>
class ReversedContinuousIterator : public ContinuousIterator<Container>{
public:
    using BaseType = ContinuousIterator<Container>;
    using RCI = ReversedContinuousIterator;
public:
    ReversedContinuousIterator() = delete;

    ReversedContinuousIterator(BaseType::PointerType ptr):
        BaseType(ptr)
    {};

    virtual ReversedContinuousIterator& operator++() override {
        --BaseType::pointer;
        return *this;
    }

    virtual ReversedContinuousIterator& operator++(int) override {
        auto tmp = new ReversedContinuousIterator(BaseType::pointer);
        --BaseType::pointer;
        return *tmp;
    }

    virtual ReversedContinuousIterator& operator+(int offset) override {
        return *new ReversedContinuousIterator(BaseType::pointer - offset);
    }

    virtual ReversedContinuousIterator& operator-(int offset) override {
        return *new ReversedContinuousIterator(BaseType::pointer + offset);
    }

    bool operator<(const ReversedContinuousIterator& right) {
        return BaseType::pointer > right.pointer;
    }

    bool operator>(const ReversedContinuousIterator& right) {
        return BaseType::pointer < right.pointer;
    }

    bool operator<=(const ReversedContinuousIterator& right) {
        return BaseType::pointer >= right.pointer;
    }

    bool operator>=(const ReversedContinuousIterator& right) {
        return BaseType::pointer <= right.pointer;
    }
};

#define RCI ReversedContinuousIterator;