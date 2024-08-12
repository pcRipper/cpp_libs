#pragma once

#include <utility>

/// @brief Iterator with index for datastructures with continuous memory allocation block
/// @tparam Container is type of the container with continuous memory allocation block
template <class Container>
class IndexedContinuousIterator{
public:
    using ValueType = typename Container::ValueType;
    using indexed_value = typename std::pair<ValueType*, int>; 
    using PointerType = indexed_value*;
    using ReferenceType = indexed_value&;
protected:
    indexed_value data;
public:
    IndexedContinuousIterator() = delete;

    IndexedContinuousIterator(ValueType* ptr, int index):
        data(std::make_pair(ptr, index))
    {};

    virtual IndexedContinuousIterator& operator++(){
        ++data.first;
        ++data.second;
        return *this;
    }

    virtual IndexedContinuousIterator operator++(int){
        auto tmp = IndexedContinuousIterator(data.first, data.second);
        ++data.first;
        ++data.second;
        return tmp;
    }

    virtual PointerType operator->(){
        return &data;
    }

    virtual ReferenceType operator*(){
        return data;
    }

    virtual bool operator==(const IndexedContinuousIterator& right) const {
        return data.first == right.data.first;
    }

    virtual bool operator!=(const IndexedContinuousIterator& right) const {
        return data.first != right.data.first;
    }

    virtual IndexedContinuousIterator operator+(int offset){
        return IndexedContinuousIterator(data.first + offset, data.second + offset);
    }

    virtual IndexedContinuousIterator operator-(int offset){
        return IndexedContinuousIterator(data.first - offset, data.second - offset);
    }

    bool operator<(const IndexedContinuousIterator& right){
        return data.first < right.data.first;
    }

    bool operator>(const IndexedContinuousIterator& right){
        return data.first > right.data.first;
    }

    bool operator<=(const IndexedContinuousIterator& right){
        return data.first <= right.data.first;
    }

    bool operator>=(const IndexedContinuousIterator& right){
        return data.first >= right.data.first;
    }
};