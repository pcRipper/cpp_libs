#pragma once

/// @brief Iterator for datastructures with continuous memory allocation block
/// @tparam Container is type of the container with continuous memory allocation block
template <class Container>
class ContinuousIterator{
public:
    using ValueType = typename Container::ValueType;
    using PointerType = ValueType*;
    using ReferenceType = ValueType&;
protected:
    PointerType pointer;
public:
    ContinuousIterator() = delete;

    ContinuousIterator(PointerType ptr):
        pointer(ptr)
    {};

    virtual ContinuousIterator& operator++(){
        ++pointer;
        return *this;
    }

    virtual ContinuousIterator& operator++(int){
        auto tmp = new ContinuousIterator(pointer);
        ++pointer;
        return *tmp;
    }

    virtual PointerType operator->(){
        return pointer;
    }

    virtual ReferenceType operator*(){
        return *pointer;
    }

    virtual bool operator==(const ContinuousIterator& right) const {
        return pointer == right.pointer;
    }

    virtual bool operator!=(const ContinuousIterator& right) const {
        return pointer != right.pointer;
    }

    virtual ContinuousIterator& operator+(int offset){
        return *new ContinuousIterator(pointer + offset);
    }

    virtual ContinuousIterator& operator-(int offset){
        return *new ContinuousIterator(pointer - offset);
    }

    bool operator<(const ContinuousIterator& right){
        return pointer < right.pointer;
    }

    bool operator>(const ContinuousIterator& right){
        return pointer > right.pointer;
    }

    bool operator<=(const ContinuousIterator& right){
        return pointer <= right.pointer;
    }

    bool operator>=(const ContinuousIterator& right){
        return pointer >= right.pointer;
    }
};