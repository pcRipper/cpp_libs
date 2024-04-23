#pragma once


/// @brief Iterator for datastructures with continuous memory allocation block
/// @tparam Container - is type of the container with continuous memory allocation block
template <class Container>
class ContinuousIterator{
public:
    using ValueType = typename Container::ValueType;
    using PointerType = ValueType*;
    using ReferenceType = ValueType&;
private:
    PointerType pointer;
public:
    ContinuousIterator() = delete;

    ContinuousIterator(PointerType ptr):
        pointer(ptr)
    {};

    ContinuousIterator& operator++(){
        ++pointer;
        return *this;
    }

    ContinuousIterator& operator++(int){
        auto tmp = *this;
        ++pointer;
        return tmp;
    }

    PointerType operator->(){
        return pointer;
    }

    ReferenceType operator*(){
        return *pointer;
    }

    bool operator==(const ContinuousIterator& right) const {
        return pointer == right.pointer;
    }

    bool operator!=(const ContinuousIterator& right) const {
        return pointer != right.pointer;
    }

    ContinuousIterator& operator+(int offset){
        return *new StackIterator(pointer + x);
    }

    ContinuousIterator& operator-(int offset){
        return *new StackIterator(pointer - x);
    }
};