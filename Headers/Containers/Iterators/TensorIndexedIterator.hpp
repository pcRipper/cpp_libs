#pragma once

#include "../Static/Tensor.hpp"
#include <stdio.h>
#include <memory>

template <class ValueType, class DimensionsType>
struct IndexedValue {
    ValueType* value;
    DimensionsType index;
};

template <class Container>
class TensorIndexedIterator {
public:
    using ValueType      = typename Container::ValueType;
    using DimensionsType = typename Container::DimensionsType; 
    using IndexedType    = IndexedValue<ValueType, DimensionsType>;

    using PointerType  = IndexedType*;
    using ReferenceType = IndexedType&;

public:
    TensorIndexedIterator() = delete;

    TensorIndexedIterator(ValueType* pointer, DimensionsType const* sizes, size_t offset):
        pointer(pointer),
        sizes(sizes),
        offset(offset)
    {}

    TensorIndexedIterator(TensorIndexedIterator const& obj):
        pointer(pointer),
        sizes(sizes),
        offset(offset)
    {}

    TensorIndexedIterator& operator++() {
        pointer += 1;
        offset += 1;
        return *this;
    }

    TensorIndexedIterator& operator++(int) {
        auto to_return = TensorIndexedIterator(*this);
        pointer += 1;
        offset += 1;
        return to_return;
    }

    PointerType operator->(){
        prepareIteratorValue();
        return &iteratorValue;
    }

    ReferenceType operator*(){
        prepareIteratorValue();
        return iteratorValue;
    }

    bool operator==(TensorIndexedIterator const& r) const {
        return pointer == r.pointer;
    }

    bool operator!=(TensorIndexedIterator const& r) const {
        return pointer != r.pointer;
    }

    ~TensorIndexedIterator(){

    }

protected:

    void prepareIteratorValue(){
        size_t t_offset = offset;
        for(size_t i = 0; i < sizes->size(); ++i){
            iteratorValue.index[i] = t_offset / (*sizes)[i];
            t_offset %= (*sizes)[i];
        }
        iteratorValue.value = pointer;
    }

protected:
    size_t offset;
    DimensionsType const* sizes;
    ValueType* pointer;
    IndexedType iteratorValue;
};

