#pragma once

#include "../Iterators/TensorIndexedIterator.hpp"
#include "../Iterators/ContinuousIterator.hpp"
#include "../Iterators/ReversedContinuousIterator.hpp"

#include <stdio.h>
#include <stdexcept>
#include <cstring>
#include <string>
#include <array>

template <class Type, size_t dimensions>
class Tensor{
public:
    static_assert(dimensions > 0, "Template parameter must be greater than 0");
public:
    //general aliases
    using ValueType          = Type;
    using ContainerType      = Tensor<Type, dimensions>;
    using NextDimensionType  = Tensor<Type, dimensions - 1>; 
    friend class Tensor<Type, dimensions + 1>;

protected:
    Tensor() = delete;

    Tensor( 
        size_t *sizes, 
        size_t *offsets_array,
        size_t data_size
    ):
        is_nested(true),
        sizes(sizes),
        offsets_array(offsets_array),
        data_size(data_size),
        nested(NextDimensionType(sizes + 1, offsets_array + 1, data_size / sizes[0]))
    {

    }

public:
    Tensor(std::array<size_t, dimensions> const dim_sizes):
        is_nested(false),
        sizes(new size_t[dimensions]),
        offsets_array(new size_t[dimensions]),
        data_size([&]() -> size_t {
            size_t res = 1;
            for(int i = 0; i < dimensions; ++i){
                offsets_array[dimensions - i - 1] = res;
                sizes[i] = dim_sizes[i];
                res *= dim_sizes[i];
            }
            return res;
        }()),
        data(new Type[data_size]),
        nested(NextDimensionType(sizes + 1, offsets_array + 1, data_size / sizes[0]))
    {}

    NextDimensionType& operator [](size_t index) {
        if(index >= sizes[0]){
            throw std::runtime_error("Indexing error: out of bounds");
        }
        
        nested.data = &this->data[index * offsets_array[0]];
        
        return nested;
    }

    size_t dimension_size() const {
        return sizes[0];
    }

    void fill(Type const value){
        for(size_t i = 0; i < data_size; ++i){
            data[i] = value;
        }
    }

    ~Tensor(){
        if(is_nested)return;
        delete[] sizes;
        delete[] data;
        delete[] offsets_array;
    }

//iterators
public:
    //Simple iterator
    using ForwardIterator  = ContinuousIterator<ContainerType>;
    //Simple reversed iterator
    using ReversedIterator = ReversedContinuousIterator<ContainerType>;
    //Indexed iterator
    using DimensionsType   = typename std::array<size_t, dimensions>;
    using IndexedIterator  = TensorIndexedIterator<ContainerType>;
public:

    ForwardIterator begin() const {
        return ForwardIterator(data);
    }

    ForwardIterator end() const {
        return ForwardIterator(data + data_size);
    }

    ReversedIterator rbegin() const {
        return ReversedIterator(data + data_size - 1);
    }

    ReversedIterator rend() const {
        return ReversedIterator(data - 1);
    }

    // IndexedIterator ibegin() const {
    //     return IndexedIterator(data, &offsets_array, 0);
    // }

    // IndexedIterator iend() const {
    //     return IndexedIterator(data + data_size, &offsets_array, data_size);
    // }

protected:
    bool is_nested;
    size_t* sizes;
    size_t* offsets_array;
    size_t data_size;
    Type* data;
    NextDimensionType nested;
};

template <class Type>
class Tensor<Type, 1> {
public:
    using ContainerType = Tensor<Type, 1>;
    friend class Tensor<Type, 2>;

protected:
    Tensor() = delete;

    Tensor( 
        size_t *sizes, 
        size_t *offsets_array,
        size_t data_size
    ):
        is_nested(true),
        data_size(data_size)
    {}

public:
    Tensor(size_t data_size):
        is_nested(false),
        data_size(size),
        data(new Type[size])
    {}

    Type& operator [](size_t index) {
        if(index >= data_size){
            throw std::runtime_error("Indexing error: out of bounds");
        }

        return data[index];
    }

    size_t dimension_size(){
        return data_size;
    }

    void fill(Type const value){
        for(size_t i = 0; i < data_size; ++i){
            data[i] = value;
        }
    }

    ~Tensor(){
        if(is_nested)return;
        delete[] data;
    }

//iterators
public:
    //Simple iterator
    using ForwardIterator  = ContinuousIterator<ContainerType>;
    //Simple reversed iterator
    using ReversedIterator = ReversedContinuousIterator<ContainerType>;
    //Indexed iterator
    using DimensionsType   = typename std::array<size_t, 1>;
    using IndexedIterator  = TensorIndexedIterator<ContainerType>;
public:

    ForwardIterator begin() const {
        return ForwardIterator(data);
    }

    ForwardIterator end() const {
        return ForwardIterator(data + data_size);
    }

    ReversedIterator rbegin() const {
        return ReversedIterator(data + data_size - 1);
    }

    ReversedIterator rend() const {
        return ReversedIterator(data - 1);
    }

    // IndexedIterator ibegin() const {
    //     return IndexedIterator(data, &offsets_array, 0);
    // }

    // IndexedIterator iend() const {
    //     return IndexedIterator(data + data_size, &offsets_array, data_size);
    // }
protected:
    bool is_nested;
    size_t data_size;
    Type* data;
};