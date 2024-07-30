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
    Tensor() = default;

public:
    Tensor(std::array<size_t, dimensions> const dim_sizes):
        is_nested(false)
    {
        sizes = new size_t[dimensions];
        data_size = 1;
        for(int i = 0; i < dimensions; ++i){
            sizes[i] = dim_sizes[i];
            data_size *= dim_sizes[i];
        }

        int temp = data_size;
        for(int i = 0; i < dimensions; ++i){
            temp /= dim_sizes[i];
            sizes_array[i] = temp;
        }

        data = new Type[data_size];        
    }

    auto operator [](size_t index) const {
        if(index >= sizes[0]){
            throw std::runtime_error("Indexing error: out of bounds");
        }
        
        NextDimensionType nested = NextDimensionType();
        nested.is_nested = true;
        nested.sizes = this->sizes + 1;
        std::memcpy(&nested.sizes_array[0], &this->sizes_array[1], sizeof(size_t) * (dimensions - 1));
        nested.data_size = this->data_size / sizes[0];
        nested.data = &this->data[index * sizes[0]];
        
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

    IndexedIterator ibegin() const {
        return IndexedIterator(data, &sizes_array, 0);
    }

    IndexedIterator iend() const {
        return IndexedIterator(data + data_size, &sizes_array, data_size);
    }

protected:
    bool is_nested;
protected:
    Type* data;
    size_t data_size;
    size_t* sizes;
    std::array<size_t, dimensions> sizes_array;
};

template <class Type>
class Tensor<Type, 1> {
public:
    using ContainerType = Tensor<Type, 1>;
    friend class Tensor<Type, 2>;

protected:
    Tensor() = default;

public:
    Tensor(std::array<size_t, 1> const dim_sizes):
        is_nested(false)
    {
        sizes = new size_t[1];
        sizes[0] = dim_sizes[0];

        data_size = dim_sizes[0];

        data = new Type[data_size];
    }

    Type& operator [](size_t index) {
        if(index >= sizes[0]){
            throw std::runtime_error("Indexing error: out of bounds");
        }

        return data[index];
    }

    size_t dimension_size(){
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

    IndexedIterator ibegin() const {
        return IndexedIterator(data, &sizes_array, 0);
    }

    IndexedIterator iend() const {
        return IndexedIterator(data + data_size, &sizes_array, data_size);
    }

protected:
    bool is_nested;
protected:
    Type* data;
    size_t data_size;
    size_t* sizes;
    std::array<size_t, 1> sizes_array;
};