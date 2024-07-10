#pragma once

#include <stdio.h>
#include <exception>

#include <string>
#include <array>

template <class Type, size_t dimensions>
class Tensor{
public:
    static_assert(dimensions > 0, "Template parameter must be greater than 0");
public:
    using ClassType          = Tensor<Type, dimensions>;
    using NextDimensionType  = Tensor<Type, dimensions - 1>; 
    friend class Tensor<Type, dimensions + 1>;
public:
    //iterator types
public:
    Tensor() = delete;

    Tensor(std::array<size_t, dimensions> const dim_sizes):
        isNested(false)
    {
        sizes = new size_t[dimensions];
        data_size = 1;
        for(int i = 0; i < dimensions; ++i){
            sizes[i] = dim_sizes[i];
            data_size *= dim_sizes[i];
        }

        data = new Type[data_size];
    }

protected:
    static NextDimensionType* prepare_nested(){
        auto res = (NextDimensionType*)malloc(sizeof(NextDimensionType));
        res->isNested = true;
        return res;
    }
public:

    auto operator [](size_t index) const {
        if(index >= sizes[0]){
            throw std::runtime_error("Indexing error: out of bounds");
        }

        auto nested = prepare_nested();
        nested->sizes = this->sizes + 1;
        nested->data  = &this->data[index * sizes[0]];
        nested->data_size = this->data_size / sizes[0];

        return *nested;
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
        if(isNested)return;
        delete[] sizes;
        delete[] data;
    }
protected:
    bool isNested;
protected:
    Type* data;
    size_t data_size;
    size_t* sizes;
};

template <class Type>
class Tensor<Type, 1> {
public:
    using ClassType = Tensor<Type, 1>;
    friend class Tensor<Type, 2>;
public:
    Tensor() = delete;

    Tensor(std::array<size_t, 1> const dim_sizes):
        isNested(false)
    {
        sizes = new size_t[1];
        sizes[0] = dim_sizes[0];

        data_size = dim_sizes[0];

        data = new Type[data_size];
    }

    Type& operator [](size_t index) const {
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
        if(isNested)return;
        delete[] sizes;
        delete[] data;
    }
protected:
    bool isNested;
protected:
    Type* data;
    size_t data_size;
    size_t* sizes;
};