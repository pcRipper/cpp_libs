#pragma once
#include "StaticArray.hpp"

/// @brief  Stack with predefined max size
/// @tparam Type 
/// @tparam Size sets max size of the stack
/// @tparam Offset sets offset from start of the stack(0 index), indexing will be computed so : stack[index_required + offset]  
template<int capacity, class Type = int, int offset = 0>
class StaticStack : private StaticArray<capacity, Type> {
public:
    using BaseClassType = StaticArray<capacity, Type>;
protected:
    static constexpr int MAX_SIZE = capacity - offset;
public:
    StaticStack():
        StaticArray()
    {};

    void push(Type element){
        if(currentSize >= MAX_SIZE)return;
        array[currentSize++ + offset] = element;
    }

    void emplace(Type&& element){
        if(currentSize >= MAX_SIZE)return;
        array[currentSize++ + offset] = std::move(element);
    }

    Type& top(){
        return currentSize == 0
            ? BaseClassType::defaultValue
            : array[currentSize - 1 + offset]
        ;
    }

    Type pop(){
        auto top = back();
        currentSize -= 1;
        return top;
    }

    int size(){
        return BaseClassType::size();
    }

    void clear(){
        BaseClassType::clear();
    }

    virtual Type& operator[](int index){
        return BaseClassType::operator[](index + offset);
    }

    Iterator begin() override {
        return Iterator(array + offset);
    }

    Iterator end() override {
        return Iterator(array + offset + currentSize);
    }

    ~StaticStack() = default;
};
