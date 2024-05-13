#pragma once
#include "../Iterators/ContinuousIterator.hpp"

/// @brief Array with predefined max size
/// @tparam Type Generic content type of the array
/// @tparam capacity Max capacity of the array
template <int capacity, class Type = int>
class StaticArray {
public:
    using ValueType = Type;
    using ContainerType = StaticArray<capacity, Type>;
    using Iterator = ContinuousIterator<ContainerType>;
    static const int max_capacity = capacity;
protected:
    Type array[max_capacity];
    int currentSize;

    static Type defaultValue;
public:
    StaticArray(){
        currentSize = 0;
        memset(array, 0, sizeof(Type) * ContainerType::max_capacity);
    }

    int size(){
        return currentSize;
    }

    int capacity(){
        return ContainerType::max_capacity;
    }

    virtual void push_back(Type element){
        if(currentSize == ContainerType::max_capacity)return;
        array[currentSize++] = element;
    }

    virtual void emplace_back(Type&& element){
        if(currentSize == ContainerType::max_capacity)return;
        array[currentSize++] = std::move(element);
    }

    inline virtual void pop_back(){
        if(currentSize == 0)return;
        currentSize -= 1;
    }

    virtual Type& back(){
        if(currentSize == 0)return defaultValue;
        return array[currentSize - 1];
    }

    virtual Type& operator[](int index){
        if(index >= ContainerType::max_capacity)return defaultValue;
        return array[index];
    }

    inline void clear(){
        currentSize = 0;
    }

    Iterator begin(){
        return Iterator(array);
    }
    
    Iterator end(){
        return Iterator(array + currentSize);
    }

    virtual ~StaticArray() = default;
};

template <int capacity, class Type>
Type StaticArray<capacity, Type>::defaultValue = Type();