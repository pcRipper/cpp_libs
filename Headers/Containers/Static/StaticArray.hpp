#pragma once
#include "../Iterators/ContinuousIterator.hpp"

/// @brief Array with predefined max size
/// @tparam Type Generic content type of the array
/// @tparam capacity Max capacity of the array
template <int capacity, class Type = int>
class StaticArray {
public:
    using ValueType = Type;
    using Iterator = ContinuousIterator<StaticArray<capacity, Type>>;
protected:
    Type array[capacity];
    int currentSize;

    static Type defaultValue;
public:
    StaticArray(){
        currentSize = 0;
        memset(array, 0, sizeof(Type) * capacity);
    }

    int size(){
        return currentSize;
    }

    virtual void push_back(Type element){
        if(currentSize == capacity)return;
        array[currentSize++] = element;
    }

    virtual void emplace_back(Type&& element){
        if(currentSize == capacity)return;
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
        if(index >= capacity)return defaultValue;
        return array[index];
    }

    inline void clear(){
        currentSize = 0;
    }

    virtual Iterator begin(){
        return Iterator(array);
    }
    
    virtual Iterator end(){
        return Iterator(array + currentSize);
    }

    virtual ~StaticArray() = default;
};

template <int capacity, class Type>
Type StaticArray<capacity, Type>::defaultValue = Type();