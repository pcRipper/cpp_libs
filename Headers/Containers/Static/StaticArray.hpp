#pragma once
#include "../Iterators/ContinuousIterator.hpp"
#include "../Iterators/ReversedContinuousIterator.hpp"
#include "../FunctionalContainer.hpp"

#include <cstring>
#include <utility>

/// @brief Array with predefined max size
/// @tparam Type Generic content type of the array
/// @tparam capacity Max capacity of the array
template <int Capacity, class Type = int>
class StaticArray
{
public:
    using ValueType = Type;
    using ContainerType = StaticArray<Capacity, Type>;
    //Forwar iterator
    using ForwardIterator = ContinuousIterator<ContainerType>;
    //Reversed iterator
    using ReversedIterator = ReversedContinuousIterator<ContainerType>;

    using FunctionalContainerType = FunctionalContainer<ContainerType, ForwardIterator>;
public:
    StaticArray(int size = 0): functions(this)
    {
        currentSize = std::min(Capacity, std::max(0, size));
        memset(array, 0, sizeof(Type) * Capacity);
    }

    StaticArray(Type const& value) : functions(this)
    {
        currentSize = Capacity;
        
        for(int i = 0; i < Capacity; ++i){
            memcpy(&array[i], &value, sizeof(Type));
        }
    }

    int size(){
        return currentSize;
    }

    void fill(Type const& value){
        for(int i = 0; i < currentSize; ++i){
            memcpy(&array[i], &value, sizeof(Type));
        }
    }

    int capacity(){
        return Capacity;
    }

    void mapOn(std::function<void(Type&)> transformer){
        functions.mapOn(transformer);
    }

    virtual void push_back(Type element){
        if(currentSize == Capacity)return;
        array[currentSize++] = element;
    }

    virtual void emplace_back(Type&& element){
        if(currentSize == Capacity)return;
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
        if(index >= Capacity)return defaultValue;
        return array[index];
    }

    inline void clear(){
        currentSize = 0;
    }

    ForwardIterator begin(){
        return ForwardIterator(array);
    }
    
    ForwardIterator end(){
        return ForwardIterator(array + currentSize);
    }

    ReversedIterator rbegin(){
        return ReversedIterator(array + currentSize - 1);
    }

    ReversedIterator rend(){
        return ReversedIterator(array - 1);
    }

    virtual ~StaticArray() = default;

protected:
    Type array[Capacity];
    int currentSize;

    FunctionalContainerType functions;
    static Type defaultValue;
};

template <int Capacity, class Type>
Type StaticArray<Capacity, Type>::defaultValue = Type();