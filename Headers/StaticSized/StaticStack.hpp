#pragma once
#include "../Iterators/ContinuousIterator.hpp"

/// @brief  Stack with predefined max size
/// @tparam Type 
/// @tparam Size sets max size of the stack
/// @tparam Offset sets offset from start of the stack(0 index), indexing will be computed so : stack[index_required + offset]  
template<int capacity, class Type = int, int offset = 0>
class StaticStack{
public:
    using ValueType = Type;
    using Iterator = ContinuousIterator<StaticStack<capacity, ValueType, offset>>;
private:
    Type stack[capacity];
    int currentSize;

    static constexpr int MAX_SIZE = capacity - offset;
public:
    StaticStack(){
        currentSize = 0;
        memset(stack, 0, sizeof(ValueType) * capacity);
    };

    void push(Type element){
        if(currentSize >= MAX_SIZE)return;
        stack[currentSize++ + offset] = element;
    }

    Type top(){
        return currentSize == 0
            ? Type()
            : stack[currentSize - 1 + offset]
        ;
    }

    Type pop(){
        if(currentSize == 0)return Type();
        return currentSize == 0
            ? Type()
            : stack[--currentSize + offset]
        ;
    }

    int size(){
        return currentSize; 
    }

    void clear(){
        currentSize = 0;
    }

    Type& operator[](int index) {
        static Type defaultValue = Type();
        if(index >= currentSize)return defaultValue;
        return stack[index + offset];
    }

    Iterator begin(){
        return Iterator(stack + offset);
    }

    Iterator end(){
        return Iterator(stack + offset + currentSize);
    }

    ~StaticStack() = default;
};