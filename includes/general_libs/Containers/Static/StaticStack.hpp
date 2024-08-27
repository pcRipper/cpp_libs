#pragma once
#include "StaticArray.hpp"
#include "../Iterators/ReversedContinuousIterator.hpp"

/// @brief  Stack with predefined max size
/// @tparam Type 
/// @tparam Size sets max size of the stack
/// @tparam Offset sets offset from start of the stack(0 index), indexing will be computed so : stack[index_required + offset]  
template<int capacity, class Type = int, int offset = 0>
class StaticStack : protected StaticArray<capacity, Type> {
public:
    using ValueType = Type;
    using BaseClassType = StaticArray<capacity, Type>;
    using ContainerType = StaticStack<capacity, Type>;
    using Iterator = ReversedContinuousIterator<ContainerType>;
protected:
    static constexpr int MAX_SIZE = capacity - offset;
public:
    StaticStack():
        BaseClassType()
    {};
    
    void push(Type element){
        if(BaseClassType::currentSize >= MAX_SIZE)return;
        BaseClassType::array[BaseClassType::currentSize++ + offset] = element;
    }

    void emplace(Type&& element){
        if(BaseClassType::currentSize >= MAX_SIZE)return;
        BaseClassType::array[BaseClassType::currentSize++ + offset] = std::move(element);
    }

    Type& top(){
        return BaseClassType::currentSize == 0
            ? BaseClassType::defaultValue
            : BaseClassType::array[BaseClassType::currentSize - 1 + offset]
        ;
    }

    Type pop(){
        auto top = BaseClassType::back();
        BaseClassType::currentSize -= 1;
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

    Iterator begin() {
        return Iterator(BaseClassType::array + offset + BaseClassType::currentSize - 1);
    }

    Iterator end() {
        return Iterator(BaseClassType::array + offset - 1);
    }

    ~StaticStack() = default;
};
