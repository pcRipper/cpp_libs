#pragma once


/// @brief 
/// @tparam Type 
/// @tparam size sets max size of the stack
/// @tparam offset sets offset from start of the stack(0 index), indexing will be computed so : stack[index_required + offset]  
template<int size, class Type = int, int offset = 0>
class StaticStack{
    Type stack[size];
    int currentSize;

    static constexpr int MAX_SIZE = size - offset;
public:
    StaticStack(){
        currentSize = 0;
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

    int getSize(){
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

    ~StaticStack() = default;
};