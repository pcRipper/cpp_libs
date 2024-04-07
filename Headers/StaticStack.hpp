#pragma once

template<int size, class Type>
class StaticStack{
    static Type stack[size];
    int currentSize;

    static constexpr int MAX_SIZE = size;
public:
    StaticStack(){
        currentSize = 0;
    };

    void push(Type element){
        if(currentSize == MAX_SIZE)return;
        stack[currentSize++] = element;
    }

    Type top(){
        return currentSize == 0
            ? Type()
            : stack[currentSize - 1]
        ;
    }

    Type pop(){
        if(currentSize == 0)return Type();
        return currentSize == 0
            ? Type()
            : stack[--currentSize]
        ;
    }

    int getSize(){
        return currentSize; 
    }

    void clear(){
        currentSize = 0;
    }

    ~StaticStack() = default;
};

template<int size, class Type>
Type StaticStack<size, Type>::stack[size];