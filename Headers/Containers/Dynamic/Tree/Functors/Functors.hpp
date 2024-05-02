#pragma once
#include "../../../../includes.hpp"

template <typename Type>
Type* fromLevelOrderFunctor(const string& str) = delete;

template <>
int* fromLevelOrderFunctor(const string& str){
    if(str.length() == 0 || !isdigit(str[0]))return nullptr;
    return new int(stoi(str));
}
