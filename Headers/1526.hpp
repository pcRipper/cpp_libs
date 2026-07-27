#pragma once

#include <vector>


using namespace std;

class Solution {
public:
    int minNumberOperations(std::vector<int> const& target)
    {
        static vector<pair<int, int>> increasingStack = {{0, -1}};
        increasingStack.resize(1);

        const int SIZE = target.size();

        int result = 0;
        for(int i = 0; i < SIZE; ++i)
        {
            while(increasingStack.back().first > target[i])
            {
                auto[value, index] = increasingStack.back();
                increasingStack.pop_back();

                const auto&[valueLast, indexLast] = increasingStack.back();

                result += value - max(valueLast, target[i]);
            }
            if(target[i] == increasingStack.back().first)
            {
                increasingStack.back().second = i;
            }
            else
            {
                increasingStack.emplace_back(target[i], i);
            }
        }

        while(increasingStack.size() > 1)
        {
            auto[value, index] = increasingStack.back();
            increasingStack.pop_back();

            result += value - increasingStack.back().first;
        }

        return result;
    }
};
