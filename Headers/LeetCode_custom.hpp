#pragma once
#include "../../../GeneralLibs/Headers/includes.hpp";
#include "DataStructures.hpp"

class Solution {
public:
    string addStrings(string num1, string num2) {
        if (num1.length() < num2.length())swap(num1, num2);

        int i1 = num1.length(), i2 = num2.length();
        int carry = 0;
        while (0 < i2) {
            int digit = num1[--i1] + num2[--i2] + carry - 96;
            num1[i1] = (digit % 10) + '0';
            carry = digit / 10;
        }

        while (carry != 0 && 0 < i1) {
            int digit = num1[--i1] + carry - 48;
            num1[i1] = digit % 10 + '0';
            carry = digit / 10;
        }

        return carry == 0 ? num1 : "1" + num1;
    }

    string subtractStrings(const string greater, const string lower) {

        string result;
        result.reserve(greater.length());

        int borrow = 0;

        for (int i = greater.length() - 1, j = lower.length() - 1; i >= 0; i--, j--) {
            int diff = (greater[i] - '0') - borrow;
            if (j >= 0) {
                diff -= (lower[j] - '0');
            }

            if (diff < 0) {
                diff += 10;
                borrow = 1;
            }
            else {
                borrow = 0;
            }

            result += (char)(diff + '0');
        }

        reverse(result.begin(), result.end());

        result.erase(0, result.find_first_not_of('0'));

        return result.empty() ? "0" : result;
    }

    vector<string> NnextSquares(string first, string second, const int N) {
        string iterator = subtractStrings(second, first);

        vector<string> result(N);

        int prefix = 3;
        for (int i = 0; i < N; ++i) {
            iterator = addStrings(iterator, "2");
            second = addStrings(second, iterator);
            result[i] = second;
        }

        return result;
    }
};