#pragma once
#include <general_libs/includes.hpp>

#define LIMIT 255

class Solution {
public:
    vector<string> restoreIpAddresses(const string &s)
    {
        if(s.length() > 12 || s.length() < 4)
        {
            return {};
        }  

        vector<vector<int>> dp(s.length() + 1);
        dp[0].push_back(-1);

        for(int i = 0; i < s.length(); ++i)
        {
            int current = 0;
            int mask = 1;
            for(int j = i; 0 <= j; --j)
            {
                int currentDigit = s[j] - '0';
                current += currentDigit * mask;
                mask *= 10;

                if(current > LIMIT || i - j > 3)
                {
                    break;
                }
                if(dp[j].empty() || (i != j && currentDigit == 0))
                {
                    continue;
                }

                dp[i + 1].push_back(j);
            }            
        }

        vector<string> storage;

        assemble(dp, s, storage, s.length());

        return storage;
    }

    void assemble(const vector<vector<int>> &dp, const string &nums, vector<string> &storage, int start)
    {
        static vector<int> points;

        if(points.size() >= 5 && start != -1)
        {
            return;
        }
        if(points.size() == 5 && start == -1)
        {
            string result;
            result.reserve(nums.length() + 3);

            int point = 3;
            for(int i = 0; i < nums.length(); ++i)
            {
                if(0 <= point && points[point] == i)
                {
                    result.push_back('.');
                    --point;
                }
                result.push_back(nums[i]);
            }

            storage.emplace_back(result);
        }
        if(start == -1)
        {
            return;
        }

        points.push_back(start);

        for(int prev : dp[start])
        {
            assemble(dp, nums, storage, prev);
        }

        points.pop_back();
    }
};
