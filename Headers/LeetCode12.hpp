#pragma once

#include "DataStructures.hpp"
#include "Containers/Static/BitSet.hpp"
#include "Containers/Dynamic/PriorityFrequencies.hpp"
#include "Containers/Static/StaticStack.hpp"

#define MODULO 1000000007
#define ll long long
#define pii pair<int,int>
#define uchar unsigned char

class Solution {
public:

    //1915. Number of Wonderful Substrings
    
#define MAX_SIZE 1024
    long long wonderfulSubstrings(string word) {
        static int memo[MAX_SIZE];
        static constexpr int MEM_LEN = MAX_SIZE << 2;
        
        memset(memo, 0, MEM_LEN);
        memo[0] = 1;

        long long result = 0;
        const int SIZE = static_cast<int>(word.length());
        int mask = 0;
        for(int i = 0; i < SIZE; ++i){
            mask ^= 1 << (word[i] - 'a');
            
            result += static_cast<long long>(memo[mask]++);

            int submask = 1;
            while(submask != 1024){
                int val = mask ^ submask;
                result += static_cast<long long>(memo[val]);
                submask *= 2;
            }
        }

        return result;
    }
#undef MAX_SIZE

    //718. Maximum Length of Repeated Subarray
#define MAX_SIZE 1000
    int findLength(vector<int> nums1, vector<int> nums2) {
        static int dp[MAX_SIZE];
        
        const int TOP = static_cast<int>(nums1.size());
        const int BOT = static_cast<int>(nums2.size());

        for(int i = 0; i < TOP;++i){
            for(int j = BOT - 1; 0 <= j; --j){
                if(nums1[i] != nums2[j])continue;
                if(i * j != 0 && nums1[i - 1] == nums2[j - 1])dp[j] = max(dp[j], dp[j - 1] + 1);
                else dp[j] = max(dp[j], 1);
            }
        }
        
        int result = 0;
        for(int i = 0; i < BOT; ++i){
            result = max(result, dp[i]);
        }

        return result;
    }
#undef MAX_SIZE


    //424. Longest Repeating Character Replacement

    int characterReplacement(string s, int k) {
        PriorityFrequencies<char> pf;

        int l = 0, r = 0;
        int result = 0;
        const int LEN = static_cast<int>(s.length());
        while(r < LEN){
            pf.push(s[r++], 1);
            int f = pf.top().first;
            while(r - l - f > k){
                pf.push(s[l++], -1);
                f = pf.top().first;
            }
            result = max(result, r - l);
        }

        return result;
    }

    //957. Prison Cells After N Days

    vector<int> prisonAfterNDays(vector<int> cells, int n) {
        int num = 0;
        for(int i = 0; i < 8; ++i){
            num = num * 2 + cells[i];
        }

        n = (n - 1) % 14 + 1;
        while(n-- > 0){
            int next = 0;
            for(int i = 1; i < 7; ++i){
                if(bool(num & (1 << i - 1)) == bool(num & (1 << i + 1))){
                    next |= 1 << i;
                }
            }
            num = next;
        }

        for(int i = 0; i < 8; ++i){
            cells[7 - i] = num & 1;
            num /= 2;
        }

        return cells;
    }

    //786. K-th Smallest Prime Fraction
    class FractionComparer{
        public:
        bool operator() (const pair<int,int>& l,const pair<int,int>& r) const {
            return l.first * r.second > r.first * l.second; 
        }
    };


    vector<int> kthSmallestPrimeFraction(vector<int> arr, int k) {
        using Type = pair<int,int>;

        static const int MAX_SIZE = 30000;
        static int valToIndex[MAX_SIZE + 1];
        priority_queue<Type, vector<Type>, FractionComparer> queue;

        const int SIZE = static_cast<int>(arr.size());
        for(int i = 0; i < SIZE; ++i){
            valToIndex[arr[i]] = i;
            queue.push(make_pair(arr[i], arr.back()));
        }

        while(--k != 0){
            auto top = queue.top(); queue.pop();
            if(top.first == top.second)continue;
            queue.push(make_pair(top.first, arr[valToIndex[top.second] - 1]));
        }

        return {
            queue.top().first, 
            queue.top().second
        };
    }

    //3137. Minimum Number of Operations to Make Word K-Periodic
    int minimumOperationsToMakeKPeriodic(string word, int k) {
        unordered_map<string, int> count;

        const int LEN = static_cast<int>(word.length());
        int max_count = 0;
        for(int i = 0; i + k <= LEN; i += k){
            max_count = max(max_count, ++count[word.substr(i, k)]);
        }

        return LEN / k - max_count;
    }

};

#undef MODULO
#undef ll
#undef pii
#undef uchar

//static const auto speedup = []() {
//    std::ios::sync_with_stdio(false);
//    std::cin.tie(nullptr);
//    std::cout.tie(nullptr);
//     return 0;
//}();