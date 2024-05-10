#pragma once
#include "./DataStructures.hpp"

#define MODULO 1000000007
#define ll long long

class Solution {
public:
    //1986. Minimum Number of Work Sessions to Finish the Tasks
#define MAX_VARIATIONS 16384
#define MAX_SIZE 14

    int minSessions(vector<int> tasks, int sessionTime) {
        static char memo[MAX_SIZE + 1][MAX_VARIATIONS];

        memset(memo[0], 255, MAX_VARIATIONS);
        memo[0][0] = 200;

        const int SIZE = tasks.size();
        const int ALL_TASKS = 1 << SIZE;

        for (int i = 0; i < SIZE; ++i) {
            memset(memo[i + 1], 255, ALL_TASKS);
            for (int j = 0; j < ALL_TASKS; ++j) {
                if (memo[i][j] == 255)continue;
                if (j + 1 == ALL_TASKS)return i;

                for (int k = 0; k < SIZE; ++k) {
                    if ((j & (1 << k)) != 0)continue;
                    if (memo[i][j] + tasks[k] > sessionTime) {
                        memo[i + 1][j | (1 << k)] = min(static_cast<char>(tasks[k]), memo[i + 1][j | (1 << k)]);
                    }
                    else {
                        memo[i][j | (1 << k)] = min(static_cast<char>(tasks[k] + memo[i][j]), memo[i][j | (1 << k)]);
                    }
                }
            }
        }

        return SIZE;
    }

#undef MAX_VARIATIONS
#undef MAX_SIZE

    //
#define MAX_VARIATIONS 32768

    bool makesquare(vector<int> matchsticks) {
        static int memo[5][MAX_VARIATIONS];
        memset(memo[0], 127, MAX_VARIATIONS * 4);

        const int SIZE = matchsticks.size();
        const int ALL_MATCHSTICKS = (1 << SIZE) - 1;
        int sum = accumulate(matchsticks.begin(), matchsticks.end(), 0);

        if (sum % 4 != 0)return false;

        memo[0][0] = sum / 4;

        for (int i = 0; i < 4; ++i) {
            memset(memo[i + 1], 127, MAX_VARIATIONS * 4);

            for (int v = 0; v < ALL_MATCHSTICKS; ++v) {
                if (memo[i][v] == 2139062143)continue;

                for (int m = 0; m < SIZE; ++m) {
                    if ((v & (1 << m)) != 0)continue;
                    if (memo[i][v] == (sum / 4)) {
                        if (i == 3)return true;
                        memo[i + 1][v | (1 << m)] = matchsticks[m];
                    }
                    else if (memo[i][v] + matchsticks[m] <= (sum / 4)) {
                        memo[i][v | (1 << m)] = memo[i][v] + matchsticks[m];
                    }
                }
            }
        }

        return false;
    }
#undef MAX_VARIATIONS

    //1291. Sequential Digits
    int createSequential(int fromDigit, int len) {
        int num = 0;

        while (len-- > 0) {
            num = num * 10 + fromDigit++;
        }

        return num;
    }

    vector<int> sequentialDigits(int low, int high) {
        vector<int> result;

        int current_len = log10(low) + 1;
        const int MAX_LEN = log10(high) + 1;
        int digit = low / pow(10, current_len - 1);

        while (current_len <= MAX_LEN) {
            const int LIMIT = 11 - current_len;
            while (digit < LIMIT) {
                int temp = createSequential(digit, current_len);
                if (low <= temp && temp <= high) {
                    result.push_back(temp);
                }
                ++digit;
            }
            ++current_len;
            digit = 1;
        }

        return result;
    }

    //1043. Partition Array for Maximum Sum
#define MAX_N 500
    int maxSumAfterPartitioning(vector<int> arr, int k) {
        static int dp[MAX_N + 1] = { 0 };

        const int SIZE = arr.size();
        memset(dp + 1, 0, SIZE * 4);
        
        for (int i = 0; i < SIZE; ++i) {
            int local_max = 0;
            for (int j = 0; j < k; ++j) {
                if (i - j < 0)break;
                local_max = max(local_max, arr[i - j]);
                dp[i + 1] = max(dp[i + 1], dp[i - j] + local_max * (j + 1));
            }
            show_array(dp, SIZE + 1);
        }

        return dp[SIZE];
    }
#undef MAX_N
    
    //2787. Ways to Express an Integer as Sum of Powers
#define MAX_N 300
#define MODULO 1000000007

    int numberOfWays(int n, int x) {
        //use 2d dp for unique check

        static int dp[MAX_N + 1][MAX_N + 1] = { {1} };

        const double LIMIT = static_cast<double>(n);

        return dp[n][n];
    }

    //647. Palindromic Substrings
#define MAX_LEN 1000
    union BoolPair {
        bool pair[2];
        uint16_t solid;
    };

    int countSubstrings(string s) {
        static BoolPair dp[MAX_LEN][MAX_LEN] = { {{513}} };
        const int LEN = s.length();

        int result = LEN;
        for (int i = 1; i < LEN; ++i) {
            memset(dp[i], 0, LEN * 2);
            dp[i][0] = { true, true };

            for (int j = 0; j <= i; ++j) {
                if (dp[i - 1][j].pair[0] == false)continue;
                if (dp[i - 1][j].pair[1] && s[i] == s[i - 1]) {
                    dp[i][j + 1] = { true,true };
                    ++result;
                }
                else if (i - j - 2 >= 0 && s[i - j - 2] == s[i]) {
                    dp[i][j + 2] = { true,false };
                    ++result;
                }
            }
        }

        return result;
    }

#undef MAX_LEN

//526. Beautiful Arrangement
#define MAX_N 15
    int countArrangement(int n) {
        static int dp[MAX_N][MAX_N];
        static void (*init)() = []() 
        {
            for (int i = 0; i < MAX_N; ++i)dp[0][i] = 1;
        };
        static bool inited = false;
        
        if (inited == false) {
            init();
            inited = true;
        }

        for (int i = 1; i < n; ++i) {
            memset(dp[i], 0, n << 2);

            for (int j = 0; j < n; ++j) {
                if ((j + 1) % (i + 1) != 0 && (i + 1) % (j + 1) != 0)continue;
                for (int p = 0; p < n; ++p) {
                    //if(p)
                }
            }
        }

        int result = 0;
        for (int i = 0; i < n; ++i) {
            result += dp[n - 1][i];
        }

        return result;
    }
#undef MAX_N

//1642. Furthest Building You Can Reach
    int furthestBuilding(vector<int>& heights, int bricks, int ladders) {
        priority_queue<int, vector<int>, greater<int>> highest;

        const int SIZE = heights.size();
        int i = 1;

        int sum = 0;
        while (i < SIZE) {
            if (heights[i] - heights[i - 1] <= 0)continue;
            highest.push(heights[i] - heights[i - 1]);
            if (highest.size() > ladders) {
                int top = highest.top(); highest.pop();
                if (sum + top > bricks)break;
                sum += top;
            }
            ++i;
        }

        return i - 1;
    }

//2402. Meeting Rooms III
#define MAX_MEETINGS 100000
#define MAX_N 100
#define pli pair<long long,int>

    int mostBooked(int n, vector<vector<int>> meetings) {

        const int SIZE_MEETINGS = meetings.size();
        sort(meetings.begin(), meetings.end(), [](const auto& l, const auto& r) {
            return l[0] < r[0];
        });

        static int times_used[MAX_N];
        static int indices[MAX_N];
        for (int i = 0; i < n; ++i) {
            indices[i] = i;
            times_used[i] = 0;
        }

        //first -> end time, second -> room number
        priority_queue<pli, vector<pli>, greater<pli>> queue;
        priority_queue<int, vector<int>, greater<int>> rooms;

        for (int i = 0; i < n; ++i) {
            rooms.push(i);
        }

        //main logic
        long long current_time = 0;
        int i = 0;
        while (i < SIZE_MEETINGS) {
            if (
                !queue.empty() &&
                (meetings[i][0] >= queue.top().first
                ||
                current_time >= queue.top().first)
            ) {
                current_time = max(current_time, queue.top().first);
                while (
                    !queue.empty() &&
                    (meetings[i][0] >= queue.top().first
                        ||
                    current_time >= queue.top().first)
                ) {
                    rooms.push(queue.top().second);
                    queue.pop();
                }
            }
            if (!rooms.empty()) {
                int top = rooms.top(); rooms.pop();
                queue.push({
                    current_time > meetings[i][0]
                        ? current_time + meetings[i][1] - meetings[i][0]
                        : meetings[i][1]
                    , top
                    });
                ++i;
                ++times_used[top];
            }
            else current_time = queue.top().first;
        }

        sort(indices, indices + n, [&](int l, int r) {
            return times_used[l] == times_used[r]
                ? l < r
                : times_used[l] > times_used[r]
                ;
            });
        return indices[0];
    }
#undef MAX_MEETINGS
#undef MAX_N
#undef pii

    //787. Cheapest Flights Within K Stops
#define MAX_N 100
#define pii pair<int,pair<int,int>>
#define SV 2139062143

    int findCheapestPrice(int n, vector<vector<int>> flights, int src, int dst, int k) {
        static int dp[MAX_N][MAX_N + 1];
        for (int i = 0; i < n; ++i) {
            memset(dp[i], 127, (MAX_N+1) << 2);
        }

        vector<vector<pair<int, int>>> connections(n);
        for (const auto& flight : flights) {
            connections[flight[0]].push_back({ flight[1],flight[2] });
        }

        priority_queue<pii, vector<pii>, greater<pii>> queue;
        queue.push({ 0, {src, -1} });

        while (!queue.empty()) {
            auto [v, d] = queue.top(); queue.pop();
            if (dp[d.first][d.second + 1] <= v)continue;
            dp[d.first][d.second + 1] = v;
            if (d.second + 1 > k)continue;
            for (const auto& [to, cost] : connections[d.first]) {
                queue.push({ cost + v,{to, d.second + 1} });
            }
        }
        
        int res = SV;
        k += 1;
        for (int i = 0; i <= k; ++i) {
            res = min(dp[dst][i], res);
        }
        return res == SV ? -1 : res;
    }
#undef MAX_N
#undef pii
#undef SV

//1452. People Whose List of Favorite Companies Is Not a Subset of Another List
#define MAX_LEN 100

    vector<int> peopleIndexes(vector<vector<string>> favoriteCompanies) {
        static int indices[MAX_LEN];
        const int LEN = favoriteCompanies.size();

        for (int i = 0; i < LEN; ++i) {
            indices[i] = i;
        }

        sort(
            indices,
            indices + LEN,
            [&](int l, int r) {
                return favoriteCompanies[l].size() > favoriteCompanies[r].size();
            }
        );

        unordered_map<string, unordered_map<int, bool>> data;

        vector<int> result;

        static int others[MAX_LEN];
        for (int i = 0; i < LEN; ++i) {

            memset(others, 0, LEN << 2);
            int max_common = 0;

            for (const auto& str : favoriteCompanies[indices[i]]) {
                if (data.find(str) == data.end()) {
                    data.insert({ str, {} });
                }
                for (const auto& p : data[str]) {
                    max_common = max(max_common, ++others[p.first]);
                }
                data[str][indices[i]] = true;
            }

            if (max_common < favoriteCompanies[indices[i]].size())result.push_back(indices[i]);
        }

        sort(result.begin(), result.end());

        return result;
    }

//457. Circular Array Loop
    int move(const int SIZE, int current) {
        int k = current < 0 ? -1 : 1;
        current = (abs(current) % SIZE) * k;
        return current + (current < 0 ? SIZE : 0);
    }

    bool circularArrayLoop(vector<int> nums) {
        const int SIZE = nums.size();

        for (int i = 0; i < SIZE; ++i) {
            if (nums[i] == 0)continue;
            int head = i;
            int current = move(SIZE, i + nums[i]);
            nums[i] = 0;
            while (current != head || nums[current] != 0) {
                int prev = current;
                current = move(SIZE, current + nums[current]);
                nums[prev] = 0;
            }
            if (current == head)return true;
        }

        return false;
    }

    //1653. Minimum Deletions to Make String Balanced

#define MAX_SIZE 100000
    int minimumDeletions(string s) {
        static int dp[MAX_SIZE];
        const int LEN = s.length();

        memset(dp, 0, LEN << 2);

        int result = LEN;
        int l = 0, r = LEN - 1;
        int suml = 0, sumr = 0;
        while (0 <= r) {
            dp[l] += suml;
            dp[r] += sumr;
            suml += s[l] == 'b';
            sumr += s[r] == 'a';
            if (l >= r) {
                result = min(result, min(dp[l], dp[r]));
            }
            ++l;
            --r;
        }

        return result;
    }
#undef MAX_SIZE

    //123. Best Time to Buy and Sell Stock III
#define MAX_SIZE 100000

    int maxProfit(vector<int>& prices) {
        const int SIZE = static_cast<int>(prices.size());

        static int dp[MAX_SIZE];

        dp[0] = 0;
        int least = prices[0];
        for (int i = 1; i < SIZE; ++i) {
            dp[i] = 0;
            dp[i] = max(dp[i - 1], prices[i] - least);
            least = min(least, prices[i]);
        }

        int most = prices.back();
        for (int i = SIZE - 2; 0 <= i; --i) {
            dp[i] = max(dp[i + 1], dp[i] + max(0, most - prices[i]));
            most = max(most, prices[i]);
        }

        return dp[0];
    }

#undef MAX_SIZE

    //3043. Find the Length of the Longest Common Prefix
#define MASK_X 256
#define MASK_Y 1
#define MASK_XY 257

    struct TrieNode {
        TrieNode* next[10];
        uint16_t types;
        bool leaf;
    };

    int insert(TrieNode* root, int num, uint16_t mask) {
        static stack<int> data;
        while (num > 0) {
            data.push(num % 10);
            num /= 10;
        }

        bool same = true;
        int len = 0;

        while (!data.empty()) {
            int top = data.top(); data.pop();
            if (root->next[top] == nullptr) {
                root->next[top] = new TrieNode();
            }
            root = root->next[top];
            root->types |= mask;
            if (same && root->types == MASK_XY) {
                ++len;
            }
            else same = false;
        }
        root->types |= mask;
        root->leaf = true;

        return len;
    }

    int longestCommonPrefix(vector<int>& arr1, vector<int>& arr2) {
        TrieNode root;
        
        for (int x : arr1)insert(&root, x, MASK_X);
        int result = 0;
        for (int y : arr2)result = max(result, insert(&root, y, MASK_Y));

        return result;
    }
#undef MASK_X
#undef MASK_Y
#undef MASKXY

    //2709. Greatest Common Divisor Traversal
#define MAX_VALUES 100000
#define MAX_LEN    100000

    pair<vector<int>*, vector<bool>*> getAllPrimes(const int upTo) {
        static vector<bool> local_primes(MAX_VALUES + 1, true);
        static vector<int> primes = {};
        static int max_value = 1;

        while (++max_value <= upTo) {
            if (!local_primes[max_value])continue;
            primes.push_back(max_value);
            for (int i = max_value; i < MAX_VALUES; i += max_value) {
                local_primes[i] = false;
            }
        }

        return make_pair(& primes, &local_primes);
    }

    int find(int* unions, int index) {
        if (index == -1)return -1;
        while (index != unions[index]) {
            index = unions[index];
        }
        return index;
    }

    bool canTraverseAllPairs(vector<int> nums) {
        static auto primes = getAllPrimes(MAX_VALUES);
        static int unions[MAX_LEN];
        static int prime_unions[MAX_VALUES + 1];

        //init
        const int SIZE = nums.size();
        for (int i = 0; i < SIZE; ++i) {
            unions[i] = i;
        }
        memset(prime_unions, -1, MAX_VALUES << 2);

        //main logic
        unordered_map<int, int> memo;

        for (int i = 0; i < SIZE; ++i) {
            for (int prime : *primes.first) {
                if (nums[i] == 1)break;
                if ((*primes.second)[nums[i]]) {
                    int g = find(unions, i);
                    if (prime_unions[nums[i]] == -1)prime_unions[nums[i]] = g;
                    //else unions[i] = prime_unions;
                    break;
                }
                if (memo.count(nums[i]) != 0) {
                    unions[i] = memo[nums[i]];
                    break;
                }
                if (nums[i] % prime != 0)continue;
                int g1 = find(unions, prime_unions[prime]);
                int g2 = find(unions, i);
                if (g1 == -1)prime_unions[prime] = g2;
                else unions[g1] = g2;
                memo[nums[i]] = g2;
                while (nums[i] % prime == 0)nums[i] /= prime;
            }
        }

        int group = find(unions, 0);
        for (int i = 1; i < SIZE; ++i) {
            if (find(unions, i) != group)return false;
        }

        return true;
    }

#undef MAX_VALUES
#undef MAX_LEN

    //3040. Maximum Number of Operations With the Same Score II
#define MAX_LEN 2000
    int maxOperations(vector<int>& nums) {
        static int dp[MAX_LEN];
        const int SIZE = nums.size();

        memset(dp, 0, SIZE << 2);
        int result = 0;

        const int F = nums[0] + nums[SIZE - 1];
        const int S = nums[0] + nums[1];
        const int T = nums[SIZE - 1] + nums[SIZE - 2];
        
        for (int i = 0; i < SIZE; ++i) {

        }

        return result;
    }

    //974. Subarray Sums Divisible by K
#define MAX_RANGE 20002

    int subarraysDivByK(vector<int> nums, int k) {
        unordered_map<int, int> lefts;

        lefts[0] = 1;

        long long sum = 0;
        int result = 0;
        const int SIZE = nums.size();
        for (int i = 0; i < SIZE; ++i) {
            sum += nums[i];
            result += lefts[sum % k]++;
            if (sum % k != 0)result += lefts[-(sum % k)];
        }

        return result;
    }

    //1824. Minimum Sideway Jumps
    int minSideJumps(vector<int>& obstacles) {
        int dp[3] = { 1,0,1 };
 
        const int SIZE = obstacles.size() - 1;
        for (int i = 0; i < SIZE; ++i) {
            int next[3] = { 1000000, 1000000, 1000000 };

            for (int c = 0; c < 3; ++c) {
                if (obstacles[i] == c + 1)continue;
                for (int n = 0; n < 3; ++n) {
                    if (obstacles[i + 1] == n + 1)continue;
                    next[n] = min(
                        next[n],
                        dp[c] +
                        static_cast<bool>(n != c) +
                        static_cast<bool>(obstacles[i] == n + 1)
                    );
                }
            }
            memcpy(dp, next, 12);
        }

        return min(dp[0], min(dp[1], dp[2]));
    }

    //828. Count Unique Characters of All Substrings of a Given String

#define MAX_LEN 100001
    int uniqueLetterString(string s) {
        //first -> previous previous
        //second -> previous
        static pair<int, int> pp[26];
        memset(pp, -1, 208);

        static int DP[MAX_LEN] = { 0 };
        int* dp = &DP[1];

        int result = 0;
        const int LEN = s.length();
        for (int i = 0; i < LEN; ++i) {
            int x = s[i] - 'A';
            dp[i] = dp[i - 1] - (pp[x].first - pp[x].second) + i - pp[x].first;
            pp[x].second = pp[x].first;
            pp[x].first = i;
            result += dp[i];
        }

        show_array(dp, LEN);

        return result;
    }
#undef MAX_LEN

    //491. Non-decreasing Subsequences

#define MAX_POSSIBILITIES 32752
#define MEM_LEN 262016
#define MAX_LEN 15
    inline uint64_t EF(const int num) {
        return pow(num + 111, 5);
    }

    vector<vector<int>> findSubsequences(vector<int> nums) {
        static uint64_t keys[MAX_POSSIBILITIES];
        static uint64_t encrypted[MAX_LEN];
        memset(keys, 0, MEM_LEN);

        unordered_map<uint64_t, bool> exists;
        static pair<int, int> memo[MAX_POSSIBILITIES];

        for (int i = 0; i < nums.size(); ++i) {
            encrypted[i] = EF(nums[i]);
        }

        int mSize = 0;
        for (int i = 1; i < nums.size(); ++i) {
            const int size = mSize;
            
            for (int j = 0; j < size; ++j) {
                if (memo[j].second <= nums[i] && exists.count(keys[j] + encrypted[i]) == 0) {
                    keys[mSize] = keys[j] + encrypted[i];
                    exists.insert({ keys[j] + encrypted[i], true });

                    memo[mSize++] = {memo[j].first | (1 << i), nums[i]};
                }
            }
            for (int j = 0; j < i; ++j) {
                if (nums[j] <= nums[i] && exists.count(encrypted[j] + encrypted[i]) == 0) {
                    keys[mSize] = encrypted[j] + encrypted[i];
                    exists.insert({ encrypted[j] + encrypted[i], true });
                    memo[mSize++] = { (1 << j) | (1 << i), nums[i]};
                }
            }
        }


        //slower, than prev version
        vector<vector<int>> result(mSize);
        for (int i = 0; i < mSize; ++i) {
            int pos = 0;
            while (memo[i].first > 0) {
                if (memo[i].first & 1 != 0) {
                    result[i].push_back(nums[pos]);
                }
                memo[i].first /= 2;
                ++pos;
            }
        }

        return result;
    }
#undef MAX_POSSIBILITIES
#undef MEM_LEN
#undef MAX_LEN

//2767. Partition String Into Minimum Beautiful Substrings
#define MAX_LEN 15
#define INVALID 2139062143

    int minimumBeautifulSubstrings(string s) {
        static const unordered_map<int, bool> powers{ {1,1}, {5,1},{25,1},{125,1},{625,1},{3125,1},{15625,1}};
        static int STATIC_DP[MAX_LEN + 1];
        int* dp = &STATIC_DP[1];

        const int LEN = s.length();
        memset(dp, 127, LEN << 2);
        reverse(s.begin(), s.end());

        for (int i = 0; i < LEN; ++i) {
            if (dp[i - 1] == INVALID)continue;

            int sum = 0;
            int power = 1;
            for (int j = i; j < LEN; ++j) {
                sum += (s[j] - '0') * power;
                power *= 2;
                if (s[j] == '0')continue;
                if (powers.count(sum)) {
                    dp[j] = min(dp[j], dp[i - 1] + 1);
                }
            }
        }

        return STATIC_DP[LEN] == INVALID
            ? -1
            : STATIC_DP[LEN]
        ;
    }
#undef MAX_LEN
#undef INVALID

    //659. Split Array into Consecutive Subsequences
#define pii pair<int, int>

    bool isPossible(vector<int>& nums) {
        priority_queue<pii, vector<pii>, less<pii>> queue;
    }

    //57. Insert Interval
    vector<vector<int>> insert(vector<vector<int>>& intervals, vector<int>& newInterval) {
        //edge cases
        if (intervals.size() == 0)return { newInterval };
        if (newInterval[1] < intervals[0][0]) {
            intervals.emplace(intervals.begin(), std::move(newInterval));
            return intervals;
        }
        if (intervals.back()[1] < newInterval[0]) {
            intervals.emplace_back(std::move(newInterval));
            return intervals;
        }

        swap(newInterval[0], newInterval[1]);
        int bottom = lower_bound(
            intervals.begin(),
            intervals.end(),
            newInterval,
            [](const auto& l, const auto& r) {
                return l[1] < r[1];
            }
        ) - intervals.begin();
        int top = lower_bound(
            intervals.begin() + bottom,
            intervals.end(),
            newInterval,
            [](const auto& l, const auto& r) {
                return l[0] < r[0];
            }
        ) - intervals.begin();
        swap(newInterval[0], newInterval[1]);

        const int SIZE = static_cast<int>(intervals.size());
        if (top == SIZE || intervals[top][0] > newInterval[1]) {
            top = max(0, top - 1);
        }

        //case if interval does not intersect with other intervals at all
        if (top < bottom) {
            intervals.insert(intervals.begin() + bottom, newInterval);
            return intervals;
        }

        //new interval insert
        intervals[bottom][0] = min(newInterval[0], intervals[bottom][0]);
        intervals[bottom][1] = max(newInterval[1], intervals[top][1]);
        //data shift
        intervals.erase(intervals.begin() + bottom + 1, intervals.begin() + top + 1);

        return intervals;
    }

    //956. Tallest Billboard

#define MAX_RANGE 10001
    int tallestBillboard(vector<int>& rods) {
        //index -> differenece in height between rods,
        //value of index -> maximum height of the tallest rod
        static int F[MAX_RANGE], S[MAX_RANGE];
        static constexpr int ZERO_INDEX = MAX_RANGE / 2;
        static constexpr int MEM_LEN = MAX_RANGE << 2;
        memset(F, 0, MEM_LEN);

        int* f = &F[ZERO_INDEX], * s = &S[ZERO_INDEX];

        int sum = 0;
        for (int rod : rods) {
            memset(&s[-ZERO_INDEX], 0, MEM_LEN);

            for (int v = -sum; v <= sum; ++v) {
                if (f[v] == 0 && v != 0)continue;
                s[v] = max(f[v], s[v]);
                s[v - rod] = max(s[v - rod], f[v]);
                s[v + rod] = max(s[v + rod], f[v] + rod);
            }
            sum += rod;
            swap(f, s);
        }

        return f[0];
    }
#undef MAX_RANGE


    //675. Cut Off Trees for Golf Event
#define pt pair<int, pair<int, int>>

    int setPriority(vector<vector<int>>& forest) {
        static priority_queue<pt, vector<pt>, greater<pt>> queue;
        
        const int ROWS = forest.size();
        const int COLUMNS = forest[0].size();

        for (int r = 0; r < ROWS; ++r) {
            for (int c = 0; c < COLUMNS; ++c) {
                queue.push({ forest[r][c], {r,c} });
            }
        }

        int count = 1;
        while (queue.size() != 0) {
            auto top = queue.top(); queue.pop();
            if (top.first < 2) continue;
            forest[top.second.first][top.second.second] = ++count;
        }
        
        return count - 2;
    }

    int cutOffTree(vector<vector<int>> forest) {
        static const int directions[4][2] = { {-1,0},{1,0},{0, -1},{0, 1} };

        if (forest[0][0] == 0)return -1;

        const int ROWS = forest.size();
        const int COLUMNS = forest[0].size();

        static deque<pair<int, int>> positions;
        map<int, int> trees;
        const int COUNT = setPriority(forest);
        int level = 1;

        trees[forest[0][0]] = 0;
        positions.push_back({ 0,0 });

        while (positions.size() != 0) {
            int size = positions.size();

            set<int> next_trees;
            while (size-- > 0) {
                auto top = positions.front(); positions.pop_front();

                for (int i = 0; i < 4; ++i) {
                    int nr = top.first + directions[i][0];
                    int nc = top.second + directions[i][1];
                    if (nr < 0 || nr >= ROWS || nc < 0 || nc >= COLUMNS)continue;
                    if (forest[nr][nc] < 2) continue;
                    if (trees.count(forest[nr][nc]))continue;
                    if (next_trees.count(forest[nr][nc]))continue;
                    next_trees.insert(forest[nr][nc]);
                    positions.push_back({ nr, nc });
                }
            }

            for (int nt : next_trees) {
                trees.insert({ nt, level });
            }

            level += 1;
        }

        show_map(trees);

        int result = 0;
        int prev = 0;
        for (const auto& [_, l] : trees) {
            result += abs(prev - l);
            prev = l;
        }

        return result;
    }

};



#undef MODULO
#undef ll

//static const auto speedup = []() {
//    std::ios::sync_with_stdio(false);
//    std::cin.tie(nullptr);
//    std::cout.tie(nullptr);
//     return 0;
//}();