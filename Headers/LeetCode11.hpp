#pragma once

#include "DataStructures.hpp"
#include "Containers/StaticSized/BitSet.hpp"
#include "Containers/StaticSized/StaticStack.hpp"

#define MODULO 1000000007
#define ll long long
#define pii pair<int,int>
#define uchar unsigned char

class Solution {
public:

//41. First Missing Positive

    int firstMissingPositive(vector<int> nums) {
        const int SIZE = nums.size();

        for(int i = 0; i < SIZE; ++i){
            if(nums[i] <= 0 || nums[i] > SIZE)nums[i] = SIZE + 1;
        }

        int max_possible = -1;
        for(int i = 0; i < SIZE; ++i){
            int index = abs(nums[i]) - 1;
            if(index == SIZE){
                continue;
            }
            max_possible = max(max_possible, index);
            if(nums[index] > 0)nums[index] *= -1;
        }

        for(int i = 0; i < SIZE; ++i){
            if(nums[i] > 0)return i + 1;
        }

        return max_possible + 2;
    }

    //2709. Greatest Common Divisor Traversal
#define MAX_SIZE 100001
#define MAX_VALUE 100001

    inline int find(const int *unions, int index){
        while(index != unions[index]){
            index = unions[index];
        }
        return index;
    }

    inline void connect(int *unions, int f, int s){
        int g1 = find(unions, f);
        int g2 = find(unions, s);
        if(g1 > g2)swap(g2, g1);
        unions[g2] = g1;
    }

    bool canTraverseAllPairs(vector<int> nums) {
        static int unions[MAX_SIZE];
        static vector<int> indecies(MAX_VALUE, -1);
        static BitSet<MAX_VALUE> isPrime;
        static int max_value;

        isPrime.reset();
        memset(&indecies[0], -1, (max_value + 1) << 2);
        max_value = 0;

        const int SIZE = static_cast<int>(nums.size());

        //special case!
        if(SIZE == 1)return true;

        for (int i = 0; i < SIZE; ++i) {
            if (nums[i] == 1)return false;
            unions[i] = i;
            max_value = max(max_value, nums[i]);
            indecies[nums[i]] = i;
        }

        for (int i = 2; i <= max_value; ++i) {
            if (isPrime.getBit(i))continue;
            int prev = indecies[i];
            for (int j = i * 2; j <= max_value; j += i) {
                isPrime.set(j);
                if (indecies[j] < 0)continue;
                if (prev < 0)prev = indecies[j];
                else connect(unions, prev, indecies[j]);
            }
        }

        const int GROUP = find(unions, indecies[max_value]);
        for (int i = 0; i < SIZE; ++i) {
            if(find(unions, (indecies[nums[i]])) != GROUP)return false;
        }

        return true;
    }

#undef MAX_VALUE
#undef MAX_SIZE

    tree_node::TreeNode* recoverFromPreorder(string traversal) {
        stack<tree_node::TreeNode*> stack;
        static string num;

        const int LEN = traversal.length();
        int i = 0;
        while(i < LEN){
            int j = i;
            while(traversal[j] == '-'){
                ++j;
            }
            num.clear();
            while(j < LEN && traversal[j] != '-'){
                num += traversal[j++];
            }

            const int dashes = j - i + 1;
            const int value = stoi(num);
            while(stack.size() >= dashes){
                stack.pop();
            }
            if(stack.size() == 0){
                stack.push(new tree_node::TreeNode(value));
            }
            else if(stack.top()->left == 0){
                stack.top()->left = new tree_node::TreeNode(value);
                stack.push(stack.top()->left);
            }
            else {
                stack.top()->right = new tree_node::TreeNode(value);
                stack.push(stack.top()->right);
            }

            i = j;
        }

        return nullptr;
    }

//480. Sliding Window Median

#define pii pair<int,int>

    

    vector<double> medianSlidingWindow(vector<int>& nums, int k) {
        const int SIZE = static_cast<int>(nums.size());
        vector<double> result;
        result.reserve(SIZE - k + 1);

        priority_queue<pii,vector<pii>,less<pii>> left;
        priority_queue<pii,vector<pii>,greater<pii>> right;

        //keep diff in size of queues based on 

        int i = 0;
        while(i < SIZE){

            

            if(i+1 >= k){

            }
        }

        return result;
    }


//1015. Smallest Integer Divisible by K
#define MAX_SIZE 100000
    int disassembleInt(int *dst, int k){
        int resultSize = 0;
        while(k > 0){
            dst[resultSize++] = k % 10;
            k /= 10;
        }    
        return resultSize;
    }
    
    int smallestRepunitDivByK(int k) {
        static const int preMultiplied[10][10] = {
            {},
            {1,1,9,8,7,6,5,4,3,2},
            {},
            {7,1,3,6,9,2,5,8,1,4},
            {},{},{},
            {3,1,7,4,1,8,5,2,9,6},
            {},
            {9,1,1,2,3,4,5,6,7,8}
        };
        
        const int FIRST = k % 10; 
        static const int PRE_MASK = 0b1010001010;
        if((PRE_MASK & (1 << FIRST)) == 0)return -1;
        
        static int digits[6];
        int size = disassembleInt(digits, k);
        
        static int sections[MAX_SIZE];
        static int i = 0;
        memset(sections, 0, (i + 1) << 2);
        i = 0;

        for(;sections[i] != 0 || i == 0; ++i){
            if(sections[i] == 1)continue;
            
            const int MULTIPLIER = preMultiplied[FIRST][sections[i]];
            for(int j = 0; j < size; ++j){
                sections[i + j] += digits[j] * MULTIPLIER;
                sections[i + j + 1] += sections[i + j]/10;
                sections[i + j] %= 10;
            }
        }

        return i;
    }
#undef MAX_SIZE

    //2962. Count Subarrays Where Max Element Appears at Least K Times
    
#define MAX_SIZE 100000
    long long countSubarrays(vector<int> nums, int k) {
        static int indices[MAX_SIZE];
        int size = 0;

        ll result = 0;
        int current_max = -1;
        const int SIZE = static_cast<int>(nums.size());
        for(int i = 0; i < SIZE; ++i){
            if(current_max < nums[i]){
                current_max = nums[i];
                size = 0;
                result = 0;
            }

            if(current_max != nums[i])continue;

            indices[size++] = i;
            if(size > k){
                result += static_cast<ll>(indices[size - 1] - indices[size - 2]) * static_cast<ll>(indices[size - k - 1] + 1);
            }
        }
        if(size >= k){
            result += static_cast<ll>(SIZE - indices[size - 1]) * static_cast<ll>(indices[size - k] + 1);
        }

        return result;
    }
#undef MAX_SIZE

    //992. Subarrays with K Different Integers
#define MAX_VALUES 20001
    int sum(int n){
        return (n * n + n)/2;
    }

    int helper(const vector<int>& nums, const int K) {
        static int freqs[MAX_VALUES];
        static int prev_max = 0;

        memset(freqs, 0, (prev_max+1) << 2);
        prev_max = 0;

        int different = 0;
        int l = 0, r = 0;
        const int SIZE = static_cast<int>(nums.size());
        int result = 0;
        while(r < SIZE){
            prev_max = max(prev_max, nums[r]);
            if(++freqs[nums[r++]] == 1){
                ++different;
            }
            while(different > K){
                if(--freqs[nums[l++]] == 0)--different;
            }
            result += r - l;
        }

        return result;
    }

    int subarraysWithKDistinct(vector<int> nums, int k){
        return helper(nums, k) - helper(nums, k - 1);
    }
#undef MAX_VALUES

//2444. Count Subarrays With Fixed Bounds
    ll countSubarrays(vector<int> nums, int minK, int maxK) {
        ll result = 0;
        
        int l = 0, r = 0;
        int curr_max = INT_MIN, curr_min = INT_MAX;
        const int SIZE = static_cast<int>(nums.size());
        while(r < SIZE) {
            curr_max = max(curr_max, nums[r]);
            curr_min = min(curr_min, nums[r]);
            ++r;
            if(curr_max > maxK || curr_min < minK){
                l = r;
                curr_max = INT_MIN;
                curr_min = INT_MAX;
            }
            else if(curr_max == maxK && curr_min == minK){
            }
        }

        return result;
    }


    //840. Magic Squares In Grid

    bool compare(
        const vector<vector<int>>& grid,
        const int (*arr)[3],
        int fromR,
        int fromC
    ){
        for(int r = 0; r < 3; ++r){
            for(int c = 0; c < 3; ++c){
                if(grid[r + fromR][c + fromC] != arr[r][c]){
                    return false;
                }
            }
        }
        return true;
    }

    int numMagicSquaresInside(vector<vector<int>>& grid) {
        static const int arr[8][3][3] = {{{2, 7, 6}, {9, 5, 1}, {4, 3, 8}},{{2, 9, 4}, {7, 5, 3}, {6, 1, 8}},{{4, 3, 8}, {9, 5, 1}, {2, 7, 6}},{{4, 9, 2}, {3, 5, 7}, {8, 1, 6}},{{6, 1, 8}, {7, 5, 3}, {2, 9, 4}},{{6, 7, 2}, {1, 5, 9}, {8, 3, 4}},{{8, 1, 6}, {3, 5, 7}, {4, 9, 2}},{{8, 3, 4}, {1, 5, 9}, {6, 7, 2}}};

        int result = 0;
        const int ROW_LIMIT = static_cast<int>(grid.size()) - 2;
        const int COL_LIMIT = static_cast<int>(grid[0].size()) - 2;
        for(int r = 0; r < ROW_LIMIT; ++r){
            for(int c = 0; c < COL_LIMIT; ++c){
                for(int i = 0; i < 8; ++i){
                    result += static_cast<int>(compare(grid, arr[i], r, c));
                }
            }
        }

        return result;
    }

    //909. Snakes and Ladders
#define MAX_N 400

    pair<int,int> translate(const int N, int pos){

        pos = N * N - pos - 1;
        int row = pos / N;
        int column = pos % N;

        return {row, (N + row) & 1 ? N - column - 1 : column};
    }

    int snakesAndLadders(vector<vector<int>> board) {
        static int memo[MAX_N];
        const int N = board.size();
        const int COUNT = N * N;
        memset(&memo[1], 127, COUNT << 2);
        memo[0] = 0;
        
        show_vector(board, 0b011);

        static deque<int> queue;
        queue.push_back(0);

        while(queue.size() != 0){
            int top = queue.front(); queue.pop_front();

            for(int i = 1; i < 7; ++i){
                auto nextPos = top + i;
                if(nextPos >= COUNT)break;
                auto next = translate(N, nextPos);
                if(board[next.first][next.second] != -1){
                    nextPos = board[next.first][next.second] - 1;
                    next = translate(N, nextPos);
                }
                if(memo[nextPos] > memo[top] + 1){
                    memo[nextPos] = memo[top] + 1;
                    queue.push_back(nextPos);
                }
            }
        }

        return memo[COUNT - 1] == 2139062143 ? -1 : memo[COUNT - 1];
    }
#undef MAX_N

    //1340. Jump Game V
#define MAX_N 1000
    int maxJumps(vector<int> arr, int d) {
        static bool accessible[MAX_N][MAX_N];
        static int indices[MAX_N];

        const int SIZE = static_cast<int>(arr.size());
        for(int i = 0; i < SIZE; ++i){
            indices[i] = i;
            int l = i - 1;
            int lMax = INT_MIN;
            while(l >= 0 && i - l <= d){
                lMax = max(lMax, arr[l]);
                if(lMax >= arr[i])break;
                accessible[i][l] = arr[l] <= lMax;
                --l;
            }
            int r = i + 1;
            int rMax = INT_MIN;
            while(r < SIZE && r - i <= d){
                rMax = max(rMax, arr[r]);
                if(rMax >= arr[i])break;
                accessible[i][r] = arr[r] <= rMax;
                ++r;
            }
        }

        sort(indices, indices + SIZE,[&arr](int l, int r){return arr[l] < arr[r];});

        static int dp[MAX_N];

        int result = 1;
        for(int i = 0; i < SIZE; ++i){
            dp[i] = 1;
            for(int j = 0; j < i; ++j){
                if(!accessible[indices[i]][indices[j]])continue;
                dp[i] = max(dp[i], dp[j] + 1);
            }
            result = max(result, dp[i]);
        }

        return result;
    }
#undef MAX_N

    //1249. Minimum Remove to Make Valid Parentheses

#define MAX_LEN 100000 
    string minRemoveToMakeValid(string s) {
        static int indices[MAX_LEN];
        int indSize = 0;

        const int SIZE = static_cast<int>(s.length());
        for(int i = 0; i < SIZE; ++i){
            switch(s[i]){
                case('('):
                    indices[indSize++] = i;
                break;
                case(')'):
                    if(indSize > 0 && s[indices[indSize - 1]] == '('){
                        --indSize;
                    }
                    else {
                        indices[indSize++] = i;
                    }
                break;
            }
        }

        string result;
        result.reserve(s.length() - indSize);

        int j = 0;
        for(int i = 0; i < SIZE; ++i){
            if(indices[j] == i){
                ++j;
                continue;
            }
            result.push_back(s[i]);
        }

        return result;
    }
#undef MAX_LEN

    //1458. Max Dot Product of Two Subsequences
#define MAX_N 500

    int maxDotProduct(vector<int> nums1, vector<int> nums2) {
        static int FIRST[MAX_N + 1], SECOND[MAX_N + 1];
        
        const int M = static_cast<int>(nums1.size());
        const int N = static_cast<int>(nums2.size());
        int *f = &FIRST[1], *s = &SECOND[1];
        memset(f, 0xF0, N << 2);

		for(int i = 0; i < M; ++i){
            s[-1] = 0xF0000000;
		    for(int j = 0; j < N; ++j){
                s[j] = max(f[j], max(s[j - 1], nums1[i] * nums2[j] + max(0, f[j - 1])));
            }
            swap(f,s);
		}
		
		return f[N - 1];
    }

#undef MAX_N

//678. Valid Parenthesis String
#define MAX_SIZE 100

    bool checkValidString(string s) {
        static StaticStack<MAX_SIZE + 1, int> lefts;        
        static StaticStack<MAX_SIZE, int> stars;

        lefts.clear();
        stars.clear();

        int remove_stars = 0;
        const int SIZE = static_cast<int>(s.length());
        for(int i = 0; i < SIZE; ++i){
            switch(s[i]){
                case('*'):
                    stars.push(i);
                break;
                case('('):
                    lefts.push(i);
                break;
                case(')'):
                    if(lefts.size() > 0)lefts.pop();
                    else if(stars.size() > remove_stars){
                        ++remove_stars;
                    }
                    else return false;
                break;
            }
        }

        while(
            lefts.size() > 0 
            && 
            stars.size() - remove_stars > 0) 
        {
            if(stars.pop() < lefts.pop())return false;
        }

        return lefts.size() == 0;
    }

#undef MAX_SIZE

    //32. Longest Valid Parentheses
    
#define MAX_SIZE 30000
    int longestValidParentheses(string s) {
        //first -> value, second -> index
        static pair<int,int> lefts[MAX_SIZE];
        static int DP[MAX_SIZE + 1];
        static int *dp = &DP[1];

        int lSize = 0;
        int result = 0;
        const int LEN = static_cast<int>(s.length());

        for(int i = 0; i < LEN; ++i){
            dp[i] = 0;
            if(s[i] == '('){
                lefts[lSize++] = {0, i};
                continue;
            }
            if(lSize == 0)continue;

            auto top = lefts[--lSize];
            top.first += 2;
            if(lSize != 0)lefts[lSize - 1].first += top.first;
            dp[i] = top.first + dp[top.second - 1];

            result = max(result, max(dp[i], top.first));
        }

        return result;
    }
#undef MAX_SIZE

    //402. Remove K Digits

#define MAX_SIZE 100000
    string removeKdigits(string num, int k) {
        const int LEN = static_cast<int>(num.length());
        if(k == LEN)return "0";
        
        static StaticStack<MAX_SIZE, char> stack;
        stack.clear();

        const int REM_LEN = LEN - k;
        for(int i = 0; i < LEN; ++i){
            while(
                stack.size() > 0 &&
                stack.top() > num[i] &&
                stack.size() + LEN - i > REM_LEN
            ){
                stack.pop();
            }
            if(stack.size() < REM_LEN){
                stack.push(num[i]);
            }
        }

        int zeroes_offset = 0;
        while(zeroes_offset + 1 < REM_LEN){
            if(stack[zeroes_offset] != '0')break;
            ++zeroes_offset;
        }

        string result(REM_LEN - zeroes_offset,' ');
        memcpy(&result[0], &stack[zeroes_offset], REM_LEN - zeroes_offset);
        return result;
    }
#undef MAX_SIZE


    //2195. Append K Integers With Minimal Sum

    //sum(x) - sum(y) ->
    // if x >= y, then
    // 1) x = y + k
    // 2) sum(y + k) - sum(y) = k * (k + 2y + 1) / 2

    long long sumDiff(int from, int to){
        int diff = to - from;
        return static_cast<long long>(diff * (diff + 2 * from + 1)) / 2ll;
    }

    long long minimalKSum(vector<int> nums, int k) {
        sort(nums.begin(), nums.end());

        const int SIZE = static_cast<int>(nums.size());
        long long result = 0;
        int prev = 0;
        for(int i = 0; i < SIZE; ++i){
            int dist = min(k, max(0, nums[i] - prev - 1));
            result += sumDiff(prev, prev + dist);
            prev = nums[i];
            k -= dist;
            if(k == 0)break;
        }

        return result + sumDiff(prev, prev + k);
    }

    //2281. Sum of Total Strength of Wizards

#define MAX_SIZE 100000
    ll totalStrength(vector<int> strength) {
        
        const ll SIZE = static_cast<ll>(strength.size());

        //count reversed prefix
        static ll prefix[2][MAX_SIZE + 1];
        prefix[1][SIZE] = prefix[0][SIZE] = 0ll;
        for(ll i = SIZE - 1; 0 <= i; --i){
            prefix[0][i] = prefix[0][i + 1] + static_cast<ll>(strength[i]);
            prefix[1][i] = prefix[1][i + 1] + prefix[0][i];
        }

        //main logic
        //first -> value, second -> index
        static pair<ll,ll> stack[MAX_SIZE + 1] = {{0, -1}};
        ll stackIndex = 0;
        ll result = 0;
        ll sum_mult = 0;
        ll run_sum = 0;
        for(ll i = 0; i < SIZE; ++i){
            ll str = strength[i];
            ll prev_prefix = 0;
            ll prev_index = i;
            while(stack[stackIndex].first >= strength[i]){
                auto [x, index] = stack[stackIndex--];
                const ll diff = x - str;
                ll curr_prefix = (prefix[1][stack[stackIndex].second + 1] - prefix[1][i] - prefix[0][i] * (i - stack[stackIndex].second - 1)) % MODULO;
                run_sum = (MODULO + run_sum - (curr_prefix - prev_prefix) * diff) % MODULO;
                sum_mult = (MODULO + sum_mult - x * (index - stack[stackIndex].second)) % MODULO;
                prev_prefix = curr_prefix;
                prev_index = index;
            }
            sum_mult = (sum_mult + str * (i - stack[stackIndex].second)) % MODULO;
            run_sum = (run_sum + sum_mult * str) % MODULO;
            result = (result + run_sum) % MODULO;
            stack[++stackIndex] = {str, i};
        }

        return result;
    }
#undef MAX_SIZE

    //1190. Reverse Substrings Between Each Pair of Parentheses

    string reverseParentheses(string s) {
        stack<string> stack;
        stack.push("");

        int depth = 0;
        for(char c : s){
            if(c == '('){
                stack.push("");
                ++depth;
            }
            else if(c == ')'){
                string top = stack.top(); stack.pop();
                if(depth & 1) {
                    reverse(top.begin(), top.end());
                }
                stack.top().append(top);
                --depth;
            }
            else {
                stack.top().push_back(c);
            }
        }

        return stack.top();
    }

    //407. Trapping Rain Water II

#define MAX_ROWS 200
#define MAX_COLUMNS 200
    int trapRainWater(vector<vector<int>>& heightMap) {
        //use 2d trapRainWater with treshold
        //calc threshold
        
        const int ROWS = static_cast<int>(heightMap.size());
        const int COLUMNS = static_cast<int>(heightMap[0].size());
        
        static int threshold[MAX_ROWS][MAX_COLUMNS];
        for(int c = 0; c < COLUMNS; ++c){
            for(int r = 0; r < ROWS; ++c){
                //calc threshold for every column, write down in corresponding cell
                //threshold means the min height in both sides
            }
        }

        int result = 0;
        for(int r = 0; r < ROWS; ++r){
            //calc using 2d trapRain water based on previously calculated threshold
        }

        return 0;
    }

    string countOfAtoms(string formula) {
        stack<unordered_map<string, int>> stack;
        stack.push({});

        string current;
        int count;
        const int SIZE = static_cast<int>(formula.length());
        for(int i = 0; i < SIZE; ++i){
            if(formula[i] == '('){
                stack.top()[current] += max(1, count);
                current.clear();
                stack.push({});
                continue;
            }
            if(formula[i] == ')'){
                stack.top()[current] += max(1, count);
                current.clear();
                
                int mult = 0;
                i += 1;
                while(i < SIZE && isdigit(formula[i])){
                    mult = mult * 10 + formula[i] - '0';
                    ++i;
                }
                i -= 1;

                mult = max(1, mult);
                auto top = stack.top(); stack.pop();
                for(const auto& pair : top){
                    stack.top()[pair.first] += pair.second * mult;
                }

                continue;
            }
            if(isupper(formula[i])){
                stack.top()[current] += max(1, count);
                current.clear();
                current.push_back(formula[i]);
                count = 0;
            }
            else if(islower(formula[i])){
                current.push_back(formula[i]);
            }
            else {
                count = count * 10 + formula[i] - '0';
            }
        }
        stack.top()[current] += max(1, count);

        //clean and sort
        stack.top().erase("");
        
        vector<string> sorted;
        sorted.reserve(stack.top().size());

        for(const auto& pair : stack.top()){
            sorted.push_back(pair.first);
        }

        sort(sorted.begin(), sorted.end());

        //assemble
        string result = "";
        for(const auto& name : sorted){
            result += name;
            if(stack.top()[name] > 1){
                result += to_string(stack.top()[name]);
            }
        }

        return result;
    }

    double new21Game(int n, int k, int maxPts) {
        //?
        return 0; 
    }

    int threeSumClosest(vector<int> nums, int target) {
        sort(nums.begin(), nums.end());

        int result = INT_MAX;

        const int SIZE = static_cast<int>(nums.size());
        const int lLimit = SIZE - 2;
        for(int l = 0; l < lLimit; ++l){
            const int rLimit = l + 1;
            for(int r = SIZE - 1; rLimit < r; --r){
                const int toFind = target - nums[l] - nums[r];
                int index = upper_bound(nums.begin() + l + 1, nums.begin() + r - 1, toFind) - nums.begin();
                if(index - 1 != l && abs(toFind - nums[index]) > abs(toFind - nums[index - 1]))--index;
                if(abs(target - nums[index] - nums[l] - nums[r]) < abs(target - result)){
                    result = nums[index] + nums[l] + nums[r];
                }
            }
        }

        return result;
    }

    vector<vector<int>> findFarmland(vector<vector<int>> land) {
        const int ROWS    = static_cast<int>(land.size());
        const int COLUMNS = static_cast<int>(land[0].size());
        //main logic
        vector<vector<int>> result;
        for(int r = 0; r < ROWS; ++r){
            for(int c = 0; c < COLUMNS; ++c){
                if(land[r][c] != 0){
                    auto res = bfs(land, r, c);
                    result.emplace_back(res.begin(), res.end());
                }
            }
        }

        return result;
    }

    array<int, 4> bfs(
        vector<vector<int>>& grid,
        int y, int x
    ){
        static const int directions[2][4] = {{1,0,-1,0},{0,1,0,-1}};
        pair<int,int> topLeft = {y,x}, botRight = {y,x};
        
        //first -> rows, second -> column
        deque<pair<int,int>> queue;
        queue.push_back({y,x});

        const int ROWS = static_cast<int>(grid.size());
        const int COLUMNS = static_cast<int>(grid[0].size());
        while(queue.size() != 0){
            auto[r,c] = queue.front(); queue.pop_front();

            if(r < topLeft.first || (r == topLeft.first && c < topLeft.second)){
                topLeft = {r,c};
            }
            else if(r > botRight.first || (r == botRight.first && c > botRight.second)){
                botRight = {r,c};
            }

            for(int i = 0; i < 4; ++i){
                int nr = r + directions[0][i];
                int nc = c + directions[1][i];
                if(nr >= ROWS || nr < 0 || nc >= COLUMNS || nc < 0)continue;
                if(grid[nr][nc] == 0)continue;
                grid[nr][nc] = 0;
                queue.push_back({nr, nc});
            }
        }

        return {topLeft.first, topLeft.second, botRight.first, botRight.second};
    }


    //873. Length of Longest Fibonacci Subsequence

#define MAX_SIZE 1000
    int lenLongestFibSubseq(vector<int>& arr) {
        
        static StaticStack<MAX_SIZE, pair<int,int>> prevs[MAX_SIZE];
        
        const int SIZE = static_cast<int>(arr.size());
        unordered_map<int, int> present;
        for(int i = 0; i < SIZE; ++i){
            present.insert({arr[i], i});
            prevs[i].clear();
        }

        int result = 0;

        for(int i = 1; i < SIZE; ++i){
            
            for(int j = 0; j < prevs[i].size(); ++j){
                int sum = prevs[i][j].first + arr[i];
                if(present.count(sum) == 0)continue;
                int index = present[sum];
                result = max(result, prevs[i][j].second + 1);
                prevs[index].push({arr[i], prevs[i][j].second + 1});
            }

            for(int j = 0; j < i; ++j){
                int sum = arr[i] + arr[j];
                if(present.count(sum) == 0)continue;
                int index = present[sum];
                result = max(result ,3);
                prevs[index].push({arr[i], 3});
            }
        }

        return result;
    }

    //310. Minimum Height Trees
    #define MAX_N 20000
    vector<int> findMinHeightTrees(int n, vector<vector<int>> edges) {

        vector<pair<int,deque<int>>> connections(n);

        for(const auto& edge : edges){
            connections[edge[0]].second.push_back(edge[1]);
            connections[edge[1]].second.push_back(edge[0]);
        }

        static bool visited[MAX_N];
        memset(visited, 0, n);

        deque<int> queue;
        for(int i = 0; i < n; ++i){
            connections[i].first = connections[i].second.size();
            if(connections[i].first <= 1){
                queue.push_back(i);
                visited[i] = 1;
            }
        }
        int visited_count = queue.size();
        
        while(visited_count != n){
            int size = queue.size();
            while(size-- > 0){
                int top = queue.front(); queue.pop_front();
                for(int next : connections[top].second){
                    if(--connections[next].first == 1 && !visited[next]){
                        visited[next] = 1;
                        queue.push_back(next);
                    }
                }
            }
            visited_count += queue.size();
        }

        return vector<int>(queue.begin(), queue.end());       
    }
#undef MAX_SIZE


    bool isInterleave(string s1, string s2, string s3) {
        return false;
    }

    //514. Freedom Trail

#define MAX_SIZE 100
    int findRotateSteps(string ring, string key) {
        static int indices[26][MAX_SIZE]; 
        static int sizes[26];


        const int LENR = ring.length();
        const int LENK = key.length();

        //init indices data
        memset(sizes, 0, 104);
        for(int i = 0; i < LENR; ++i){
            int index = ring[i] - 'a';
            indices[index][sizes[index]++] = i;
        }

        //first -> index, second -> value
        deque<pair<int,int>> F, S;
        //fill the start values
        for(int i = 0; i < sizes[key[0] - 'a']; ++i){
            F.push_back({
                indices[key[0] - 'a'][i], 
                min(
                    indices[key[0]-'a'][i],    
                    LENR - indices[key[0]-'a'][i]
                )
            });
        }

        //bfs
        auto f = &F,s = &S;
        for(int i = 1; i < LENK; ++i){
            s->clear();
            int charIndex = key[i] - 'a';
            for(int j = 0; j < sizes[charIndex]; ++j){
                int minVal = INT_MAX;
                for(const auto&[index, val] : *f){
                    minVal = min(
                        minVal, 
                        val + min(
                            abs(index - indices[charIndex][j]),
                            LENR - max(index, indices[charIndex][j]) + min(index, indices[charIndex][j])
                        )
                    );
                }
                s->push_back({indices[charIndex][j], minVal});
            }
            swap(f,s);
        }

        int result = INT_MAX;
        for(const auto&[_,val] : *f){
            result = min(result, val);
        }

        return LENK + result;
    }
#undef MAX_SIZE

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