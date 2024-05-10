#pragma once
#include "../../../GeneralLibs/Headers/includes.hpp";
#include "DataStructures.hpp"


class Solution {
public:
    //228. Summary Ranges
    vector<string> summaryRanges(vector<int>& nums) {

        if (nums.size() == 0)return {};
        vector<string> result;

        nums.push_back(nums.back());

        const size_t size = nums.size();
        int beginning = 0;
        for (int k = 1; k < size; ++k) {
            if (int64_t(nums[k]) - nums[k - 1] == 1)continue;
            result.push_back(to_string(nums[beginning]));
            if (k - beginning > 1)result.back() += "->" + to_string(nums[k - 1]);
            beginning = k;
        }

        return result;
    }

    //435. Non-overlapping Intervals

    int eraseOverlapIntervals(vector<vector<int>> intervals) {
        sort(intervals.begin(), intervals.end(),
            [](const vector<int>& l, const vector<int>& r) {
                if (l[0] == r[0])return l[1] < r[1];
                return l[0] < r[0];
            }
        );

        int result = 0;
        int previous = 0;
        for (int i = 1; i < intervals.size(); ++i) {
            if (intervals[previous][1] <= intervals[i][1] 
                &&
                intervals[i][0] < intervals[previous][1]) {
                ++result;
            }
            else previous = i;
        }

        return result;
    }

    //49. Group Anagrams
    uint64_t count(const string& str) {
        static const uint64_t data[26] = { 88529281,92236816,96059601,100000000,104060401,108243216,112550881,116985856,121550625,126247696,131079601,136048896,141158161,146410000,151807041,157351936,163047361,168896016,174900625,181063936,187388721,193877776,200533921,207360000,214358881,221533456 };
        uint64_t result = 0;
        for (char c : str)result += c - 'a';
        return result;
    }

    vector<vector<string>> groupAnagrams(vector<string>& strs) {
        const size_t size = strs.size();
        vector<pair<uint64_t, size_t>> as_int(size);
        for (size_t k = 0; k < size; ++k) {
            as_int[k] = { count(strs[k]),k };
        }

        sort(as_int.begin(), as_int.end(),
            [](const auto& l, const auto& r)
            {return l.first < r.first; }
        );

        vector<vector<string>> result = {};
        for (int k = 0; k < size; ++k) {
            result.push_back({ strs[as_int[k].second] });
            while (k < size - 1 && as_int[k].first == as_int[k + 1].first) {
                result.back().push_back(strs[as_int[++k].second]);
            }
        }

        return result;
    }

    //342. Power of Four
    bool isPowerOfFour(int n) {
        return n - (n - 1) == 1;
    }

    //2583. Kth Largest Sum in a Binary Tree
    void dfs_2583(tree_node::TreeNode* root, size_t level, unordered_map<size_t, uint64_t>& data) {
        if (root == nullptr)return;
        data[level] += root->val;
        dfs_2583(root->left, level + 1, data);
        dfs_2583(root->right, level + 1, data);
    }

    long long kthLargestLevelSum(tree_node::TreeNode* root, int k) {
        unordered_map<size_t, uint64_t> data;
        dfs_2583(root, 0, data);
        if (data.size() < k)return -1;
        
        priority_queue<uint64_t> queue;
        for (const auto& pair : data) {
            queue.push(pair.second);
            if (queue.size() > k)queue.pop();
        }

        return queue.top();
    }

    //1569. Number of Ways to Reorder Array to Get Same BST
    struct ListNode1 {
        uint16_t val;
        ListNode1* next;
        ListNode1(uint16_t x, ListNode1* next = nullptr) : val(x), next(next) {}
    };

    class List1 {
    public:
        size_t size = 0;
        ListNode1* root = nullptr;
        ListNode1* tail = nullptr;

        inline void add(ListNode1* node) {
            if (root == nullptr)
                tail = root = node;
            else
                tail = tail->next = node;
            ++size;
        }
    };

        
    static const size_t modulo = 1000000007;
    uint64_t combinations(uint16_t n, uint16_t k) {
        static vector<vector<uint64_t>> data = { {1},{1,1} };

        for (size_t current_size = data.size(); current_size <= n; ++current_size) {
            data.push_back(vector<uint64_t>(data.back().size() + 1, 1));

            for (int k = 1; k < current_size; ++k) {
                data[current_size][k] = (data[current_size - 1][k - 1] + data[current_size - 1][k]) % modulo;
            }
        }

        return data[n][k];
    }

    uint64_t calculateWays(List1& data) {
        if (data.size < 3) return 1;

        List1 left, right;

        ListNode1* itr = data.root->next;
        for (size_t i = 1; i < data.size; ++i) {
            if (itr->val < data.root->val)
                left.add(itr);
            else
                right.add(itr);
            itr = itr->next;
        }

        uint64_t l = calculateWays(left) % modulo;
        uint64_t r = calculateWays(right) % modulo;

        return ((l * r) % modulo) * combinations(data.size - 1, left.size) % modulo;
    }

    int numOfWays(const vector<int>& nums) {
        const size_t size = nums.size();
        List1 list;

        for (size_t k = 0; k < size; ++k) {
            list.add(new ListNode1(nums[k]));
        }

        return (calculateWays(list) - 1) % modulo;
    }


    //1267. Count Servers that Communicate
    int dfs(vector<vector<int>>& grid, uint16_t x, uint16_t y) {
        int result = 0;
        
        grid[y][x] = 0;

        for (uint16_t k = 0; k < grid.size(); ++k) {
            if (grid[k][x])result += dfs(grid, x, k);
        }
        for (uint16_t k = 0; k < grid[0].size(); ++k) {
            if (grid[y][k])result += dfs(grid, k, y);
        }

        return result + 1;
    }

    int countServers(vector<vector<int>>& grid) {
        const uint16_t N = grid.size();
        const uint16_t M = grid[0].size();

        int result = 0;

        for (uint16_t y = 0; y < N; ++y) {
            for (uint16_t x = 0; x < M; ++x) {
                if (grid[y][x]) result = max(result, dfs(grid, x, y));
            }
        }

        return result == 1 ? 0 : result;
    }

    //84. Largest Rectangle in Histogram
    int largestRectangleArea(const vector<int>& heights) {
        const size_t SIZE = heights.size();

        int result = 0;
        stack<pair<int, int>> data;

        for (int i = 0; i < SIZE; ++i) {
            int min = i;
            while (!data.empty() && heights[i] < data.top().first) {
                auto top = data.top(); data.pop();
                min = top.second;
                result = max(result, top.first * (i - top.second));
            }
            data.push({ heights[i],min });
        }

        while (not data.empty()) {
            int top = data.top().first * (SIZE - data.top().second);
            data.pop();
            if (top > result)result = top;
        }

        return result;
    }

    //63. Unique Paths II
    int uniquePathsWithObstacles(vector<vector<int>>& obstacleGrid) {

        if (obstacleGrid[0][0] == 1 || obstacleGrid.back().back() == 1)return 0;

        obstacleGrid[0][0] = -1;

        const char ROWS = obstacleGrid.size();
        const char COLUMNS = obstacleGrid[0].size();
        for (char r = 0; r < ROWS; ++r) {
            for (char c = 0; c < COLUMNS; ++c) {
                if (obstacleGrid[r][c] == 1)
                    continue;
                if (c + 1 < COLUMNS && obstacleGrid[r][c + 1] != 1)
                    obstacleGrid[r][c + 1] += obstacleGrid[r][c];
                if (r + 1 < ROWS && obstacleGrid[r + 1][c] != 1)
                    obstacleGrid[r + 1][c] += obstacleGrid[r][c];
            }
        }

        return abs(obstacleGrid.back().back());
    }
    
    //2328. Number of Increasing Paths in a Grid

    int dfs_2328(const vector<vector<int>>& grid,int length,uint16_t x,uint16_t y) {
        const vector<pair<int, int>> directions = { {0, 1}, {1, 0}, {-1, 0}, {0, -1} };
        
        int result = 0;
        bool extend = false;
        for (auto& pair : directions) {
            uint16_t xn = pair.first + x;
            uint16_t yn = pair.second + y;

            if (yn < grid.size() && xn < grid[0].size() && grid[yn][xn] > grid[x][y]) {
                extend = true;
                result = (result + dfs_2328(grid, length + 1, xn, yn)) % modulo;
            }

        }

        if (!extend)result += 0;
        return result % modulo;
    }

    //unsolved
    int countPaths(vector<vector<int>>& grid) {
        return 0;
    }

    //2090. K Radius Subarray Averages
    vector<int> getAverages(vector<int>& nums, int k) {
        const size_t SIZE = nums.size();
        const size_t D = k * 2 + 1;
        vector<int> result(SIZE, -1);

        if (SIZE < D)return result;

        size_t sum = 0;
        size_t i = 0;
        while (i < D) {
            sum += nums[i++];
        }

        for (size_t n = 0; true;++n) {
            result[k++] = sum / D;
            if (SIZE <= i)break;
            sum += nums[i++] - nums[n];
        }

        return result;
    }

    //15. 3Sum
    //idea : 
    //left at most 2 repeatable elements
    vector<pair<int, int>> twoSumFind(const vector<int>& nums, int from, int target) {
        vector<pair<int, int>> pairs;
        int r = nums.size() - 1;
        while (from < r) {
            if (nums[from] + nums[r] == target) {
                if (pairs.empty())
                    pairs.push_back({nums[from],nums[r]});
                else if(pairs.back() != make_pair(nums[from],nums[r]))
                    pairs.push_back({ nums[from],nums[r] });
            }
            if (target <= nums[from] + nums[r])--r;
            else ++from;
        }

        return pairs;
    }

    vector<vector<int>> threeSum(vector<int> nums) {
        sort(nums.begin(), nums.end());
        vector<vector<int>> result;

        const int LIMIT = nums.size() - 2;
        int i = -1;
        while (++i < LIMIT) {
            auto pairs = twoSumFind(nums, i + 1, -nums[i]);
            if (pairs.size() > 0) {
                for (auto& [l, r] : pairs) {
                    result.push_back({ nums[i],l,r });
                }
            }
            while (i < LIMIT && nums[i] == nums[i + 1])++i;
        }

        return result;
    }

    //463. Island Perimeter
    int dfs_463(vector<vector<int>>& grid,uint16_t x,uint16_t y) {
        const static vector<pair<int, int>> directions = { {0, 1}, {1, 0}, {-1, 0}, {0, -1} };
        int result = 0;

        grid[y][x] = 2;

        for (auto& pair : directions) {
            uint16_t xn = x + pair.first;
            uint16_t yn = y + pair.second;
            if (grid[0].size() <= xn || grid.size() <= yn)++result;
            else if (grid[yn][xn] == 0)++result;
            else if (grid[yn][xn] != 2)result += dfs_463(grid, xn, yn);
        }
        
        return result;
    }

    int islandPerimeter(vector<vector<int>>& grid) {
        
        const uint16_t ROWS = grid.size();
        const uint16_t COLUMNS = grid[0].size();

        for (uint16_t r = 0; r < ROWS; ++r) {
            for (uint16_t c = 0; c < COLUMNS; ++c) {
                if (grid[r][c] != 0)return dfs_463(grid, c, r);
            }
        }

        return 0;
    }

    //2685. Count the Number of Complete Components
    //pair<int, int> dfs(const vector<vector<char>>& relations, vector<bool>& visited, int n) {
    //    if (visited[n])return { 0,0 };
    //    visited[n] = true;
    //    
    //    pair<int, int> result = { 1,0 };

    //    for (int e : relations[n]) {
    //        auto nested = dfs(relations, visited, e);
    //        result.first += nested.
    //    }

    //    return result;
    //}

    int countCompleteComponents(int n, vector<vector<int>>& edges) {
        vector<vector<char>> relations(n);

        for (auto& edge : edges) {
            relations[edge[0]].push_back(edge[1]);
            relations[edge[1]].push_back(edge[0]);
        }

        vector<bool> visited(n,false);
        for (int k = 0; k < n; ++k) {

        }

        return 0;
    }

    //2470. Number of Subarrays With LCM Equal to K
    int subarrayLCM(vector<int>& nums, int k) {
        const size_t SIZE = nums.size();
        int result = 0;
        int l = 0, r = 0;
        while (r < SIZE) {
            if (nums[r] == k)++result;
            if (k % nums[r] == 0)r++;
            else {
                result += 0;
            }
        }
        return result;
    }

    //61. Rotate List
    ListNode* rotateRight(ListNode* head, int k) {
        if (head == nullptr || k == 0)return head;

        ListNode* right = head, * left = head;

        int offset = 0;
        while (offset++ < k && right->next != nullptr) {
            right = right->next;
        }

        if (offset <= k) return rotateRight(head, k % offset);

        while (right->next != nullptr) {
            left = left->next;
            right = right->next;
        }

        ListNode* result = left->next;
        left->next = nullptr;
        right->next = head;

        return result;
    }

    //322. Coin Change
    int coinChange(vector<int> coins, int amount) {
        if (amount == 0)return 0;

        vector<int> dp(amount + 1, 0);

        for (int coin : coins) {
            if (coin <= amount)dp[coin] = 1;
        }

        for (int64_t i = 0; i < amount; ++i) {
            if (dp[i] == 0)continue;
            for (int coin : coins) {
                if (amount < i + coin)continue;
                if (dp[i + coin] == 0)dp[i + coin] = dp[i] + 1;
                else if (dp[i + coin] > dp[i] + 1)dp[i + coin] = dp[i] + 1;
            }
        }

        return dp[amount] == 0 ? -1 : dp[amount];
    }

    //2462. Total Cost to Hire K Workers
    long long totalCost(vector<int> costs, int k, int candidates) {
        if (costs.size() == 1)return costs[0];

        priority_queue<int, vector<int>, greater<int>> left, right;

        candidates = min(int(costs.size()) / 2, candidates);
        int l = 0, r = costs.size() - 1;

        while (l < candidates) {
            left.push(costs[l++]);
            right.push(costs[r--]);
        }

        long long answer = 0;

        while (0 < k) {
            if (left.empty() || right.empty())break;

            if (left.top() <= right.top()) {
                answer += left.top(); left.pop();
                if (l <= r)left.push(costs[l++]);
            }
            else {
                answer += right.top(); right.pop();
                if (l <= r)right.push(costs[r--]);
            }
            --k;
        }

        while (0 < k && not left.empty()) {
            answer += left.top(); left.pop();
            --k;
        }

        while (0 < k && not right.empty()) {
            answer += right.top(); right.pop();
            --k;
        }

        return answer;
    }

    //66. Plus One
    vector<int> plusOne(vector<int>& digits) {
        const size_t SIZE = digits.size();

        ++digits.back();
        int i = SIZE - 1;
        while (0 < i && digits[i] > 9) {
            digits[i - 1] += digits[i] % 10;
            digits[i] %= 10;
        }

        if (digits[0] == 10) {
            digits[0] = 1;
            digits.push_back(0);
        }

        return digits;
    }

    //373. Find K Pairs with Smallest Sums
    vector<vector<int>> kSmallestPairs(vector<int> nums1, vector<int> nums2, int k) {
        vector<vector<int>> result;
        const int SIZE_L = nums1.size() - 1;
        const int SIZE_R = nums2.size() - 1;

        k = min((SIZE_L+1) * (SIZE_R+1), k);

        int l = 0, r = 0;
        while (0 < k--) {
            result.push_back({ nums1[l],nums2[r] });
            if (SIZE_L <= l && SIZE_R <= r)break;
            if (SIZE_L <= l)++r;
            else if (SIZE_R <= r)++l;
            else if (nums1[l] + nums2[r + 1] < nums1[l + 1] + nums2[r])++r;
            else ++l;
        }

        return result;
    }

    int longestCommonSubsequence(string text1, string text2) {
        if (text1.length() > text2.length())text1.swap(text2);
        const size_t LEN1 = text1.length();
        const size_t LEN2 = text2.length();

        vector<int> dp(LEN1+1, 0);

        for (size_t l1 = 0; l1 < LEN1; ++l1) {
            for (size_t l2 = 1; l2 < LEN1; ++l2) {
                dp[l1] = max(dp[l1],dp[l1-1] + (text1[l1] == text2[l2]));
            }
        }

        return dp[LEN1 - 1];
    }

    //795. Number of Subarrays with Bounded Maximum
    int numSubarrayBoundedMax(vector<int> nums, int left, int right) {
        int result = 0;

        const size_t SIZE = nums.size();
        int l = 0, r = 0;
        int local_max = nums[0];
        while (r < SIZE) {
            if (local_max < nums[r])local_max = nums[r];

            if (left <= local_max && local_max <= right)++r;
            else {
                local_max = nums[r];
                ++l;
            }
            result += r-l;
        }

        return result;
    }

    //1493. Longest Subarray of 1's After Deleting One Element
    int longestSubarray(vector<int>& nums) {
        const size_t SIZE = nums.size();
        if (SIZE == 1)return 0;

        vector<int> zeroes = { -1 };
        int i = -1;
        while (++i < SIZE) {
            if (nums[i] == 0)zeroes.push_back(i);
        }
        if (zeroes.size() < 2)return SIZE - 1;

        zeroes.push_back(SIZE);
        const size_t SIZE2 = zeroes.size() - 1;

        int answer = 0;
        i = 0;
        while (++i < SIZE2) {
            answer = max(answer, zeroes[i + 1] - zeroes[i - 1] - 2);
        }

        return answer;
    }

    //1268. Search Suggestions System
    vector<vector<string>> suggestedProducts(vector<string>& products, string searchWord) {
        return {};
    }

    //931. Minimum Falling Path Sum
    int minFallingPathSum(vector<vector<int>>& matrix) {
        const size_t N = matrix.size();

        if (N == 1)return matrix[0][0];

        vector<vector<int>> dp(N - 1, vector<int>(N, INT_MAX));
        vector<int>* current = &matrix[0];

        for (int r = 0; r < N - 1; ++r) {
            for (int c = 0; c < N; ++c) {
                dp[r][c] = min(dp[r][c], (*current)[c] + matrix[r + 1][c]);
                if (c > 0)dp[r][c - 1] = min(dp[r][c - 1], (*current)[c] + matrix[r + 1][c - 1]);
                if (c+1 < N)dp[r][c + 1] = min(dp[r][c + 1], (*current)[c] + matrix[r + 1][c + 1]);
            }
            current = &dp[r];
        }

        return *min_element(dp.back().begin(), dp.back().end());
    }

    //1424. Diagonal Traverse II
    inline void diagonalTraverse(const int ROWS, const int COLUMNS, pair<int, int>& coordinates) {
        int diagonal = coordinates.first + coordinates.second;
        coordinates.first--;
        coordinates.second++;
        /*if (coordinates.second == COLUMNS) {
            coordinates.first = ROWS - 1;
            coordinates.second = diagonal % (ROWS - 1);
        }
        else if (coordinates.first < 0) {
            coordinates.first = diagonal + 1;
            coordinates.second = 0;
        }*/
        if (coordinates.second == COLUMNS || 0 < coordinates.first) {
            if (ROWS <= diagonal + 1) {
                coordinates.first = ROWS - 1;
                coordinates.second = diagonal - ROWS + 1;
            }
            else {
                coordinates.first = diagonal + 1;
                coordinates.second = 0;
            }
        }
    }
    
    //static const int right_mask = 0b00000000000000001111111111111111;

    //static bool diagonalComparer(int l, int r) {
    //    int dl = (l & right_mask) + (l >> 16);
    //    int dr = (r & right_mask) + (r >> 16);
    //    return dl == dr ? (r >> 16) < (l >> 16) : dl < dr;
    //}

    vector<int> findDiagonalOrder(vector<vector<int>> nums) {
        
        vector<pair<size_t,size_t>> coordinates;
        const size_t R = nums.size();
        for (int r = 0; r < R; ++r) {
            for (int c = 0; c < nums[r].size(); ++c) {
                coordinates.push_back({r,c});
            }
        }

        sort(coordinates.begin(), coordinates.end(), 
            [](const auto& l, const auto& r)
            {
                if (l.first + l.second == r.first + r.second) {
                    return r.first < l.first;
                }
                return l.first + l.second < r.first + r.second;
            }
        );
        const size_t SIZE = coordinates.size();
        vector<int> result(SIZE);
        for (int i = 0; i < SIZE; ++i) {
            result[i] = nums[coordinates[i].first][coordinates[i].second];
        }

        return result;
    }

    //529. Minesweeper
    //not solved
    vector<vector<char>>& updateBoard(vector<vector<char>>& board, vector<int>& click) {
        static const vector<pair<int, int>> directions = { {0,1},{1,0},{-1,0},{0,-1},{1,1},{-1,1},{1,-1},{-1,-1} };

        if (board[click[0]][click[1]] == 'M') {
            board[click[0]][click[1]] = 'X';
            return board;
        }

        const uint16_t ROWS = board.size();
        const uint16_t COLUMNS = board[0].size();

        deque<pair<uint16_t, uint16_t>> queue{ {click[0],click[1]} };

        while (not queue.empty()) {
            auto [y, x] = queue.front(); queue.pop_front();
            if (board[y][x] != 'E')continue;

            char mines_around = 0;

            for (const auto& direction : directions) {
                uint16_t ny = y + direction.first;
                uint16_t nx = x + direction.second;
                if (ny < ROWS && nx < COLUMNS) {
                    if (board[ny][nx] == 'M')++mines_around;
                    else queue.push_back({ ny,nx });
                }
            }

            if (0 < mines_around) {
                board[y][x] = mines_around + 48;
                continue;
            }

            board[y][x] = 'B';
        }

        return board;
    }

    //2055. Plates Between Candles
    vector<int> platesBetweenCandles(string s, vector<vector<int>>& queries) {
        const size_t LEN = s.length();
        vector<int> candles;
        for (int i = 0; i < LEN; ++i) {
            if (s[i] == '|') candles.push_back(i);
        }

        candles.push_back(LEN);

        const int SIZE = queries.size();
        vector<int> answer(SIZE, 0);
        for (size_t i = 0; i < SIZE; ++i) {
            int l = lower_bound(candles.begin(), candles.end(), queries[i][0]) - candles.begin();
            int r = lower_bound(candles.begin(), candles.end(), queries[i][1]) - candles.begin();
            if (l < SIZE && candles[l] < queries[i][0]) ++l;
            if (0 < r && candles[r] > queries[i][1]) --r;
            answer[i] = max(0, candles[r] - candles[l] - r + l);
        }

        return answer;
    }

    //983. Minimum Cost For Tickets
    //finally solved!!!
    int mincostTickets(vector<int>& days, vector<int>& costs) {
        static vector<int> dp(366, 0);
        const int LAST = days.back();

        memset(&dp[1], 9999999, LAST * 4);

        // costs[0] = min(costs[0], min(costs[1], costs[2]));
        // costs[1] = min(costs[1], costs[2]);

        int day = 0;
        int i = 1;
        while (i <= LAST) {
            if (i < days[day])dp[i] = min(dp[i - 1], dp[i]);
            else {
                dp[i] = min(dp[i - 1] + costs[0], dp[i]);
                dp[min(LAST, i + 6)] = min(dp[i - 1] + costs[1], dp[min(LAST, i + 6)]);
                dp[min(LAST, i + 29)] = min(dp[i - 1] + costs[2], dp[min(LAST, i + 29)]);
                ++day;
            }
            ++i;
        }

        return dp[LAST];
    }

    //2305. Fair Distribution of Cookies
    int splitCookies(vector<int>::iterator bag, const vector<int>::iterator& end, vector<int>& children) {

        if (bag == end) {
            return *max_element(children.begin(), children.end());
        }

        int best = INT_MAX;

        for (int k = 0; k < children.size(); ++k) {
            children[k] += *bag;
            best = min(best, splitCookies(bag + 1, end, children));
            children[k] -= *bag;
        }

        return best;
    }

    int distributeCookies(vector<int>& cookies, int k) {
        vector<int> children(k, 0);
        return splitCookies(cookies.begin(), cookies.end(), children);
    }

    //1672. Richest Customer Wealth
    int maximumWealth(vector<vector<int>>& accounts) {
        int result = 0;
        for (auto& account : accounts) {
            result = max(result, accumulate(account.begin(), account.end(), 0));
        }
        return result;
    }

    static inline void UP    (pair<int, int>& crd) { ++crd.first;  }
    static inline void DOWN  (pair<int, int>& crd) { --crd.first;  }
    static inline void RIGHT (pair<int, int>& crd) { ++crd.second; }
    static inline void LEFT  (pair<int, int>& crd) { --crd.second; }

    bool judgeCircle(string moves) {
        static unordered_map<char, function<void(pair<int, int>&)>> operations{
            {'U', UP},
            {'D', DOWN},
            {'L', LEFT},
            {'R', RIGHT}
        };

        pair<int, int> coordinate = { 0,0 };
        for (char c : moves) {
            operations[c](coordinate);
        }
        return coordinate.first == coordinate.second == 0;
    }

    //1601. Maximum Number of Achievable Transfer Requests
    int isPossible(const vector<vector<int>>& requests, int n, int num) {
        vector<short> states(n, 0);
        int amount = 0;
        for (int i = 0; i < requests.size(); ++i) {
            if (num & (1 << i)) {
                --states[requests[i][0]];
                ++states[requests[i][1]];
                ++amount;
            }
        }

        for (int i = 0; i < n; ++i) {
            if (states[i] != 0)return 0;
        }

        return amount;
    }

    int maximumRequests(int n, vector<vector<int>>& requests) {
        int i = 0;
        int limit = 1 << requests.size();
        int answer = 0;
        while (++i < limit) {
            answer = max(answer, isPossible(requests, n, i));
        }
        return answer;
    }

    //107. Binary Tree Level Order Traversal II
    vector<vector<int>> levelOrderBottom(tree_node::TreeNode* root) {

        if (root == nullptr)return {};
        
        vector<vector<int>> levels;

        deque<tree_node::TreeNode*> level{ {root} };
        while (not level.empty()) {
            int size = level.size();
            levels.push_back({});
            while (0 < --size) {
                auto x = level.front(); level.pop_front();
                levels.back().push_back(x->val);
                if (x->left != nullptr)level.push_back(x->left);
                if (x->right != nullptr)level.push_back(x->right);
            }
        }

        reverse(levels.begin(), levels.end());

        return levels;
    }

    //16. 3Sum Closest
    //iterate over array with left and right pointer
    //find middle as : target - nums[l] + nums[r] <- with binary search
    //shift one of the pointers depending on previous search 
    int threeSumClosest(vector<int>& nums, int target) {
        sort(nums.begin(), nums.end());

        int l = 0,r = nums.size() - 1;
        while (l < r) {

        }

        return 1;
    }

    //1465. Maximum Area of a Piece of Cake After Horizontal and Vertical Cuts
    int maxArea(int h, int w, vector<int>& horizontalCuts, vector<int>& verticalCuts) {
        static const uint64_t modulo = 1000000007;

        sort(horizontalCuts.begin(), horizontalCuts.end());
        sort(verticalCuts.begin(), verticalCuts.end());
        
        horizontalCuts.push_back(h);
        verticalCuts.push_back(w);

        int previous = 0;
        uint64_t maxVertical = 0;
        for (int vc : verticalCuts) {
            if (maxVertical < vc - previous)maxVertical = vc - previous;
            previous = vc;
        }

        previous = 0;
        uint64_t maxHorizontal = 0;
        for (int hc : horizontalCuts) {
            if (maxHorizontal < hc - previous)maxHorizontal = hc - previous;
            previous = hc;
        }

        return (maxHorizontal * maxVertical) % modulo;
    }

    //57. Insert Interval
    vector<vector<int>> insert(vector<vector<int>>& intervals, vector<int>& newInterval) {

        auto left = lower_bound(intervals.begin(), intervals.end(), newInterval,
            [](const auto& l, const auto& r) {
                return l[1] < r[1];
        });

        if (left == intervals.end())--left;

        cout << (*left)[0] << "x" << (*left)[1] << "\n";

        return intervals;
    }

    //79. Word Search
    bool dfs(vector<vector<char>> & board,pair<uint16_t,uint16_t> pos,const string& word,size_t itr){
        const vector<pair<int, int>> directions = { {0, 1}, {1, 0}, {-1, 0}, {0, -1} };
        if (itr == word.length())return true;
 
        board[pos.first][pos.second] += 27;
        for (auto& direction : directions) {
            uint16_t ny = pos.first + direction.first;
            uint16_t nx = pos.second + direction.second;

            if (ny < board.size() && nx < board[0].size() && word[itr] == board[ny][nx])
                if(dfs(board, { ny,nx },word ,itr + 1))
                    return true;
        }
        board[pos.first][pos.second] -= 27;

        return false;
    }

    bool exist(vector<vector<char>>& board, string word) {
        const uint16_t ROWS = board.size();
        const uint16_t COLUMNS = board[0].size();

        for (uint16_t r = 0; r < ROWS; ++r) {
            for (uint16_t c = 0; c < COLUMNS; ++c) {
                if (board[r][c] == word[0]) {
                    if (dfs(board, { r,c }, word, 1))return true;
                }
            }
        }

        return false;
    }

    //86. Partition List
    //rewrite : two cycles for find heads and third for split
    ListNode* partition(ListNode* head, int x) {
        if (head == nullptr)return nullptr;

        ListNode* lower = nullptr, * lowerHead = nullptr;
        ListNode* higher = nullptr, * higherHead = nullptr;

        while (head != nullptr) {
            if (head->val < x) {
                if (lowerHead == nullptr)lower = lowerHead = head;
                else lower = lower->next = head;
            }
            else {
                if (higherHead == nullptr)higher = higherHead = head;
                else higher = higher->next = head;
            }
            head = head->next;
        }

        if (lowerHead == nullptr)return higherHead;
        lower->next = higherHead;
        if (higherHead != nullptr)higher->next = nullptr;

        return lowerHead;
    }

    //124. Binary Tree Maximum Path Sum
    int result124;
    int helper(tree_node::TreeNode* root) {
        //first -> max of straight path
        //second -> max of v-shaped path
        if (root == nullptr)return 0;
        int l = helper(root->left);
        int r = helper(root->right);
        int out = root->val + max(0, max(l, r));
        result124 = max(out, max(result124, root->val + l + r));
        return out;
    }

    int maxPathSum(tree_node::TreeNode* root) {
        result124 = INT_MIN;
        helper(root);
        root->left = root->right = 0;
        return result124;
    }

    int sumOddLengthSubarrays(vector<int>& arr) {
        const size_t SIZE = arr.size();
        int answer = 0;
        int sum = 0;

        for (int i = 0; i < SIZE; ++i) {
            answer += arr[i] * ((SIZE - i - 1) / 2) + 1;
        }

        return answer;
    }

    string reformatNumber(string number) {
        string numbers;
        for (char c : number) {
            if (isdigit(c))numbers += c;
        }

        number = "";
        size_t l = 0, r = numbers.length() - 1;

        while (l < r) {
            if (r - l == 3) {
                number += numbers.substr(l, 2) + "-" + numbers.substr(l + 2, 2);
                break;
            }
            else if (r - l == 1) {
                number += numbers.substr(l, 2);
                break;
            }
            else {
                number += numbers.substr(l, 3);
                if (l + 4 < r) number += '-';
                l += 3;
            }
        }

        return number;
    }

    //452. Minimum Number of Arrows to Burst Balloons
    int findMinArrowShots(vector<vector<int>> points) {
        sort(points.begin(), points.end(),
            [](const auto& l, const auto& r) {return l[1] < r[1]; }
        );
        const size_t SIZE = points.size();
        int answer = 1;
        auto& range = points[0];

        show_vector(points);

        for (int i = 1; i < SIZE; ++i) {
            if (range[1] <= points[i][1] && points[i][0] <= range[1]) {
                range[0] = points[i][0];
            }
            else {
                answer++;
                range = points[i];
            }
        }

        return answer;
    }

    //464. Can I Win
    bool canIWin(int maxChoosableInteger, int desiredTotal) {
        return false;
    }

    //2454. Next Greater Element IV
    //unsolved
    vector<int> secondGreaterElement(vector<int> nums) {
        int i = nums.size();

        if (i < 3) {
            return vector<int>(i, -1);
        }

        set<int> data{nums[--i],nums[--i]};
        nums[i] = nums[i+1] = -1;

        while (0 <= --i) {
            auto itr = upper_bound(data.begin(), data.end(), i+1);
            cout << nums[i] << " -> ";
            cout << (itr != data.end() ? *itr : -1) << "\n";
            data.insert(nums[i]);
        }

        return nums;
    }

    //316. Remove Duplicate Letters
    struct Node {
        char c;
        Node* previous;
        Node* next;
        Node(char c, Node* next = nullptr, Node* previous = nullptr):
            c(c),next(next),previous(previous)
        {};
    };

    class List {
    public:
        Node* head = nullptr;
        Node* tail = nullptr;

        void insert(Node* element){
            if (head == nullptr)tail = head = element;
            else {
                element->previous = tail;
                tail = tail->next = element;
                element->next = nullptr;
            }
        }

        void remove(Node* element) {
            if (element->previous == nullptr)head = element->next;
            else {
                element->previous->next = element->next;
            }
        }
    };

    string removeDuplicateLetters(string s) {
        //first -> current index from string,second -> index of previous 
        vector<Node*> letters(26, 0);
        List chars;

        for (int i = 0; i < s.length(); ++i) {
            Node* current = letters[s[i] - 'a'];
            if (current == nullptr) {
                chars.insert(new Node(s[i]));
                letters[s[i]-'a'] = chars.tail;
                continue;
            }

            if (current->previous == current->next)continue;
            
            if (current->previous == nullptr) {
               if (current->c > current->next->c) {
                   chars.remove(current);
                   chars.insert(current);
               }
            }
            else if (current->previous->c > chars.tail->c) {
                chars.remove(current);
                chars.insert(current);
            }
        }

        string result = "";

        return result;
    }

    //1475. Final Prices With a Special Discount in a Shop
    vector<int> finalPrices(vector<int> prices) {
        stack<pair<int, size_t>> data{ {{prices[0],0}} };
        
        const size_t SIZE = prices.size();
        for(size_t i = 1;i < SIZE;++i){
            while (not data.empty() && prices[i] <= data.top().first) {
                prices[data.top().second] = data.top().first - prices[i];
                data.pop();
            }
            data.push({ prices[i],i });
        }

        return prices;
    }

    //209. Minimum Size Subarray Sum
    int minSubArrayLen(int target, vector<int>& nums) {
        const size_t SIZE = nums.size();
        int l = 0, r = 0;
        int answer = INT_MAX;
        int sum = 0;
        while (r < SIZE) {
            if (target <= sum) {
                answer = min(r - l, answer);
                sum -= nums[l++];
            }
            else {
                sum += nums[r++];
            }
        }

        while (l < SIZE) {
            if (sum < target)break;
            answer = min(r - l, answer);
            sum -= nums[l++];
        }

        return answer == INT_MAX ? 0 : answer;
    }

};

//1146. Snapshot Array
bool comparer(const pair<uint16_t, int>& l, const pair<uint16_t, int>& r) {
    return l.first < r.first;
}

class SnapshotArray {
    const size_t size;
    vector<int> values;
    vector<vector<pair<uint16_t, int>>> data;
    uint16_t snapshots;
    static stack<uint16_t> indexes;
public:
    SnapshotArray(int length) : size(length) {
        values = vector<int>(length, 0);
        data = vector<vector<pair<uint16_t, int>>>(length);
        snapshots = 0;
    }

    void set(int index, int val) {
        values[index] = val;
        indexes.push(index);
    }

    int snap() {

        while (not indexes.empty()) {
            auto& k = indexes.top(); indexes.pop();
            auto& s = data[k];
            if (s.empty()) {
                s.push_back({ snapshots,values[k] });
            }
            else if (s.back().second != values[k]) {
                s.push_back({ snapshots,values[k] });
            }
        }

        return snapshots++;
    }

    int get(int index, int snap_id) {
        if (data[index].empty()) return 0;

        auto itr = lower_bound(data[index].begin(), data[index].end(), make_pair(snap_id, 0),comparer);

        if (itr == data[index].begin() && itr->first > snap_id) return 0;
        if (itr == data[index].end() || itr->first > snap_id) return (itr - 1)->second;
        return itr->second;
    }
};

//352. Data Stream as Disjoint Intervals
class SummaryRanges {
    set<int> data;
public:
    SummaryRanges() {

    }

    void addNum(int value) {
        data.insert(value);
    }

    vector<vector<int>> getIntervals() {
        if (data.empty())return {};

        vector<vector<int>> result = { {*data.begin(),*data.begin()} };
        auto itr = data.begin();
        while(++itr != data.end()){
            if (result.back()[1] + 1 == *itr)++result.back()[1];
            else result.push_back({ *itr,*itr });
        }
        return result;
    }
};

//2080. Range Frequency Queries
class RangeFreqQuery {
    map<int, vector<int>> data;
public:
    RangeFreqQuery(vector<int>& arr) {
        const size_t SIZE = arr.size();
        for (int i = 0; i < SIZE; ++i) {
            data[arr[i]].push_back(i);
        }
    }

    int query(int left, int right, int value) {
        vector<int>& indexes = data[value];
        if (indexes.size() == 0)return 0;
        int l = lower_bound(indexes.begin(), indexes.end(), left) - indexes.begin();
        int r = lower_bound(indexes.begin(), indexes.end(), right) - indexes.begin();
        if (r == indexes.size())--r;
        if (l == indexes.size())--l;

        return max(0, r - l - 1 + (left <= indexes[l]) + (indexes[r] <= right));
    }
};