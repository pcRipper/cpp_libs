#pragma once
#include "../../../GeneralLibs/Headers/includes.hpp";
#include "DataStructures.hpp"

class Solution {
    const vector<pair<int, int>> directions = { {0, 1}, {1, 0}, {-1, 0}, {0, -1} };
public:
    //724. Find Pivot Index
    int pivotIndex(vector<int>& nums) {
        const size_t size = nums.size();
        long long right = 0;
        for (int k = 0; k < size; ++k)right += nums[k];

        long long left = 0;
        for (int k = 0; k < size; ++k) {
            left += nums[k];
            if (left == right)return k;
            right -= nums[k];
        }
        return -1;
    }

    //1679. Max Number of K-Sum Pairs
    int maxOperations(vector<int>& nums, int k) {
        sort(nums.begin(), nums.end());
        int l = 0, r = nums.size() - 1;
        int count = 0;
        while (l < r) {
            if (nums[l] + nums[r] == k) {
                ++count;
                ++l;
                --r;
            }
            else if (nums[l] + nums[r] < k)++l;
            else --r;

        }
        return count;
    }

    //1732. Find the Highest Altitude
    int largestAltitude(vector<int>& gain) {
        int result = 0;
        int current = 0;
        int i = -1;
        while (++i < gain.size()) {
            current += gain[i];
            if (current > result)result = current;
        }
        return result;
    }

    //1207. Unique Number of Occurrences
    bool uniqueOccurrences(vector<int>& arr) {
        unordered_map<short, short> nums;
        for (size_t i = 0; i < arr.size(); ++i)++nums[arr[i]];

        unordered_map<short, bool> exists;
        for (auto& [f, s] : nums) {
            if (exists[s])return false;
            exists[s] = true;
        }

        return true;
    }

    //934. Shortest Bridge

    //can be implemented through the capes approach
    //struct Capes {
    //    uint16_t top;
    //    uint16_t bottom;
    //    uint16_t left;
    //    uint16_t right;

    //    Capes() {
    //        left = top = -1;
    //        right = bottom = 0;
    //    }

    //    void insert(pair<uint16_t, uint16_t> point) {
    //        if (top > point.first)top = point.first;
    //        if (bottom < point.first)bottom = point.first;
    //        if (left > point.second)left = point.second;
    //        if (right < point.second)right = point.second;
    //    }
    //};

    void dfs(vector<vector<int>>& grid, queue<pair<uint16_t, uint16_t>>& points, pair<uint16_t, uint16_t> pos) {
        grid[pos.first][pos.second] = 2;
        points.push(pos);

        for (auto& point : directions) {
            uint16_t y = point.first + pos.first;
            uint16_t x = point.second + pos.second;
            if (y < grid.size() && x < grid.size() && grid[y][x] == 1) {
                dfs(grid, points, { y, x });
            }
        }
    }

    int shortestBridge(vector<vector<int>>& grid) {
        queue<pair<uint16_t, uint16_t>> points;
        const uint16_t N = grid.size();
        for (uint16_t r = 0; r < N; ++r) {
            for (uint16_t c = 0; c < N; ++c) {
                if (grid[r][c] == 1) {
                    dfs(grid, points, { r, c });
                    goto ns;
                }
            }
        }
    ns:
        int iteration = -1;
        while (!points.empty()) {
            ++iteration;
            int size = points.size();
            while (size--) {
                auto pair = points.front(); points.pop();
                for (auto& dir : directions) {
                    uint16_t y = pair.first + dir.first;
                    uint16_t x = pair.second + dir.second;
                    if (y < N && x < N && grid[y][x] != 2) {
                        if (grid[y][x] == 1) return iteration;
                        points.push({ y, x });
                        grid[y][x] = 2;
                    }
                }
            }
        }
        return -1;
    }

    //347. Top K Frequent Elements
    vector<int> topKFrequent(vector<int>& nums, int k) {

        unordered_map<int, short> references;
        for (int k = 0; k < nums.size(); ++k) {
            ++references[nums[k]];
        }

        set<pair<short, int>> tree;
        for (auto& pair : references) {
            tree.insert({ pair.second,pair.first });
        }

        vector<int> result(k);
        int to_go = tree.size() - k;
        auto itr = tree.begin();
        while (to_go--)++itr;
        while (itr != tree.end()) {
            result[++to_go] = itr->second;
            ++itr;
        }

        return result;
    }

    //394. Decode String
    void insert(string& str, string to_add, int n) {
        while (n-- > 0) {
            str += to_add;
        }
    }

    string decodeString(string s) {
        string result = "";
        int k = 0;

        while (k < s.length()) {

            while (k < s.length() && isalpha(s[k]))result += s[k++];

            string digit_part = "";
            while (k < s.length() && isdigit(s[k])) digit_part += s[k++];

            int insertions = 1;
            string inner_data = "";
            while (k < s.length()) {
                ++k;
                if (s[k] == '[')++insertions;
                if (s[k] == ']')--insertions;
                if (insertions < 1)break;
                inner_data += s[k];
            }

            insert(result, decodeString(inner_data), digit_part == "" ? 1 : stoi(digit_part));
            ++k;
        }

        return result;
    }

    //735. Asteroid Collision
    vector<int> asteroidCollision(vector<int> asteroids) {
        const size_t size = asteroids.size();
        stack<int> stack;

        int k = 0;
        while (k < size) {
            if (stack.size() > 0 && stack.top() > 0 && asteroids[k] < 0) {
                if (abs(asteroids[k]) >= stack.top()) {
                    if (abs(asteroids[k]) == stack.top())++k;
                    stack.pop();
                }
                else ++k;
            }
            else stack.push(asteroids[k++]);
        }

        int i = stack.size();
        vector<int> result(stack.size());
        while (i-- > 0) {
            result[i] = stack.top();
            stack.pop();
        }

        return result;
    }

    //51. N-Queens
    vector<vector<string>> solveNQueens(int n) {
        return {};
    }

    vector<vector<int>> merge(vector<vector<int>>& intervals) {
        sort(intervals.begin(), intervals.end(), [](vector<int>& l, vector<int>& r) { return l[0] < r[0]; });

        vector<vector<int>> result = { intervals[0] };

        for (int k = 1; k < intervals.size(); ++k) {
            if (intervals[k][0] <= result.back()[1])result.back()[1] = max(intervals[k][1], result.back()[1]);
            else result.push_back(intervals[k]);
        }

        return result;
    }

    //2542. Maximum Subsequence Score
    void hoareSort(vector<int>& nums1, vector<int>& nums2, int l, int r) {
        if (r <= l)return;

        int x = nums1[(l + r) / 2];
        int i = l, j = r;

        while (i < j)
        {
            while (nums1[i] > x) i++;
            while (nums1[j] < x) j--;

            if (i <= j) {
                swap(nums1[i], nums1[j]);
                swap(nums2[i++], nums2[j--]);
            }
        }

        hoareSort(nums1, nums2, l, j);
        hoareSort(nums1, nums2, i, r);
    }

    long long maxScore(vector<int>& nums1, vector<int>& nums2, int k) {
        const size_t size = nums1.size();
        hoareSort(nums2, nums1, 0, size - 1);

        multiset<int> data;
        long long sum = 0;
        int j = 0;
        while (j < k) {
            data.insert(nums1[j]);
            sum += nums1[j++];
        }

        long long answer = sum * nums2[j - 1];
        while (j < size) {
            sum += nums1[j] - *data.begin();
            data.erase(data.begin());
            data.insert(nums1[j]);
            if (answer < nums2[j] * sum)answer = nums2[j] * sum;
            ++j;
        }

        return answer;
    }

    //328. Odd Even Linked List
    ListNode* oddEvenList(ListNode* head) {
        ListNode* oddHead = head, * odd = head;
        ListNode* evenHead = head->next, * even = head->next;

        bool isOdd = true;
        while (head) {
            if (odd) {
                odd = odd->next = head;
            }
            else {
                even = even->next = head;
            }
            isOdd = !isOdd;
            head = head->next;
        }

        odd->next = evenHead;
        even->next = nullptr;

        return oddHead;
    }

    //837. New 21 Game
    double new21Game(int n, int k, int maxPts) {
        vector<int> dp(n, 1);


        return 0;
    }

    //236. Lowest Common Ancestor of a Binary Tree
    void dfs(tree_node::TreeNode* root, const int val, vector<bool>*& pointer, int level = 0) {
        static vector<bool> path(99999);
        if (!root) return;
        if (root->val == val) {
            pointer = new vector<bool>(path.begin(), path.begin() + level);
            return;
        }
        path[level] = 0;
        dfs(root->left, val, pointer, level + 1);
        path[level] = 1;
        dfs(root->right, val, pointer, level + 1);
    }

    tree_node::TreeNode* lowestCommonAncestor(tree_node::TreeNode* root, tree_node::TreeNode* p, tree_node::TreeNode* q) {
        vector<bool>* path1, * path2;
        dfs(root, p->val, path1);
        dfs(root, q->val, path2);

        int k = 0;
        while (k < path1->size() && k < path2->size()) {
            if ((*path1)[k] != (*path2)[k])break;
            root = (*path1)[k] ? root->right : root->left;
        }

        delete path1, path2;
        return root;
    }

    //1161. Maximum Level Sum of a Binary Tree
    int maxLevelSum(tree_node::TreeNode* root) {
        queue<tree_node::TreeNode*> levels;

        levels.push(root);

        pair<int, int> data = { 1,0 };
        int current_level = 1;
        while (levels.size() > 0) {
            int width = levels.size();
            int local_sum = 0;
            while (0 < width--) {
                local_sum += levels.front()->val;
                levels.push(levels.front()->right);
                levels.push(levels.front()->left);
                levels.pop();
            }
            if (local_sum > data.second)data = { current_level,local_sum };
            ++current_level;
        }
        return data.first;
    }

    //216. Combination Sum III
    vector<vector<int>> combinationSum3(int k, int n, int from = 0) {
        vector<vector<int>> result;

        while (++from < 10) {
            if (from == n && k == 1)result.push_back({ from });
            else if (from < n && k > 1) {
                auto nested = combinationSum3(k - 1, n - from, from);
                if (!nested.empty()) {
                    for (auto& vars : nested) {
                        vars.push_back(from);
                        result.push_back(vars);
                    }
                }
            }
        }

        return result;
    }
    int helper(pair<int, int> length, pair<int, int> range, const vector<int>& cuts) {
        if (range.first > range.second)return 0;
        int index = distance(
            cuts.begin(),
            lower_bound(
                cuts.begin() + range.first,
                cuts.begin() + range.second,
                (length.first + length.second) >> 1
            )
        );
        //cout << index << "\n";
        return
            (length.second - length.first) +
            helper({ length.first,cuts[index] }, { range.first,index - 1 }, cuts) +
            helper({ cuts[index],length.second }, { index + 1,range.second }, cuts)
            ;
    }

    //1547. Minimum Cost to Cut a Stick
    int minCost(int n, vector<int> cuts) {
        sort(cuts.begin(), cuts.end());
        return helper({ 0,n }, { 0,cuts.size() - 1 }, cuts);
    }

    //437. Path Sum III
    int dfs(tree_node::TreeNode* root, unordered_map<long long, int>& counter, long long sum, int target) {
        if (root == nullptr)return 0;
        sum += root->val;
        int result = counter[sum - target];
        ++counter[sum];
        result += dfs(root->left, counter, sum, target) + dfs(root->right, counter, sum, target);
        --counter[sum];
        return result;

    }
    int pathSum(tree_node::TreeNode* root, int targetSum) {
        unordered_map<long long, int> counter{ {0,1} };
        return dfs(root, counter, 0, targetSum);
    }

    //841. Keys and Rooms
    bool canVisitAllRooms(vector<vector<int>>& rooms) {
        const size_t size = rooms.size();
        vector<bool> visited(size, false);
        int amount = 0;
        queue<int> to_visit;

        to_visit.push(0);

        while (!to_visit.empty()) {
            int iteration = to_visit.size();
            while (iteration--) {
                int key = to_visit.front(); to_visit.pop();
                if (visited[key])continue;
                visited[key] = true;
                amount++;
                for (int nk : rooms[key])to_visit.push(nk);
            }
        }

        return amount == size;
    }

    //739. Daily Temperatures
    vector<int> dailyTemperatures(vector<int>& temperatures) {
        static vector<size_t> holding(100000);
        const size_t size = temperatures.size();
        vector<int> result(size, 0);

        int hSize = 0;
        for (int k = 0; k < size; ++k) {

            while (0 < hSize) {
                if (temperatures[holding[hSize - 1]] >= temperatures[k])break;
                result[holding[hSize]] = k - holding[--hSize];
            }

            holding[hSize++] = k;
        }

        return result;
    }

    //1641. Count Sorted Vowel Strings
    int countVowelStrings(int n) {
        int a = 1, e = 1, i = 1, o = 1;

        for (int k = 0; k < n;)
            a += e += i += o += ++k;

        return a;
    }

    //151. Reverse Words in a String
    string reverseWords(string s) {
        string result;

        int r = s.length() - 1;

        while (0 <= r) {
            if (s[r] == ' ') {
                --r;
                continue;
            }
            int l = r;
            if (result.length() > 0)result += ' ';
            while (0 <= r && s[r] != ' ')--r;
            result += s.substr(r + 1, l - r);
        }

        return result;
    }

    //1091. Shortest Path in Binary Matrix
    int shortestPathBinaryMatrix(vector<vector<int>>& grid) {
        static const vector<pair<int, int>> directions = { {0,1}, {1,0}, {-1,0}, {0,-1}, {1,1}, {-1,1}, {1,-1}, {-1,-1} };
        queue<pair<uint16_t, uint16_t>> queue;
        const size_t N = grid.size();

        if (grid[0][0] != 0 || grid[N - 1][N - 1] != 0)return -1;

        queue.push({ 0,0 });
        grid[0][0] = 2;

        while (!queue.empty()) {
            int size = queue.size();
            while (0 < size--) {
                auto point = queue.front(); queue.pop();
                for (auto& pair : directions) {
                    uint16_t y = point.first + pair.first;
                    uint16_t x = point.second + pair.second;
                    if (y < N && x < N) {
                        if (grid[y][x] == 0) {
                            queue.push({ y,x });
                            grid[y][x] = grid[point.first][point.second] + 1;
                            continue;
                        }
                        if (grid[y][x] > 1) {
                            grid[y][x] = min(grid[y][x], grid[point.first][point.second] + 1);
                        }
                    }
                }
            }
        }

        return grid[N - 1][N - 1] - 1;
    }

    //540. Single Element in a Sorted Array
    int singleNonDuplicate(vector<int>& nums) {
        const size_t size = nums.size();
        int l = 0, r = nums.size() - 1;
        while (l < r) {
            int middle = (l + r) >> 1;
            if (nums[middle] == nums[middle + 1]) {
                if ((r - middle - 1) & 1)l = middle + 2;
                else r = middle - 1;
                continue;
            }
            else if (nums[middle] == nums[middle - 1]) {
                if ((middle - 1 - l) & 1)r = middle - 2;
                else l = middle + 1;
                continue;
            }
            return nums[middle];
        }
        return nums[l];
    }

    //1104. Path In Zigzag Labelled Binary Tree
    vector<int> pathInZigZagTree(int label) {
        int level = log2(label);
        vector<int> path(level + 1, 1);
        for (int num = 1 << level; 0 < level; --level) {

            path[level] = label;
            label = num - (label - num) / 2 - 1;

            num = num >> 1;
        }
        return path;
    }

    //2101. Detonate the Maximum Bombs
    int dfs(const vector<vector<bool>>& matrix, vector<bool>& visited, int edge) {
        int result = 1;
        visited[edge] = true;

        for (int k = 0; k < matrix.size(); ++k)
            if (matrix[edge][k] && !visited[k])
                result += dfs(matrix, visited, k);

        return result;
    }

    int maximumDetonation(vector<vector<int>>& bombs) {
        const size_t size = bombs.size();
        vector<vector<bool>> matrix(size, vector<bool>(size));

        for (char k = 0; k < size; ++k) {
            const uint64_t R2 = uint64_t(bombs[k][2]) * bombs[k][2];
            for (char j = 0; j < size; ++j) {
                matrix[k][j] = pow(bombs[k][0] - bombs[j][0], 2) + pow(bombs[k][1] - bombs[j][1], 2) <= R2;
            }
        }

        int result = 1;
        vector<bool> visited(size);
        for (int k = 0; k < size; ++k) {
            visited.assign(size, false);
            result = max(dfs(matrix, visited, k), result);
        }

        return result;
    }

    //1376. Time Needed to Inform All Employees
    int dfs(int index, const vector<vector<int>>& relations, const vector<int>& informTime, vector<int>& memo) {
        if (memo[index] != -1) {
            return memo[index];
        }

        int sum = 0;
        for (int k = 0; k < relations[index].size(); ++k) {
            sum = max(sum, dfs(relations[index][k], relations, informTime, memo));
        }

        memo[index] = informTime[index] + sum;
        return memo[index];
    }

    int numOfMinutes(int n, int headID, vector<int>& manager, vector<int>& informTime) {
        vector<vector<int>> relations(n);
        for (int k = 0; k < n; ++k) {
            if (0 <= manager[k]) {
                relations[manager[k]].push_back(k);
            }
        }

        vector<int> memo(n, -1);
        return dfs(headID, relations, informTime, memo);
    }

    //1466. Reorder Routes to Make All Paths Lead to the City Zero
    int count = 0;
    void dfs(int node, int parent, vector<vector<pair<int, bool>>>& adj) {

        for (auto& [child, sign] : adj[node]) {
            if (child != parent) {
                count += sign;
                dfs(child, node, adj);
            }
        }
    }

    int minReorder(int n, vector<vector<int>>& connections) {
        vector<vector<pair<int, bool>>> adj(n);
        for (auto& connection : connections) {
            adj[connection[0]].push_back({ connection[1], 1 });
            adj[connection[1]].push_back({ connection[0], 0 });
        }
        dfs(0, -1, adj);
        return count;
    }

    //875. Koko Eating Bananas
    int minEatingSpeed(vector<int> piles, int h) {
        int l = 1, r = *max_element(piles.begin(), piles.end());

        const size_t size = piles.size();
        while (l < r) {
            int middle = (r + l) / 2;
            int in_hours = 0;
            for (int k = 0; k < size; ++k)in_hours += (piles[k] + middle - 1) / middle;
            if (in_hours > h)l = middle + 1;
            else r = middle;
        }

        return l;
    }

    //790. Domino and Tromino Tiling
    int numTilings(int n) {
        static const int modulo = 1000000007;
        vector<size_t> dp(n, 1);
        for (int k = 1; k < n; ++k) {
            dp[k] = (dp[k - 1] + 2 + (k > 1)) % modulo;
        }
        return dp.back();
    }

    //1232. Check If It Is a Straight Line
    pair<int, int> coof(vector<int>& l, vector<int>& r) {
        int k = abs(l[0] - r[0]) / abs(l[1] - r[1]);
        return { k,l[1] - k * l[0] };
    }
    bool checkStraightLine(vector<vector<int>>& coordinates) {

        pair<int, int> data = coof(coordinates[0], coordinates[1]);

        const size_t size = coordinates.size();
        for (int k = 2; k < size; ++k)
            if (data != coof(coordinates[k], coordinates[k - 1]))return false;


        return true;
    }

    //125. Valid Palindrome
    inline bool isMatch(char k) { return isalpha(k) || isdigit(k); }
    bool isPalindrome(string s) {
        int l = 0, r = s.length() - 1;
        while (l < r) {
            while (l < r && !isMatch(s[l]))++l;
            while (l < r && !isMatch(s[r]))--r;
            if (tolower(s[l]) != tolower(s[r]))return false;
            ++l; --r;
        }
        return true;
    }

    //73. Set Matrix Zeroes
    void setZeroes(vector<vector<int>>& matrix) {
        const size_t R = matrix.size();
        const size_t C = matrix[0].size();
        vector<bool> rows(R);
        vector<bool> columns(C);

        for (int r = 0; r < R; ++r) {
            for (int c = 0; c < C; ++c) {
                if (matrix[r][c] == 0)columns[c] = rows[r] = true;
            }
        }

        for (int r = 0; r < R; ++r) {
            for (int c = 0; c < C; ++c) {
                if (columns[c] || rows[r])matrix[r][c] = 0;
            }
        }

    }

    //219. Contains Duplicate II
    bool containsNearbyDuplicate(vector<int>& nums, int k) {
        unordered_map<int, size_t> occurence;
        const size_t size = nums.size();

        for (size_t i = 0; i < size; ++i) {
            if (occurence[nums[i]] != 0 && i - occurence[nums[i]] <= k)return true;
            occurence[nums[i]] = i + 1;
        }

        return false;
    }

    //128. Longest Consecutive Sequence
    int longestConsecutive(vector<int>& nums) {
        const size_t size = nums.size();

        if (size < 1)return 0;

        sort(nums.begin(), nums.end());
        int result = 1;
        int nested = 1;
        int prev = nums[0] - 1;
        for (int k = 1; k < size; ++k) {
            if (nums[k] == prev)continue;
            if (nums[k] - prev == 1)++nested;
            else {
                if (result < nested)result = nested;
                nested = 1;
            }
            prev = nums[k];
        }
        return max(result, nested);
    }
    //150. Evaluate Reverse Polish Notation
    //template<char i>
    //struct Evaluator {
    //    static inline int evaluate(int l, int r) { return 0; }
    //};

    //template<> struct Evaluator<'*'> { static inline int evaluate(int l, int r) { return l * r; } };
    //template<> struct Evaluator<'-'> { static inline int evaluate(int l, int r) { return l - r; } };
    //template<> struct Evaluator<'+'> { static inline int evaluate(int l, int r) { return l + r; } };
    //template<> struct Evaluator<'/'> { static inline int evaluate(int l, int r) { return l / r; } };

    int evalRPN(vector<string>& tokens) {
        static vector<int> data(5001);
        size_t size = 0;

        for (string& s : tokens) {
            if ((s[0] == '-' && s.length() > 1) || isdigit(s[0])) {
                data[size++] = stoi(s);
                continue;
            }
            int r = data[--size];
            int* l = &data[size - 1];
            switch (s[0])
            {
            case '*':
                *l = *l * r;
                break;
            case '+':
                *l = *l + r;
                break;
            case '-':
                *l = *l - r;
                break;
            case '/':
                *l = *l / r;
                break;
            }
        }

        return data[0];
    }

    //1493. Longest Subarray of 1's After Deleting One Element
    int longestSubarray(vector<int>& nums) {
        const size_t size = nums.size();

        vector<int> zeroes = { -1 };
        for (int k = 0; k < size; ++k) {
            if (nums[k] == 0)zeroes.push_back(k);
        }
        zeroes.push_back(size);

        const size_t sizeZ = zeroes.size() - 1;
        for (int k = 1; k < sizeZ; ++k) {

        }

        return 0;
    }
    //39. Combination Sum
    vector<vector<int>> combinationSum(vector<int> candidates, int target) {
        return {};
    }

    //1926. Nearest Exit from Entrance in Maze
    int nearestExit(vector<vector<char>>& maze, const vector<int>& entrance) {
        static const vector<array<int, 2>> directions = { {0, 1}, {1, 0}, {-1, 0}, {0, -1} };

        const uint16_t N = maze.size() - 1;
        const uint16_t M = maze[0].size() - 1;
        deque<array<uint16_t, 2>> points;

        points.push_back({ uint16_t(entrance[0]),uint16_t(entrance[1]) });
        maze[entrance[0]][entrance[1]] = '+';

        int iterations = 0;
        while (not points.empty()) {
            ++iterations;

            int size = points.size();
            while (0 < size--) {
                {
                    auto& point = points.front();

                    for (auto& [y1, x1] : directions) {
                        uint16_t yn = y1 + point[0];
                        uint16_t xn = x1 + point[1];
                        if (yn <= N && xn <= M && maze[yn][xn] == '.') {
                            if (yn == 0 || xn == 0 || yn == N || xn == M)return iterations;
                            maze[yn][xn] = '+';
                            points.push_back({ yn,xn });
                        }
                    }
                }
                points.pop_front();
            }
        }

        return -1;
    }

    int subarraySum(vector<int>& nums, int k) {
        unordered_map<int, int> sums;
        uint64_t sum = 0;
        const size_t size = nums.size();
        for (size_t k = 0; k < size; ++k) {
            ++sums[nums[k]];
            if (sums[sum - k]) {
                ++sums[k];
            }
        }
        return sums[k];
    }

    //1351. Count Negative Numbers in a Sorted Matrix
    int binaryNegative(const vector<int>& row) {
        int l = 0, r = row.size() - 1;
        while (l < r) {
            int middle = (r + l) >> 1;
            if (row[middle] < 0)r = middle;
            else l = middle + 1;
        }
        return row[l] < 0 ? l : row.size();
    }

    //int countNegatives(vector<vector<int>>& grid) {
    //    const size_t size = grid.size();
    //    const int    sizeC = grid[0].size();

    //    int result = 0;
    //    for (int k = 0; k < size; ++k) {
    //        result += sizeC - binaryNegative(grid[k]);
    //    }
    //    return result;
    //}

    int countNegatives(vector<vector<int>>& grid) {
        const size_t R = grid.size();
        const size_t C = grid[0].size() - 1;

        int result = 0;
        int index = C;
        for (int k = 0; k < R; ++k) {
            while (0 <= index && grid[k][index] < 0)--index;
            result += C - index;
        }

        return result;
    }

    //523. Continuous Subarray Sum
    bool checkSubarraySum(vector<int>& nums, int k) {
        const size_t size = nums.size();

        if (size < 2)return false;

        unordered_map<size_t, size_t> used({ 0,0 });
        long long sum = 0;

        for (size_t i = 0; i < size; ++i) {
            sum += nums[i];
            size_t leftover = sum % k;
            if (used.find(leftover) == used.end())used[leftover] = i + 1;
            if (i > used[leftover])return true;
        }

        return false;
    }

    //127. Word Ladder
    array<char, 26>* split(const string& word) {
        array<char, 26>* data = new array<char, 26>{};
        for (char c : word)++(*data)[c - 'a'];
        return data;
    }

    bool different(const array<char, 26>& a1, const array<char, 26>& a2) {
        short diff = 0;
        size_t l = 0, r = 25;
        while (l < r) {
            diff += abs(a1[l] - a2[l]) + abs(a1[r] - a2[r]);
            if (diff > 1)return false;
            ++l; --r;
        }
        return true;
    }

    int ladderLength(string beginWord, string endWord, vector<string> wordList) {

        const size_t size = wordList.size();
        vector<array<char, 26>*> wordSplited(26, 0);
        vector<bool> visited(size, false);
        for (int k = 0; k < size; ++k) wordSplited[k] = split(wordList[k]);

        array<char, 26>* begin = split(beginWord), * end = split(endWord);
        deque<array<char, 26>*> queue;

        queue.push_back(begin);

        int iterations = 0;
        int qSize = 0;
        while (not queue.empty()) {
            ++iterations;
            qSize = queue.size();

            while (0 < qSize--) {
                auto top = queue.front(); queue.pop_front();

                if (different(*top, *end))return iterations;

                for (int k = 0; k < size; ++k) {
                    if (visited[k])continue;
                    if (different(*top, *wordSplited[k])) {
                        visited[k] = true;
                        queue.push_back(wordSplited[k]);
                    }
                }
            }
        }
        return 0;
    }

    //744. Find Smallest Letter Greater Than Target
    char nextGreatestLetter(vector<char>& letters, char target) {
        if (letters.back() <= target)return letters.front();
        return *lower_bound(letters.begin(), letters.end(), target + 1);
    }

    //450. Delete Node in a BST
    //pair<tree_node::TreeNode*, tree_node::TreeNode*> find(tree_node::TreeNode* root, int target) {


    //}
    //tree_node::TreeNode* deleteNode(tree_node::TreeNode* root, int key) {
    //    auto node = find(root, key);
    //    return root;
    //}

    //1802. Maximum Value at a Given Index in a Bounded Array
    int64_t sum(int64_t len, int64_t n) {
        if (n <= len)return (n * n + n) / 2 + (len - n);
        return (2 * n - len + 1) * len / 2;
    }
    int maxValue(int n, int index, int maxSum) {

        n -= ++index;
        int l = 1, r = maxSum / 2;
        while (l < r) {
            int middle = (r + l + 1) / 2;
            int64_t middleSum = sum(index, middle) + sum(n, middle - 1);
            if (middleSum > maxSum)r = middle - 1;
            else l = middle;

            if (middleSum < maxSum && maxSum - middleSum < n + index)break;
        }

        return l;
    }
};



class KthLargest {
    priority_queue<int, vector<int>, greater<int>> heap;
    int K;
public:
    KthLargest(int k, vector<int>& nums) {
        K = k;
        for (int i = 0; i < nums.size(); ++i) heap.push(nums[i]);

        while (heap.size() > k)heap.pop();
    }

    int add(int val) {
        if (heap.size() < K) {
            heap.push(val);
        }
        else if (val > heap.top()) {
            heap.push(val);
            heap.pop();
        }
        return heap.top();
    }
};

struct Node {
    int val;
    Node* next;
    Node(int val) :val(val) {};
    ~Node() {
        next = nullptr;
    }
};

class RecentCounter {
    list<int> segment;
public:
    RecentCounter() {
        segment = {};
    }

    int ping(int t) {
        segment.push_back(t);
        while (segment.front() + 3000 < t)segment.pop_front();
        return segment.size();
    }
};

struct Line {
    vector<Line*> links;
    bool end;

    Line() {
        links = vector<Line*>(26, nullptr);
        end = false;
    }

    void insert(const string& word, int index) {
        if (word.length() <= index) {
            end = true;
            return;
        }
        if (links[word[index] - 'a'] == nullptr)links[word[index] - 'a'] = new Line();
        links[word[index] - 'a']->insert(word, index + 1);
    }

    Line* prefix(const string& word, int index) {
        if (word.length() == index)
            return this;
        if (links[word[index] - 'a'] != nullptr)
            return links[word[index] - 'a']->prefix(word, index + 1);
        return nullptr;
    }
};

class Trie {
    Line root;
public:
    Trie() {
        root = {};
    }

    void insert(string word) {
        root.insert(word, 0);
    }

    bool search(string word) {
        Line* last = root.prefix(word, 0);
        if (last == nullptr)return false;
        return last->end;
    }

    bool startsWith(string prefix) {
        return root.prefix(prefix, 0);
    }
};

//1603. Design Parking System
class ParkingSystem {
    vector<int> places;
public:
    ParkingSystem(int big, int medium, int small) {
        places = { big,medium,small };
    }

    bool addCar(int carType) {
        if (places[carType - 1]) {
            --places[carType - 1];
            return true;
        }
        return false;
    }
};

//901. Online Stock Span
//struct MultiNode {
//    int val;
//    int childrenL;
//    MultiNode* left;
//    MultiNode* right;
//
//    MultiNode(int val) :
//        val(val),
//        childrenL(1),
//        left(nullptr),
//        right(nullptr)
//    {};
//};
//
//class MultiSet {
//    MultiNode* root;
//public:
//    int insert(int val) {
//        
//        if (root == nullptr) {
//            root = new MultiNode(val);
//            return 0;
//        }
//
//        MultiNode* itr = root;
//        MultiNode* previous = nullptr;
//        int left = 0;
//
//        while (itr != nullptr) {
//            previous = itr;
//            if (val == itr->val) {
//                return itr->childrenL++;
//            }
//            if (val > itr->val) {
//                left += itr->childrenL;
//                itr = itr->right;
//            }
//            else {
//                ++itr->childrenL;
//                itr = itr->left;
//            }
//        }
//
//        if (previous->val < val) previous->right = new MultiNode(val);
//        else previous->left = new MultiNode(val);
//
//        return left;
//    }
//};

class StockSpanner {
    static vector<int> data;
    int elements;
public:
    StockSpanner() {
        elements = 0;
    }

    int next(int price) {
        int amount = 1;
        for (int k = elements - 1; 0 <= k; ++k) {
            if (data[k] > price)break;
            ++amount;
        }
        data[elements++] = price;
        return amount;
    }
};

vector<int> StockSpanner::data = vector<int>(100000);

//705. Design HashSet
struct SetNode {
    int val;
    bool useable;
    SetNode* left;
    SetNode* right;

    SetNode(int val) {
        this->val = val;
        left = right = nullptr;
        useable = true;
    }
};

class MyHashSet {
    SetNode* root;
public:
    MyHashSet() {
        root = nullptr;
    }

    void add(int key) {
        SetNode* res = search(key);
        if (res == nullptr) {
            root = new SetNode(key);
        }
        else if (res->val != key) {
            if (res->val < key)res->right = new SetNode(key);
            else res->left = new SetNode(key);
        }
        else res->useable = true;
    }

    void remove(int key) {
        SetNode* res = search(key);
        if (res != nullptr && res->val == key) {
            res->useable = false;
        }
    }

    bool contains(int key) {
        SetNode* res = search(key);
        if (res == nullptr)return false;
        return res->val == key && res->useable;
    }
private:
    SetNode* search(int val) {
        SetNode* iter = root;
        SetNode* prev = nullptr;

        while (iter != nullptr) {
            prev = iter;
            if (iter->val == val)return iter;
            if (iter->val < val)iter = iter->right;
            else iter = iter->left;
        }

        return prev;
    }
};

//1396. Design Underground System

struct MyNode {
    int id;
    string from;
    int time;
    MyNode* left;
    MyNode* right;

    MyNode(int id, string from, int time)
        :id(id),
        from(from),
        time(time),
        left(nullptr),
        right(nullptr)
    {};
};

class MySet {
    MyNode* root = nullptr;
public:
    MyNode* search(int id) {
        MyNode* itr = root;
        MyNode* previous = nullptr;

        while (itr != nullptr) {
            previous = itr;
            if (itr->id == id)break;
            itr = itr->id < id ? itr->right : itr->left;
        }

        return previous;
    }
    void insert(int id, string from, int time) {
        if (root == nullptr) {
            root = new MyNode(id, from, time);
            return;
        }

        MyNode* current = this->search(id);

        if (current->id == id) {
            current->from = from;
            current->time = time;
            return;
        }
        if (current->id < id)current->right = new MyNode(id, from, time);
        else current->left = new MyNode(id, from, time);
    }
};

class UndergroundSystem {
    MySet set;
    unordered_map<string, pair<double, int>> avg_time;
public:
    UndergroundSystem() {}

    void checkIn(int id, string stationName, int t) {
        MyNode* res = set.search(id);
        set.insert(id, stationName, t);
    }

    void checkOut(int id, string stationName, int t) {
        MyNode* res = set.search(id);
        avg_time[res->from + " " + stationName].first += t - res->time;
        avg_time[res->from + " " + stationName].second += 1;
    }

    double getAverageTime(string startStation, string endStation) {
        auto& p = avg_time[startStation + " " + endStation];
        return p.first / p.second;
    }
};

class RandomizedSet {
    unordered_map<int, int> presence;
    vector<int> data;
public:
    RandomizedSet() {
        srand(time(0));
    }

    bool insert(int val) {
        if (presence[val])return false;

        data.push_back(val);
        presence[val] = data.size();

        return true;
    }

    bool remove(int val) {
        if (presence[val] == 0)return false;

        int num = data.back();

        data[presence[val] - 1] = data.back();
        presence[num] = presence[val];

        data.pop_back();
        presence[val] = 0;

        return true;
    }

    int getRandom() {
        return data[rand() % data.size()];
    }
};

//155. Min Stack
class MinStack {
    vector<uint16_t> min;
    vector<int> stack;
public:
    MinStack() {}

    void push(int val) {
        stack.push_back(val);
        if (min.empty() || stack[min.back()] > val) {
            min.push_back(stack.size() - 1);
        }
    }

    void pop() {
        stack.pop_back();
        if (min.back() == stack.size()) {
            min.pop_back();
        }
    }

    inline int top() {
        return stack.back();
    }

    inline int getMin() {
        return stack[min.back()];
    }
};