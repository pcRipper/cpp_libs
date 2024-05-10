#pragma once
#include "../../../GeneralLibs/Headers/includes.hpp";
#include "DataStructures.hpp"

class Solution {
public:
    //1647. Minimum Deletions to Make Character Frequencies Unique
    int minDeletions(string s) {
        static int frequency[26];

        const int len = s.length();
        for (int i = 0; i < len; ++i) {
            ++frequency[s[i] - 'a'];
        }

        map<int, int,greater<int>> data;
        for (int i = 0; i < 26; ++i) {
            ++data[frequency[i]];
            frequency[i] = 0;
        }

        int answer = 0;

        for (auto&[x, y] : data) {
            if (x == 0)break;
            if (y == 1)continue;
            answer += y - 1;
            data[x - 1] += y - 1;
        }

        return answer;
    }

    //135. Candy
    int candy(vector<int>& ratings) {
        vector<int> data(ratings.size(), 1);

        for (int i = 0; i < ratings.size(); ++i) {
            
        }

        return 0;
    }

    //792. Number of Matching Subsequences
    int numMatchingSubseq(const string& s, vector<string>& words) {

        unordered_map<char, list<pair<int, int>>> letters;

        for (int i = 0; i < words.size(); ++i) {
            letters[words[i][0]].push_back({0,i});
        }

        int amount = 0;
        for (int i = 0; i < s.length() && amount < words.size(); ++i) {
            auto data = &letters[s[i]];
            const int size = data->size();
            for (int i = 0; i < size; ++i) {
                auto top = data->front(); data->pop_front();
                if (++top.first == words[top.second].length()) {
                    amount += 1;
                    continue;
                }
                letters[words[top.second][top.first]].push_back(top);
            }
        }

        return amount;
    }

    //332. Reconstruct Itinerary
    vector<string> result;

    void dfs(unordered_map<string, vector<pair<string, bool>>>& connections, string current, int amount) {
        static vector<string> path;

        path.push_back(current);

        if (amount == 0) {
            result = path;
            path.clear();
            return;
        }

        for (pair<string, bool>& next : connections[current]) {
            if (next.second)continue;
            next.second = true;
            dfs(connections, next.first, amount - 1);
            if (result.size() != 0) return;
            next.second = false;
        }

        path.pop_back();
    }


    vector<string> findItinerary(vector<vector<string>> tickets) {

        unordered_map<string, vector<pair<string, bool>>> connections;

        result.clear();

        for (const auto& ticket : tickets) {
            connections[ticket[0]].push_back({ ticket[1],0 });
        }

        for (auto& [x, y] : connections) {
            sort(y.begin(), y.end());
        }

        dfs(connections, "JFK", tickets.size());

        return result;
    }

    //1584. Min Cost to Connect All Points
    int minCostConnectPoints(vector<vector<int>> points) {
        Unions unions(points.size());
        map<int, vector<pair<int, int>>,greater<int>> nums;

        for (int i = 0; i < points.size(); ++i) {
            for (int j = i + 1; j < points.size(); ++j) {
                nums[abs(points[i][0] - points[j][0]) + abs(points[i][1] - points[j][1])].push_back({i,j});
            }
        }

        int sum = 0;
        int n = 1;
        while (n < points.size()) {
            ++n;
        }

        return sum;
    }

    //30. Substring with Concatenation of All Words
    vector<int> findSubstring(string s, vector<string>& words) {
        //map all words : unordered_map<string,int> -> key(string),value(count of entraces)
        // custom hashing function?(depends only on chars, dont mind the positons)
        //create whole check sum
        //if check sum is matching - try to split up
        return {};
    }

    //218. The Skyline Problem
    vector<vector<int>> getSkyline(vector<vector<int>> buildings) {

        static priority_queue<pair<int, int>, vector<pair<int, int>>, less<pair<int, int>>> queue;

        sort(buildings.begin(), buildings.end(), [](const auto& l, const auto& r) {
            if (l[0] != r[0])return true;
            return l[1] < r[1];
        });

        vector<vector<int>> result;
        for (const auto& building : buildings) {
            while (queue.size() != 0 && building[0] > queue.top().second) {
                auto top = queue.top(); queue.pop();
                while (queue.size() != 0 && queue.top().second <= top.second)queue.pop();
                if (result.back()[0] > top.second)continue;
                result.push_back({top.second,queue.size() == 0 ? 0 : queue.top().first});
            }
            if (queue.size() == 0 || queue.top().first < building[2]) {
                result.push_back({ building[0],building[2] });
            }
            queue.push({ building[2],building[1] });
        }

        while (queue.size() != 0) {
            auto top = queue.top(); queue.pop();
            while (queue.size() != 0 && queue.top().second <= top.second)queue.pop();
            result.push_back({ top.second, queue.size() == 0 ? 0 : queue.top().first });
        }

        return result;
    }

    //1043. Partition Array for Maximum Sum
    int maxSumAfterPartitioning(vector<int>& arr, int k) {
        

        return 0;
    }

    //1658. Minimum Operations to Reduce X to Zero
    int minOperations(vector<int>& nums, int x) {
        if (x == 0)return nums.size();

        ios::sync_with_stdio(false);
        cin.tie(0);

        const int N = nums.size();
        int sum = 0;
        for (int i = 0; i < N; ++i) {
            sum += nums[i];
        }

        if (sum < x)return -1;

        int has_to_be = sum - x;
        int l = 0, r = 0;

        int current_sum = 0;
        int answer = 100001;
        while (l < N) {
            while (current_sum < has_to_be && r < N) {
                current_sum += nums[r++];
            }
            if (current_sum == has_to_be) {
                answer = min(answer, N - r + l);
            }
            current_sum -= nums[l++];
        }

        return answer == 100001 ? -1 : answer;
    }

    //1027. Longest Arithmetic Subsequence
    int longestArithSeqLength(vector<int>& nums) {
        const size_t SIZE = nums.size();

        vector<unordered_map<int, int>> dp(nums.size());

        int answer = 0;
        for (int k = 0; k < nums.size(); ++k) {
            for (int j = k + 1; j < nums.size(); ++j) {
                int diff = nums[j] - nums[k];
                dp[j][diff] = max(dp[k][diff] + 1,dp[j][diff]);
                answer = max(answer, dp[j][diff]);
            }
        }

        return answer + 1;
    }

    //2799. Count Complete Subarrays in an Array
    int countCompleteSubarrays(vector<int>& nums) {
        static unordered_map<int, int> window;
        unordered_map<int, bool> uniques;

        const int SIZE = nums.size();
        for (int i = 0; i < SIZE; ++i) {
            uniques[nums[i]] = true;
        }

        int l = 0, r = 0;
        int answer = 0;
        while (l < SIZE) {
            while (r < SIZE && window.size() < uniques.size()) {
                ++window[nums[r++]];
            }
            if (window.size() == uniques.size()) {
                answer += SIZE - r + 1;
            }
            --window[nums[l]];
            if (window[nums[l]] == 0) {
                window.erase(nums[l]);
            }
            ++l;
        }
        return answer;
    }

    //2342. Max Sum of a Pair With Equal Sum of Digits
    int sumUp(int x) {
        int sum = 0;
        while (x != 0) {
            sum += x % 10;
            x /= 10;
        }
        return sum;
    }

    int maximumSum(vector<int>& nums) {
        unordered_map<int, int> data;

        int result = 0;
        const int SIZE = nums.size();
        for (int i = 0; i < SIZE; ++i) {
            int sum = sumUp(nums[i]);
            if (data[sum] != 0) {
                result = max(result, nums[i] + data[sum]);
            }
            data[sum] = max(data[sum], nums[i]);
        }

        return result == 0 ? -1 : result;
    }

    //450. Delete Node in a BST
    pair<tree_node::TreeNode*, tree_node::TreeNode*> find(tree_node::TreeNode* root, int key) {
        tree_node::TreeNode* prev = nullptr;

        while (root != nullptr) {
            if (root->val == key)break;
            prev = root;
            root = root->val < key ? root->right : root->left;
        }

        return { prev,root };
    }

    pair<tree_node::TreeNode*, tree_node::TreeNode*> nodeMin(pair<tree_node::TreeNode*, tree_node::TreeNode*> l, pair<tree_node::TreeNode*, tree_node::TreeNode*> r, const int value) {
        if (l.second == nullptr)return r;
        if (r.second == nullptr)return l;
        return abs(value - l.second->val) < abs(value - r.second->val) ? l : r;
    }

    pair<tree_node::TreeNode*, tree_node::TreeNode*> minDifference(tree_node::TreeNode* root, const int value) {
        if (root == nullptr)return { 0,0 };
        auto l = minDifference(root->left, value);
        auto r = minDifference(root->right, value);
        if (l.second == r.second)return { 0,root };
        auto res = nodeMin({ 0,root }, nodeMin(l, r, value), value);
        return res.first == nullptr ? make_pair(root, res.second) : res;
    }

    tree_node::TreeNode* deleteNode(tree_node::TreeNode* root, int key) {

        if (root == nullptr)return nullptr;
        if (root->left == root->right && root->val == key)return nullptr;

        auto itr = find(root, key);
        if (itr.second == nullptr)return root;

        while (true) {
            auto next = nodeMin(
                minDifference(itr.second->left, itr.second->val),
                minDifference(itr.second->right, itr.second->val),
                itr.second->val
            );
            if (next.second == nullptr)break;
            if (next.first == nullptr) {
                next.first = itr.second;
            }
            itr.second->val = next.second->val;
            itr = next;
        }

        if (itr.first->left == itr.second)itr.first->left = nullptr;
        else itr.first->right = nullptr;

        return root;
    }
    
    //1048. Longest String Chain
    bool isSimilar(const string& small, const string& big) {

        if (small.length() != big.length() - 1)return false;

        int s = 0, b = 0;
        while (b < big.length()) {
            if (s + 1 < b)return false;
            if (small[s] == big[b])++s;
            ++b;
        }

        return s + 1 == b;
    }

    int longestStrChain(vector<string>& words) {
        map<int, vector<pair<string*, int>>> data;

        for (int i = 0; i < words.size(); ++i) {
            data[words[i].length()].push_back({ &words[i],0 });
        }

        int answer = 0;
        for (const auto& [len, ws] : data) {
            for (const auto& pair : ws) {
                for (auto& [str, dp] : data[len + 1]) {
                    if (pair.second >= dp && isSimilar(*pair.first, *str)) {
                        dp = pair.second + 1;
                        answer = max(dp, answer);
                    }
                }
            }
        }

        return answer + 1;
    }

    //775. Global and Local Inversions
    bool isIdealPermutation(vector<int>& nums) {
        const int SIZE = nums.size() - 1;
        for (int i = 0; i < SIZE; ++i) {
            if (nums[i] == i)continue;
            if (nums[i] != i + 1 || nums[i + 1] != i)return false;
            swap(nums[i], nums[i + 1]);
        }
        return true;
    }

    //799. Champagne Tower
    double champagneTower(int poured,int query_row,int query_glass) {
        static float tower[101][101];

        tower[0][0] = poured;
        for (int r = 0; r <= query_row; ++r) {
            tower[r + 1][0] = 0;
            for (int c = 0; c <= r; ++c) {
                tower[r + 1][c + 1] = 0;
                if (tower[r][c] < 1.0)continue;
                tower[r + 1][c] += (tower[r][c] - 1) / 2;
                tower[r + 1][c + 1] += (tower[r][c] - 1) / 2;
            }
        }

        return min(1.0f, tower[query_row][query_glass]);
    }

    //23. Merge k Sorted Lists
    class ListComparer {
    public:
        bool operator()(const ListNode* l, const ListNode* r) const {
            return l->val > r->val;
        }
    };

    ListNode* mergeKLists(vector<ListNode*>& lists) {
        static priority_queue<ListNode*, vector<ListNode*>, ListComparer> queue;

        for (const auto& node : lists) {
            if (node == nullptr)continue;
            queue.push(node);
        }

        ListNode* head = new ListNode(0);
        ListNode* itr = head;
        while (queue.size() != 0) {
            itr = itr->next = queue.top();
            queue.pop();
            if (itr->next == nullptr)continue;
            queue.push(itr->next);
        }

        return head->next;
    }

    //2817. Minimum Absolute Difference Between Elements With Constraint
    int minAbsoluteDifference(vector<int>& nums, int x) {
        static map<int, int> sorted;

        for (int i = x; i < nums.size(); ++i) {
            ++sorted[nums[i]];
        }

        int result = INT_MAX;
        int i = 0;
        while (sorted.size() != 0) {
            auto itr = sorted.lower_bound(nums[i]);
            if (itr == sorted.end())--itr;
            result = min(result, abs(nums[i] - itr->first));
            if (itr != sorted.begin()) {
                --itr;
                result = min(result, abs(nums[i] - itr->first));
            }
            if (--sorted[nums[x]] == 0) {
                sorted.erase(nums[x]);
            }
            ++x;
            ++i;
        }

        return result;
    }

    //2768. Number of Black Blocks
    struct PointHash {
        size_t operator()(const std::pair<int, int>& p) const {
            return (static_cast<size_t>(p.first) << 32) + static_cast<size_t>(p.second);
        }
    };

    vector<long long> countBlackBlocks(const int m,const int n, vector<vector<int>> coordinates) {
        static vector<long long> result(5);
        static const vector<pair<int, int>> directions = { {0,0},{-1,-1},{-1,0},{0,-1} };
        result = { 0,0,0,0,0 };

        unordered_map<pair<int, int>, bool, PointHash> boxes, black;

        for (const auto& coordinate : coordinates) {
            black[{coordinate[0], coordinate[1]}] = true;
        }

        for (const auto& [coordinate, _] : black) {
            for (const auto& pair : directions) {
                int nx = coordinate.first + pair.first;
                int ny = coordinate.second + pair.second;
                if (0 <= nx && 0 <= ny && nx + 1 < m && ny + 1 < n && boxes.find({nx,ny}) == boxes.end()) {
                    ++result[
                        int(black.find({nx,ny}) == black.end())+ 
                        int(black.find({nx + 1,ny + 1}) == black.end())+ 
                        int(black.find({nx + 1,ny}) == black.end())+ 
                        int(black.find({nx,ny + 1}) == black.end()) 
                    ];
                    boxes[{nx, ny}] = true;
                }
            }
        }

        result[0] = (n - 1) * (m - 1) - boxes.size();

        return result;
    }

    //386. Lexicographical Numbers
    void dfs(int current, int n,vector<int>& result) {
        if (current > n)return;
        result.push_back(current);
        current *= 10;
        for (int i = 0; i < 10; ++i) {
            dfs(current + i, n, result);
        }
    }

    vector<int> lexicalOrder(int n) {
        vector<int> result;
        result.reserve(n);

        dfs(1, n, result);
        
        return result;
    }

    //880. Decoded String at Index
    string decodeAtIndex(string s, int k) {
        static pair<string, size_t> stack[51] = { {"",0} };
        static int size;

        stack[1].second = 0;
        size = 1;

        int i = 0;
        while (true) {

            stack[size].first.clear();
            while (i < s.length() && !isdigit(s[i])) {
                stack[size].first += s[i++];
            }

            size_t repeat = 1;
            while (i < s.length() && isdigit(s[i])) {
                repeat *= (s[i++] - '0');
            }
            
            stack[size].second += stack[size].first.length();
            stack[size + 1].second = repeat * stack[size].second;

            if (k <= stack[size].second)break;
            ++size;
        }
        
        //show_array(&stack[1], size);
        while (size > 0) {
            if (k <= stack[size].second && k >= stack[size].second - stack[size].first.length()) {
                return string(1, stack[size].first[k - stack[size].second + stack[size].first.length() - 1]);
            }
            k %= stack[size--].second;
        }
        
        return string(1,stack[1].first[k-1]);
    }

    //984. String Without AAA or BBB
    string strWithout3a3b(int a, int b) {
        string result;
        result.reserve(a + b);

        while (a > 1 && b > 1) {
            if (abs(a - b) < 2) {
                result += "ab";
                a -= 1;
                b -= 1;
                continue;
            }
            if (a > b) {
                result += "aab";
                a -= 2;
                --b;
            }
            else {
                result += "bba";
                b -= 2;
                --a;
            }
        }
        result += string(a, 'a');
        result += string(b, 'b');

        return result;
    }

    //1024. Video Stitching
    int videoStitching(vector<vector<int>> clips, int time) {
        sort(clips.begin(), clips.end(), [](const auto& l, const auto& r) {
            return l[0] < r[0];
        });

        int prev = 0;
        int local_max = 0;
        int amount = 1;
        for (const auto& clip : clips) {
            if (prev < clip[0]) {
                if (local_max < clip[0])return -1;
                prev = local_max;
                ++amount;
            }
            local_max = max(local_max, clip[1]);
            if (time <= local_max)return amount;
        }

        return -1;
    }

    //22. Generate Parentheses
    vector<string> generateParenthesis(int n) {
        static int last = 0;
        static vector<unordered_map<string, bool>> memo(100);
        memo[0][""] = true;

        for (int i = last + 1; i <= n; ++i) {
            for (const auto& [k, v] : memo[i - 1]) {
                memo[i][k + "()"] = true;
                memo[i]["()" + k] = true;
                memo[i]["(" + k + ")"] = true;
            }
            int l = 0,r = i;
            while (++l <= --r) {
                for (const auto& [k, v] : memo[l]) {
                    for (const auto& [nk, nv] : memo[r]) {
                        memo[i][k + nk] = true;
                        memo[i][nk + k] = true;
                    }
                }
            }
        }

        if (last < n)last = n;

        vector<string> result;
        result.reserve(memo[n].size());
        for (const auto& [k, v] : memo[n]) {
            result.push_back(k);
        }
        return result;
    }

    //456. 132 Pattern
    bool find132pattern(vector<int> nums) {
        static pair<int, int> monotonic[200000];
        static int msize;

        msize = 0;

        const int SIZE = nums.size();

        for (int i = 0; i < SIZE; ++i) {
            int min_one = nums[i];
            while (msize != 0 && monotonic[msize-1].first < nums[i]) {
                min_one = min(min_one, monotonic[--msize].second);
            }
            if (msize != 0 && monotonic[msize-1].second < nums[i] && nums[i] < monotonic[msize-1].first) {
                return true;
            }
            monotonic[msize++] = { nums[i], min_one };
        }

        return false;
    }

    //1996. The Number of Weak Characters in the Game
    int numberOfWeakCharacters(vector<vector<int>> properties) {
        static int decreasing[100000];
        static int size = 0;
        sort(properties.begin(), properties.end(), [](const auto& l, const auto& r) {
            if (l[0] == r[0])return l[1] < r[1];
            return l[0] > r[0];
            });

        size = 0;

        const int SIZE = properties.size();
        int answer = 0;
        for (int i = 0; i < SIZE; ++i) {
            if (
                size != 0 &&
                properties[decreasing[size - 1]][0] > properties[i][0] &&
                properties[decreasing[size - 1]][1] > properties[i][1]
                )
            {
                ++answer;
                continue;
            }
            decreasing[size++] = i;
        }
        return answer;
    }

    //436. Find Right Interval
    vector<int> findRightInterval(const vector<vector<int>>& intervals) {
        static vector<int> sorted(20000);
        const int SIZE = intervals.size();

        for (int i = 0; i < SIZE; ++i) {
            sorted[i] = i;
        }

        sort(sorted.begin(), sorted.begin() + SIZE, [&intervals](int l, int r) {
            return intervals[l][0] < intervals[r][0];
        });

        vector<int> result = vector<int>(SIZE, -1);
        for (int i = 0; i < SIZE; ++i) {
            int target = intervals[sorted[i]][1];
            int l = i, r = SIZE - 1;
            while (l < r) {
                int middle = l + (r - l) / 2;
                if (intervals[sorted[middle]][0] >= target)r = middle;
                else l = middle + 1;
            }
            if (intervals[sorted[l]][0] < target)continue;
            result[sorted[i]] = sorted[l];
        }

        return result;
    }

    //229. Majority Elemenet II
    vector<int> majorityElement(vector<int> nums) {
        if (nums.size() < 3)return nums;
        pair<int, int> f = { nums[0],1 }, s = { nums[1],1 },t;
        const int SIZE = nums.size();

        for (int i = 2; i < SIZE; ++i) {
            if (nums[i] == f.first)++f.second;
            else if (nums[i] == s.first)++s.second;
            else if (f.second == 1)f.first = nums[i];
            else if (s.second == 1)s.first = nums[i];
        }

        unordered_map<int, int> data;
        for (int i = 0; i < SIZE; ++i) {
            ++data[nums[i]];
        }
        show_unordered_map(data);

        cout << to_string(f) << "," << to_string(s) << "\n";

        vector<int> result;
        return result;
    }

    //343. Integer Break
    int integerBreak(int n) {
        if (n < 4)return n - 1;

        static int dp[59];
        
        for (int i = 0; i <= n; ++i) {
            dp[i] = 1;
        }

        while (n > 1) {
            for (int i = 2; i <= n; ++i) {
                dp[n - i] = max(dp[n - i], dp[n] * i);
            }
            --n;
        }

        return dp[0];
    }

    //1420. Build Array Where You Can Find The Maximum Exactly K Comparisons
    int numOfArrays(int n, int m, int k) {
        vector<vector<vector<int>>> dp(n, vector<vector<int>>(k));

        for (int i = 1; i <= m; ++i) {
            dp[0][0][i] = true;
        }
        
        int sum = 0;
        for (int i = 0; i < n-1; ++i) {
            for (int s = 0; s < k; ++s) {
                for (bool num : dp[i][s]) {

                }
            }
        }

        return sum;
    }

    //2009. Minimum Number of Operations to Make Array Continuous
    int minOperations(vector<int>& nums) {
        //find min and max 
    }

    //873. Length of Longest Fibonacci Subsequence
    int lenLongestFibSubseq(vector<int> arr) {
        
        //key -> num,value -> pos in sequence
        unordered_map<int, int> nums;
        const int SIZE = arr.size();



        int answer = 0;
        for (int i = 2; i < SIZE; ++i) {
            for (const auto& pair : nums) {
                if (nums.find(arr[i] - pair.first) == nums.end())continue;
                nums[arr[i]] = max(nums[arr[i]], pair.second + 1);
            }
            answer = max(answer, nums[arr[i]]);
        }
        
        return answer;
    }

    //954. Array of Doubled Pairs
    bool canReorderDoubled(vector<int> arr) {

        map<int, int> pos, neg;
        for (int num : arr) {
            if (num < 0)++neg[num];
            else ++pos[num];
        }

        while (pos.size() != 0) {
            auto doubled = pos.find(pos.begin()->first * 2);
            if (doubled == pos.end())return false;
            if (--doubled->second == 0)pos.erase(doubled);
            if (--pos.begin()->second == 0)pos.erase(pos.begin());
        }
        while (neg.size() != 0) {
            auto doubled = neg.find(neg.rbegin()->first * 2);
            if (doubled == neg.end())return false;
            if (--doubled->second == 0)neg.erase(doubled);
            if (--neg.rbegin()->second == 0)neg.erase(neg.rbegin()->first);
        }

        return true;
    }

    //935. Knight Dialer
    int knightDialer(int n) {
        static size_t* first = new size_t[10], * second = new size_t[10];

        if (n == 1)return 10;

        for (int i = 0; i < 10; ++i)first[i] = 1;
        first[5] = second[5] = 0;
        while (n-- > 1) {
            second[0] = (first[4] + first[6]) % 1000000007;
            second[1] = (first[8] + first[6]) % 1000000007;
            second[2] = (first[7] + first[9]) % 1000000007;
            second[3] = (first[4] + first[8]) % 1000000007;
            second[4] = (first[3] + first[9] + first[0]) % 1000000007;
            second[6] = (first[1] + first[7] + first[0]) % 1000000007;
            second[7] = (first[2] + first[6]) % 1000000007;
            second[8] = (first[1] + first[3]) % 1000000007;
            second[9] = (first[2] + first[4]) % 1000000007;
            swap(first, second);
        }

        int result = 0;
        for (int i = 0; i < 10; ++i) {
            result = (result + first[i]) % 1000000007;
        }

        return result;
    }

    //2251. Number of Flowers in Full Bloom
    vector<int> fullBloomFlowers(vector<vector<int>>& flowers, vector<int>& people) {
        static vector<int> ended(50000), temp(2);
        const int SIZE = flowers.size();

        for (int i = 0; i < SIZE; ++i) {
            ended[i] = flowers[i][1];
        }

        sort(flowers.begin(), flowers.end(), [](const auto& l, const auto& r) {return l[0] < r[0]; });
        sort(ended.begin(), ended.begin() + SIZE);

        unordered_map<int, int> memo;
        for (int i = 0; i < people.size(); ++i) {

            if (memo.find(people[i]) == memo.end()) {
                temp[0] = people[i] + 1;
                int itr1 = lower_bound(
                    flowers.begin(),
                    flowers.end(),
                    temp,
                    [](const auto& l, const auto& r) {return l[0] < r[0]; }
                ) - flowers.begin();
                int itr2 = upper_bound(ended.begin(), ended.begin() + SIZE, people[i] - 1) - ended.begin();
                memo[people[i]] = itr1 - itr2;
            }
            people[i] = memo[people[i]];
        }

        return people;
    }

    //1616. Split Two Strings to Make Palindrome
    bool isPalindrome(const string& str, int l, int r) {
        while (++l < --r) {
            if (str[l] != str[r])return false;
        }
        return true;
    }
    bool isPalindrome2(const string& l, const string& r) {
        int lp = -1, rp = r.length();
        while (++lp < --rp) {
            if (l[lp] != r[rp])break;
        }
        return isPalindrome(r, lp - 1, rp + 1);
    }
    bool checkPalindromeFormation(string a, string b) {
        return
            isPalindrome2(a, b)
            ||
            isPalindrome2(b, a)
            ;
    }

    //1095. Find in Mountain Array
    class MountainArray {
        
    public:
        int get(int index);
        int length();
    };

    unordered_map<int, int> mountain_memo;

    int getIndex(MountainArray& arr, int index) {
        if (mountain_memo.find(index) == mountain_memo.end()) {
            mountain_memo[index] = arr.get(index);
        }
        return mountain_memo[index];
    }

    int findPick(MountainArray& arr) {
        int left = 0;
        int right = arr.length() - 1;

        while (left < right) {
            int mid = left + (right - left) / 2;
            if (getIndex(arr, mid) < getIndex(arr, mid + 1)) {
                left = mid + 1;
            }
            else {
                right = mid;
            }
        }

        return left;
    }

    int binarySearch(MountainArray& arr, bool(*pred)(int, int), int l, int r, int target) {
        while (l < r) {
            int middle = (l + r) / 2;
            int val = getIndex(arr, middle);
            if (val == target)return middle;
            if (pred(target, val))r = middle;
            else l = middle + 1;
        }
        return getIndex(arr, l) == target ? l : -1;
    }

    int findInMountainArray(int target, MountainArray& mountainArr) {
        int pick = findPick(mountainArr);
        int l = binarySearch(mountainArr, [](int l, int r) {return l < r; }, 0, pick, target);
        if (l != -1)return l;
        return binarySearch(mountainArr, [](int l, int r) {return l > r; }, pick + 1, mountainArr.length() - 1, target);
    }

    //1269. Number of Ways to Stay in the Same Place After Some Steps
    int numWays(int steps, int arrLen) {
        static int* f = new int[501], * s = new int[501];

        arrLen = min(arrLen, steps);
        f[0] = 1;

        for (int i = 1; i <= steps; ++i) {
            s[0] = s[1] = f[0];
            int limit = min(i, arrLen);
            for (int k = 1; k < limit; ++k) {
                s[k + 1] = f[k];
                s[k - 1] = (s[k - 1] + f[k]) % 1000000007;
                s[k] = (s[k] + f[k]) % 1000000007;
            }
            swap(f, s);
        }

        return f[0];
    }
    
    //433. Minimum Genetic Mutation
    bool canMutate(const string& l, const string& r) {
        int sum = 0;
        for (int i = 0; i < 8; ++i) {
            sum += l[i] == r[i];
        }
        return sum == 7;
    }
    int minMutation(const string& startGene, const string& endGene, const vector<string>& bank) {
        deque<string> queue{ {startGene} };
        static bool used[10];

        memset(&used[0], 0, 10);

        int steps = 0;
        while (not queue.empty()) {

            int size = queue.size();
            while (size-- != 0) {
                string top = queue.front(); queue.pop_front();
                if (top == endGene)return steps;
                for (int i = 0; i < bank.size(); ++i) {
                    if (used[i])continue;
                    if (canMutate(top, bank[i])) {
                        queue.push_back(bank[i]);
                        used[i] = true;
                    }
                }
            }

            steps += 1;
        }

        return -1;
    }

    //40. Combination Sum II
    vector<vector<int>> combinationSum2(vector<int> candidates, int target) {
        //memorization based on count of specific numbers
        //how to implement the fastest?
        //pair<string,string> : key -> available numbers,value -> used numbers
        //two elements for each num : first char -> number it self, second char -> count of this number

        //optimize?

        unordered_map<int, int> numbers;
        for (int cand : candidates) {
            ++numbers[cand];
        }
        const int len = numbers.size() * 2;

        pair<string, string> init;
        for (const auto& pair : numbers) {
            init.first += string(1, char(pair.first)) + char(pair.second);
            init.second += string(1, char(pair.first)) + char(0);
        }

        vector<unordered_map<string, string>> memo(target + 1);
        memo[0].insert(init);

        for (int t = 0; t < target; ++t) {
            for (const auto& data : memo[t]) {
                for (int i = 0; i < len; i += 2) {
                    if (data.first[i + 1] == 0)continue;
                    if (t + data.first[i] > target)continue;
                    pair<string, string> copy = data;
                    --copy.first[i + 1];
                    ++copy.second[i + 1];
                    memo[t + data.first[i]].insert(copy);
                }
            }
        }

        vector<vector<int>> result;
        result.reserve(memo[target].size());
        for (auto& data : memo[target]) {
            vector<int> next;
            for (int i = 0; i < len; i += 2) {
                while (data.second[i + 1]-- > 0) {
                    next.push_back(int(data.second[i]));
                }
            }
            result.push_back(next);
        }

        return result;
    }

    //1361. Validate Binary Tree Nodes
    bool validateBinaryTreeNodes(int n, vector<int>& leftChild, vector<int>& rightChild) {
        static bool visited[10001];
        //add static pair<int,int> nodes[10000] ?
        memset(&visited[1], 0, n);

        int sum = 0;
        for (int i = 0; i < n; ++i) {
            if (leftChild[i] != -1)sum += !visited[leftChild[i] + 1];
            if (rightChild[i] != -1)sum += !visited[rightChild[i] + 1];
            visited[leftChild[i] + 1] = visited[rightChild[i] + 1] = 1;
        }

        if (sum + 1 != n)return false;

        int head;
        for (int i = 1; i <= n; ++i) {
            if (visited[i] == false) {
                head = i - 1;
                break;
            }
        }

        memset(&visited[0], 0, n);
        deque<int> levels{ {head} };

        int nodes_count = 0;
        while (not levels.empty()) {
            int size = levels.size();
            nodes_count += size;
            while (size-- > 0) {
                int top = levels.front(); levels.pop_front();
                if (leftChild[top] != -1) {
                    if (visited[leftChild[top]])return false;
                    levels.push_back(leftChild[top]);
                    visited[leftChild[top]] = true;
                }
                if (rightChild[top] != -1) {
                    if (visited[rightChild[top]])return false;
                    levels.push_back(rightChild[top]);
                    visited[rightChild[top]] = true;
                }
            }
        }

        return nodes_count == n;
    }

    //13. Roman to Integer
    int romanToInt(string s) {
        static unordered_map<string, int> fromRoman = {
            {"I",1},
            {"V",5},
            {"X",10},
            {"L",50},
            {"C",100},
            {"D",500},
            {"M",1000},
            {"IV",4},
            {"IX",9},
            {"XL",40},
            {"XC",90},
            {"CD",400},
            {"CM",900}
        };

        int sum = 0;
        for (int i = 0; i < s.length(); ++i) {
            string two = i + 2 <= s.length() ? s.substr(i, 2) : "";
            string single = s.substr(i, 1);
            if (fromRoman.find(two) != fromRoman.end()) {
                sum += fromRoman[two];
                i += 1;
            }
            else {
                sum += fromRoman[single];
            }
        }
        return sum;
    }

    //2050. Parallel Courses III
    int minimumTime(int n, vector<vector<int>>& relations, vector<int>& time) {
        unordered_map<int, vector<int>> dependencies;

        for (const auto& relation : relations) {
            dependencies[relation[1]].push_back(relation[0]);
        }

        //deque<int>

        return 0;
    }

    //1654. Minimum Jumps to Reach Home
    int minimumJumps(vector<int>& forbidden, int b, int a, int x) {
        static char used[6001]{ 1 };
        memset(&used[1], 0, 6001);

        for (int i = 0; i < forbidden.size(); ++i) {
            used[forbidden[i]] = true;
        }

        int steps = 0;
        deque<pair<int, bool>> jumps{ {0,1} };
        while (jumps.size() != 0) {
            int size = jumps.size();
            while (size-- > 0) {
                auto [pos, canJumpBackwards] = jumps.front(); jumps.pop_front();
                if (pos == x)return steps;
                if (pos + b <= 6000 && used[pos + b] != 1) {
                    used[pos + b] = 1;
                    jumps.push_back({ pos + b,1 });
                }
                if (pos - a > 0 && canJumpBackwards && used[pos - a] == 0) {
                    used[pos - a] = 2;
                    jumps.push_back({ pos - a,0 });
                }
            }
            ++steps;
        }

        return -1;
    }

    //310. Minimum Height Trees
    static int levels[20000];
    static bool visited[20000];

    int dfs(int current, int level, const vector<vector<int>>& relations) {
        visited[current] = true;

        levels[current] = level;
        int maxNext = 0;
        unordered_map<int,bool> maxes;
        for (int edge : relations[current]) {
            if (visited[edge])continue;
            int maxNested = dfs(edge, level + 1, relations);
            if (maxNext < maxNested) {
                maxes.clear();
                maxNext = maxNested;
            }
            if (maxNested == maxNext) {
                maxes[edge] = true;
            }
        }
        
        for (int edge : relations[current]) {
            if (maxes.find(edge) != maxes.end() || visited[edge])continue;
            levels[edge] = max(levels[edge], maxNext+1);
        }

        visited[current] = false;
        return maxNext+1;
    }

    vector<int> findMinHeightTrees(int n, vector<vector<int>> edges) {
        vector<vector<int>> data(n);
        for (const auto& edge : edges) {
            data[edge[0]].push_back(edge[1]);
            data[edge[1]].push_back(edge[0]);
        }
        
        dfs(0, 0, data);

        int minHeight = n;
        vector<int> result;
        for (int i = 0; i < n; ++i) {
            if (levels[i] < minHeight) {
                minHeight = levels[i];
                result.clear();
            }
            if (levels[i] == minHeight) {
                result.push_back(i);
            }
        }

        return result;
    }

    //1425. Constrained Subsequence Sum
    int constrainedSubsetSum(vector<int> nums, int k) {
        priority_queue<pair<int, int>, vector<pair<int, int>>, less<pair<int, int>>> data;

        int single = nums[0];
        data.push({ single,0 });
        const int SIZE = nums.size();
        int i = 1;
        while (i < SIZE) {
            while (i - data.top().second > k) {
                data.pop();
            }
            int local_max = nums[i] + max(0, data.top().first);
            if (single < local_max)single = local_max;
            data.push({ local_max,i });
            i += 1;
        }

        return single;
    }

    //37. Sudoku Solver
    void solveSudoku(vector<vector<char>> board) {
        unordered_map<int, bool> data[9][9];

        vector<pair<int, int>> sorted;
        for (int r = 0; r < 9; ++r) {
            for (int c = 0; c < 9; ++c) {
                int num = board[r][c] - '0';
                if (board[r][c] != '.') {
                    for (int i = 0; i < 9; ++i) {
                        data[i][c][num] = data[r][i][num] = false;
                    }
                    int sr = (r / 3) * 3;
                    int sc = (c / 3) * 3;
                    for (int i = 0; i < 3; ++i) {
                        for (int j = 0; j < 3; ++j) {
                            data[sr + i][sc + j][num] = false;
                        }
                    }
                }
                else sorted.push_back({ r,c });
            }
        }
        
        while (!sorted.empty()) {
            sort(sorted.begin(), sorted.end(), 
                [data](const auto& l, const auto& r) {
                    return data[l.first][l.second].size() < data[r.first][r.second].size();
                });
            auto [r, c] = sorted.back();
            auto& cell = data[r][c]; sorted.pop_back();
            if (cell.size() != 8)break;
            int num = 0;
            for (int i = 1; i < 10; ++i)
                if (cell.find(i) == cell.end()) {
                    num = i;
                    break;
                }
            
            for (int i = 0; i < 9; ++i) {
                data[i][c][num] = data[r][i][num] = false;
            }
            int sr = (r / 3) * 3;
            int sc = (c / 3) * 3;
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    data[sr + i][sc + j][num] = false;
                }
            }
            board[r][c] = num + '0';
        }

        show_vector(board,0x011);

    }

    //321. Create Maximum Number
    vector<int> maxNumber(vector<int>& nums1, vector<int>& nums2, int k) {
        vector<int> result(k);
        //case [1,1,1,1,8] with [1], k = 3
        const int S1 = nums1.size() - 1, S2 = nums2.size() - 1;
        int i1 = 0, i2 = 0;
        while (i1 < S1 && i2 < S2) {
            
        }

        return result;
    }

    //1793. Maximum Score of a Good Subarray
    int maximumScore(vector<int>& nums, int k) {
        const int LAST_INDEX = nums.size() - 1;
        int l = k, r = k;
        int local_min = nums[k];
        int answer = nums[k];
        
        while (l > 0 && r < LAST_INDEX) {
            if (nums[l-1] > nums[r+1])local_min = min(local_min,nums[--l]);
            else local_min = min(local_min, nums[++r]);
            answer = max(answer, local_min * (r - l + 1));
        }

        while (l > 0) {
            local_min = min(local_min, nums[--l]);
            answer = max(answer, local_min * (r - l + 1));
        }

        while (r < LAST_INDEX) {
            local_min = min(local_min, nums[++r]);
            answer = max(answer, local_min * (r - l + 1));
        }

        return answer;
    }

    //static const auto speedup = []() {
    //    std::ios::sync_with_stdio(false); std::cin.tie(nullptr); return 0;
    //}();
};

int Solution::levels[20000] = {};
bool Solution::visited[20000] = {};

//2671. Frequency Tracker
class FrequencyTracker {
    unordered_map<int, int> nums, frequencies;
public:
    FrequencyTracker() {

    }

    void add(int number) {
        if (nums[number] != 0) {
            --frequencies[nums[number]];
        }
        ++nums[number];
        ++frequencies[nums[number]];
        
    }

    void deleteOne(int number) {
        if (nums[number] == 0)return;
        --frequencies[nums[number]];
        --nums[number];
        ++frequencies[nums[number]];
    }

    bool hasFrequency(int frequency) {
        return frequencies[frequency] > 0;
    }
};

//706. Design HashMap
class MyHashMap {
    vector<int> data;
public:
    MyHashMap() {

    }

    void put(int key, int value) {
        if (key + 1 > data.size()) {
            data.resize(key + 1);
        }
        data[key] = value + 1;
    }

    int get(int key) {
        if (key + 1 > data.size())return -1;
        return data[key] - 1;
    }

    void remove(int key) {
        if (key + 1 > data.size())return;
        data[key] = 0;
    }
};


//173. Binary Search Tree Iterator
class BSTIterator {
    vector<pair<bool, tree_node::TreeNode*>> stack;
public:
    BSTIterator(tree_node::TreeNode* root) {
        while (root != nullptr) {
            stack.push_back({ 0,root });
            root = root->left;
        }
    }

    int next() {
        if (stack.size() == 0)return 0;
        int to_return = stack.back().second->val;
        getNext();
        return to_return;
    }

    bool hasNext() {
        return not stack.empty();
    }
private:
    void getNext() {
        if (stack.empty())return;

        stack.back().first = true;
        tree_node::TreeNode* itr = stack.back().second->right;
        while (itr != nullptr) {
            stack.push_back({ false,itr });
            itr = itr->left;
        }

        while (stack.size() != 0 && stack.back().first) {
            stack.pop_back();
        }
    }
};


//295. Find Median from Data Stream
class MedianFinder {
    int leftS = 0,rightS;
    map<int, int, less<int>> right;
    map<int, int, greater<int>> left;
public:
    MedianFinder() {

    }
    void addNum(int num) {
        
        if (right.empty() || right.begin()->second <= num) {
            ++right[num];
            ++rightS;
        }
        else {
            ++left[num];
            ++leftS;
        }

        if (leftS == rightS + 2) {

            ++right[left.begin()->first];
            if (--left.begin()->second == 0) {
                left.erase(left.begin());
            }
            --leftS;
            ++rightS;
            return;
        }
        if (leftS + 2 == rightS) {
            ++left[right.begin()->first];
            if (--right.begin()->second == 0) {
                right.erase(right.begin());
            }
            ++leftS;
            --rightS;
        }
    }

    double findMedian() {
        if (leftS > rightS)return left.begin()->first;
        if (rightS > leftS)return right.begin()->first;
        return (left.begin()->first + right.begin()->first) / 2.0;
    }
};