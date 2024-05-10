#pragma once
#include "../../../GeneralLibs/Headers/includes.hpp";
#include "DataStructures.hpp"


class Solution {
public:
    //396. Rotate Function
    int maxRotateFunction(vector<int> nums) {
        static int prefix[100001] = {};
        const int SIZE = nums.size();
        
        int sum = 0;
        for (int i = 0; i < SIZE; ++i) {
            prefix[i + 1] = prefix[i] + nums[i];
            sum += nums[i] * i;
        }

        int answer = sum;
        for (int i = 1; i < nums.size(); ++i) {
            sum += (prefix[i] - prefix[SIZE]) - (prefix[i - 1] - prefix[0]) + (nums[i - 1]) * (nums.size() - 1);
            if (sum > answer)answer = sum;
        }
        
        return answer;
    }

    //699. Falling Squares
    struct Square {
        int l;
        int r;
        int height;
        Square(int l, int r, int height) :l(l), r(r), height(height) {};
    };

    vector<int> fallingSquares(vector<vector<int>> positions) {
        const static auto comparer = [](const Square& l, const Square& r) { return l.r < r.r; };

        set<Square, decltype(comparer)> ranges(comparer);
        vector<int> result(0, positions.size());
        auto itr = positions.begin();

        result[0] = positions[0][1];
        ranges.insert({ (*itr)[0],(*itr)[0] + (*itr)[1],(*itr)[1] });

        auto tallest = ranges.begin();
        for (int j = 1; ++itr != positions.end(); ++j) {

            //Square square( 0, (*itr)[0], 0);
            //auto range = lower_bound(ranges.begin(), ranges.end(), square);
            //if (range == ranges.end()) {
            //    
            //}



        }

        return result;
    }

    //2477. Minimum Fuel Cost to Report to the Capital
    //first - count of passengers, second -> already used fuel
    pair<size_t, long long> dfs(const vector<vector<int>>& connections, int edge, const int seats) {
        static vector<bool> visited(100000, false);

        pair<size_t, long long> answer = { 1,0 };
        visited[edge] = true;
        for (int connection : connections[edge]) {
            if (!visited[connection] && connection != 0) {
                auto [f, s] = dfs(connections, connection, seats);
                answer.first += f;
                answer.second += s;
            }
        }
        visited[edge] = false;

        if (answer.second == 0)return { 1,1 };
        answer.second += (answer.first + seats - 1) / seats;

        return answer;
    }

    long long minimumFuelCost(vector<vector<int>> roads, int seats) {
        if (roads.size() < 2)return roads.size();

        vector<vector<int>> connections(roads.size() + 1);

        for (auto& road : roads) {
            connections[road[0]].push_back(road[1]);
            connections[road[1]].push_back(road[0]);
        }

        long long answer = 0;
        for (int road : connections[0]) {
            answer += dfs(connections, road, seats).second;
        }

        return answer;
    }

    //234. Palindrome Linked List
    bool isPalindrome(ListNode* head) {
        if (head->next == nullptr)return true;
        ListNode* fast = head->next, * slow = head;
        ListNode* prev = nullptr, * next = head->next;

        while (fast != nullptr) {
            fast = fast->next;
            slow->next = prev;
            prev = slow;
            slow = next;
            next = next->next;
            if (fast->next == nullptr) {
                next = next->next;
                break;
            }
            fast = fast->next;
        }

        while (slow != nullptr) {
            if (slow->val != next->val)return false;
            slow = slow->next;
            next = next->next;
        }

        return false;
    }


    //2024. Maximize the Confusion of an Exam
    int helper(const string& answers, int k, char trigger) {
        int l = 0, r = 0;
        int used = 0;
        int answer = 0;

        const size_t LEN = answers.length();
        while (r < LEN) {
            if (k < used) {
                if (answers[l++] != trigger)--used;
                continue;
            }
            answer = max(r - l, answer);
            if (answers[r++] != trigger)++used;
        }
        return answer;
    }

    int maxConsecutiveAnswers(string answerKey, int k) {
        return max(helper(answerKey, k, 'T'), helper(answerKey, k, 'F'));
    }

    //349. Intersection of Two Arrays
    vector<int> intersection(vector<int>& nums1, vector<int>& nums2) {
        vector<int> result;
        vector<int>* min_len = nums1.size() >= nums2.size() ? &nums2 : &nums1;
        vector<int>* max_len = nums1.size() >= nums2.size() ? &nums1 : &nums2;

        set<int> data(min_len->begin(), min_len->end());
        sort(max_len->begin(), max_len->end());

        for (int i : data) {
            if (binary_search(max_len->begin(), max_len->end(), i)) {
                result.push_back(i);
            }
        }

        return result;
    }

    int thirdMax(vector<int>& nums) {
        int first, second, third;
        return 0;
    }

    //148. Sort List

    ListNode* sortList(ListNode* head) {
        if (head == nullptr) return nullptr;
        if (head->next == nullptr) return head;

        ListNode* slow = head, * fast = head->next;

        while (fast != nullptr) {
            fast = fast->next;
            if (fast == nullptr)break;
            slow = slow->next;
            fast = fast->next;
        }

        fast = slow->next;
        slow->next = nullptr;

        ListNode* l = sortList(head);
        ListNode* r = sortList(fast);

        if (l->val > r->val) {
            head = r;
            r = r->next;
        }
        else {
            head = l;
            l = l->next;
        }

        ListNode* itr = head;
        while (l != nullptr && r != nullptr) {
            if (l->val > r->val) {
                itr = itr->next = r;
                r = r->next;
            }
            else {
                itr = itr->next = l;
                l = l->next;
            }
        }

        if (r == nullptr && l != nullptr)itr->next = l;
        else if (l == nullptr && r != nullptr)itr->next = r;

        return head;
    }

    //147. Insertion Sort List
    ListNode* insertionSortList(ListNode* head) {
        if (head == nullptr || head->next == nullptr) return head;

        ListNode* prev = head, * picked, * current;

        for (current = head->next; current != nullptr; current = current->next) {

            if (current->val < prev->val) {

                picked = current;
                prev->next = current->next;

                if (picked->val < head->val) {
                    picked->next = head;
                    head = picked;
                }
                else {
                    for (current = head; current->next != nullptr && !(picked->val < current->next->val); current = current->next);

                    picked->next = current->next;
                    current->next = picked;
                }

                current = prev;
            }
            prev = current;

        }

        return head;
    }

    //2658. Maximum Number of Fish in a Grid
    int dfs_2658(vector<vector<int>>& grid, uint16_t y, uint16_t x) {
        int result = grid[y][x];

        grid[y][x] = 0;

        if (y + 1 < grid.size() && grid[y + 1][x] != 0)result += dfs_2658(grid, y + 1, x);
        if (x + 1 < grid[0].size() && grid[y][x + 1] != 0)result += dfs_2658(grid, y, x + 1);
        if (y - 1 < grid.size() && grid[y - 1][x] != 0)result += dfs_2658(grid, y - 1, x);
        if (x - 1 < grid[0].size() && grid[y][x - 1] != 0)result += dfs_2658(grid, y, x - 1);

        return result;
    }

    int findMaxFish(vector<vector<int>> grid) {
        int answer = 0;
        for (uint16_t r = 0; r < grid.size(); ++r) {
            for (uint16_t c = 0; c < grid[0].size(); ++c) {
                if (grid[r][c] != 0)answer = max(answer, dfs_2658(grid, r, c));
            }
        }
        return answer;
    }

    //665. Non-decreasing Array
    bool checkPossibility(vector<int> nums) {
        int index = -1;
        int max = -1;
        for (int i = nums.size() - 1; 0 < i; --i) {
            if (nums[i - 1] > nums[i]) {
                if (index == -1) {
                    index = i - 1;
                    max = nums[i];
                    if (i + 1 < nums.size())max = std::max(max, nums[i + 1]);
                    continue;
                }
                if (i != index || max < nums[i - 1])return false;
            }
        }

        return true;
    }

    //92. Reverse Linked List II
    ListNode* reverseBetween(ListNode* head, int left, int right) {
        ListNode* prevLeft = head;
        int i = 1;
        while (++i < left)prevLeft = prevLeft->next;
        ListNode* nestedTail = prevLeft;
        while (i++ <= right)nestedTail = nestedTail->next;

        ListNode* nestedHead = left == 1 ? head : prevLeft->next;
        ListNode* tail = nestedTail->next;

        if (left != 1)prevLeft->next = nullptr;
        nestedTail->next = nullptr;

        {
            ListNode* prev = nullptr;
            ListNode* next = nestedHead->next;

            while (true) {
                nestedHead->next = prev;
                if (next == nullptr)break;
                prev = nestedHead;
                nestedHead = next;
                next = next->next;
            }
        }

        if (left != 1) prevLeft->next = nestedTail;
        else head = nestedTail;

        nestedHead->next = tail;

        return head;
    }

    //300. Longest Increasing Subsequence
    int lengthOfLIS(vector<int> nums) {
        vector<pair<int, int>> data(nums.size());

        for (int i = 0; i < nums.size(); ++i) {
            data[i] = { nums[i],i };
        }

        sort(data.begin(), data.end());

        show_vector(data);

        return 0;
    }

    //172. Factorial Trailing Zeroes
    int trailingZeroes(int n) {
        int answer = 0;
        int i = 5;
        while (i <= n) {
            answer += n / i;
            i *= 5;
        }
        return answer;
    }

    //1218. Longest Arithmetic Subsequence of Given Difference
    int longestSubsequence(vector<int> arr, int difference) {
        static int dp[100000];
        unordered_map<int, int> diffs;
        const size_t SIZE = arr.size();

        memset(&dp[0], 0, SIZE << 2);
        int answer = 0;

        show_array(&dp[0], SIZE);

        for (int i = 0; i < SIZE; ++i) {
            int num = arr[i];
            int prevNum = num - difference;

            if (diffs.find(prevNum) != diffs.end()) {
                dp[i] = dp[diffs[prevNum]] + 1;
            }

            diffs[num] = i;
            if (dp[i] > answer)answer = dp[i];
        }

        return answer + 1;
    }

    //518. Coin Change II
    int change(int amount, vector<int>& coins) {
        vector<int> dp(amount + 1);
        dp[0] = 1;

        for (int i = coins.size() - 1; 0 <= i; --i) {
            for (int j = coins[i]; j <= amount; ++j) {
                dp[j] += dp[j - coins[i]];
            }
        }

        return dp[amount];
    }

    //877. Stone Game
    bool stoneGame(vector<int>& piles) {
        return false;
    }

    //1547. Minimum Cost to Cut a Stick
    int minCost(int n, vector<int>& cuts) {
        return 0;
    }

    //47. Permutations II
    vector<vector<int>> permuteUnique(vector<int> nums) {
        sort(nums.begin(), nums.end());

        vector<vector<int>> result = { nums };

        while (next_permutation(nums.begin(), nums.end())) {
            result.push_back(nums);
        }

        return result;
    }

    //143. Reorder List
    void reorderList(ListNode* head) {
        if (head == nullptr)return;

        ListNode* slow = head, * fast = head->next;
        while (fast != nullptr) {
            fast = fast->next;
            if (fast == nullptr)break;
            slow = slow->next;
            fast = fast->next;
        }

        stack<ListNode*> tail;
        while (true) {
            slow = slow->next;
            if (slow == nullptr)break;
            tail.push(slow);
        }

        ListNode* oldItr = head, * newItr = head;
        while (not tail.empty()) {
            ListNode* next = oldItr->next;
        }
    }

    //2487. Remove Nodes From Linked List
    ListNode* removeNodes(ListNode* head) {
        static vector<ListNode*> stack;
        stack.clear();

        ListNode* newHead = nullptr;
        while (head != nullptr) {
            while (stack.size() != 0 && stack.back()->val < head->val) stack.pop_back();
            if (stack.size() != 0)stack.back()->next = head;
            else newHead = head;
            stack.push_back(head);
            head = head->next;
        }

        stack.back()->next = nullptr;

        return newHead;
    }

    //239. Sliding Window Maximum
    vector<int> maxSlidingWindow(vector<int> nums, int k) {
        map<int, size_t> data;
        size_t i = 0;
        while (i < k - 1) {
            ++data[nums[i++]];
        }
        
        vector<int> result(nums.size() - k + 1);
        size_t j = 0, left = 0;
        while (i < nums.size()) {
            ++data[nums[i++]];
            result[j++] = data.rbegin()->first;
            if (data[nums[left]] == 1)data.erase(nums[left]);
            else --data[nums[left++]];
        }

        return result;
    }

    //435. Non-overlapping Intervals
    int eraseOverlapIntervals(vector<vector<int>>& intervals) {
        sort(intervals.begin(), intervals.end(),
            [](const vector<int>& l, const vector<int>& r) {
                return l[0] < r[0];
            }
        );

        const size_t SIZE = intervals.size() - 1;
        int result = 0;
        
        for (int i = 0; i < SIZE; ++i) {
            if (intervals[i][1] <= intervals[i + 1][0])continue;
            ++result;
            intervals[i + 1][1] = intervals[i][1];
        }

        return result;
    }

    //299. Bulls and Cows
    string getHint(string secret, string guess) {
        static pair<short, short> digits[10] = {};

        int bulls = 0;
        for (size_t i = 0; i < secret.length(); ++i) {
            if (secret[i] == guess[i]) {
                ++bulls;
                continue;
            }
            ++digits[secret[i] - '0'].first;
            ++digits[guess[i] - '0'].second;

        }

        int cows = 0;
        for (size_t i = 0; i < 10; ++i) {
            cows += min(digits[i].first, digits[i].second);
            digits[i] = { 0,0 };
        }
        
        return to_string(bulls) + "A" + to_string(cows) + "B";
    }

    //451. Sort Characters By Frequency
    string frequencySort(string  s) {
        unordered_map<char, int> symbols;
        for (size_t i = 0; i < s.length(); ++i) ++symbols[s[i]];

        vector<pair<int, char>> sorted(symbols.size());
        int i = 0;
        for (auto& [k, v] : symbols) sorted[i++] = { v,k };
        sort(sorted.begin(), sorted.end(), [](const auto& l, const auto& r) {return l.first > r.first; });

        int offset = 0;
        for (auto& [k, v] : sorted) {
            int j = -1;
            while (++j < k)s[offset++] = v;
        }
        
        return s;
    }

    //525. Contiguous Array
    int findMaxLength(vector<int>& nums) {
        unordered_map<int, int> sums;
        int prefix = 0;

        int answer = 0;
        for (int i = 0; i < nums.size(); ++i) {
            prefix += nums[i];
            sums[prefix] = i;
            /*if(sums.find())*/
        }
        
        return answer;
    }

    //684. Redundant Connection
    vector<int> findRedundantConnection(vector<vector<int>>& edges) {
        Unions unions(edges.size() + 1);

        for (auto& edge : edges) {
            if (!unions.insert(edge[0], edge[1]))return edge;
        }

        return {};
    }

    //498. Diagonal Traverse
    vector<int> findDiagonalOrder(vector<vector<int>>& mat) {
        //max length of diagonal = min(C,R)
        //range of max length = [min(R-1,C-1),max(R-1,C-1)]
        //pos = sum of diagonals before + pos in diagonal(need a size of diagonal)
        const size_t R = mat.size();
        const size_t C = mat[0].size();
        
        vector<int> result(R*C, 0);
        for (size_t r = 0; r < R; ++r) {
            for (size_t c = 0; c < C; ++c) {
                int diagonal = r + c;
                result[0] = mat[r][c];
            }
        }

        return result;
    }

    //673. Number of Longest Increasing Subsequence
    int findNumberOfLIS(vector<int>& nums) {
        int n = nums.size();
        vector<int> dp(n, 1);
        vector<int> count(n, 1);

        for (int i = n - 1; i >= 0; --i) {
            int local_max = 0;
            int ways = 1;

            for (int j = i + 1; j < n; ++j) {
                if (nums[j] <= nums[i]) continue;
                if (dp[j] > local_max) {
                    local_max = dp[j];
                    ways = count[j];
                }
                else if (dp[j] == local_max) ways += count[j];
            }

            dp[i] = local_max + 1;
            count[i] = ways;
        }

        int max_len = *max_element(dp.begin(), dp.end());
        int answer = 0;

        for (int i = 0; i < n; ++i) {
            if (dp[i] == max_len) {
                answer += count[i];
            }
        }

        return answer;
    }

    //300. Longest Increasing Subsequence
    int lengthOfLIS(vector<int>& nums) {
        

        // Custom comparison function for the set
        //static const auto comparePairs = [](const pair<int, int*>& a, const pair<int, int*>& b) {
        //    return a.first < b.first;
        //};

        //// Using set with the custom comparison function
        //set<pair<int, int*>, decltype(comparePairs)> dp(comparePairs);

        //int answer = 0;
        //for (int i = nums.size() - 1; 0 <= i; --i) {
        //    int local_max = 0;
        //    auto itr = lower_bound(dp.begin(), dp.end(), make_pair(nums[i] + 1, 0));

        //    while (itr != dp.end()) {
        //        ++itr;
        //    }

        //    itr = dp.find({ nums[i],0 });
        //    if (itr != dp.end())dp.insert({ nums[i],new int(local_max + 1)});
        //    /*else *(itr->second);*/
        //}

        vector<int> dp(nums.size(), 1);

        int answer = 0;
        for (int i = nums.size() - 2; 0 <= i; --i) {
            int local_max = 0;
            for (int j = i + 1; j < nums.size(); ++j) {
                if (nums[j] <= nums[i])continue;
                if (dp[j] > local_max)local_max = dp[j];
            }
            dp[i] = local_max + 1;
            if (local_max > answer)answer = local_max;
        }

        return answer + 1;
    }

    //628. Maximum Product of Three Numbers
    int maximumProduct(vector<int> nums) {
        set<int> data;
        for (int i = 0; i < nums.size(); ++i) {
            data.insert(nums[i]);
            if (data.size() > 3)data.erase(data.begin());
        }
        show_set(data,0x11);
        return *data.begin() * (*++data.begin()) * (*data.rbegin());
    }

    //50. Pow(x, n)
    double powLight(double x, long long n) {
        if (n == 0)return 1;
        if (n & 1)return x * powLight(x, n - 1);
        double nested = powLight(x, n / 2);
        return nested * nested;
    }

    double myPow(double x, long long n) {
        if (n == 0)return 1;
        if (x == 1.0)return 1;
        if (x == 0)return 0;
        if (n < 0)return 1 / powLight(x, -n);
        return powLight(x, n);
    }

    //103. Binary Tree Zigzag Level Order Traversal
    vector<vector<int>> zigzagLevelOrder(tree_node::TreeNode* root) {
        static deque<tree_node::TreeNode*> queue;

        queue.push_back(root);

        int direction = 0;
        vector<vector<int>> result;
        while (queue.size() != 0) {

            int size = queue.size();
            result.push_back({});

            while (0 <= --size) {
                auto top = queue.front(); queue.pop_front();
                if (top == nullptr)continue;
                result.back().push_back(top->val);
                queue.push_back(top->left);
                queue.push_back(top->right);
            }
            if (++direction & 1) {
                reverse(result.back().begin(), result.back().end());
                direction = 0;
            }
        }

        return result;
    }
    //827. Making A Large Island
    unordered_map<uint32_t, bool> water;

    void explore_island(vector<vector<int>>& grid, unordered_map<uint32_t, int*>& data,int& size, uint16_t y, uint16_t x) {
        static const vector<pair<short, short>> directions = { {0, 1}, {1, 0}, {-1, 0}, {0, -1} };
        
        grid[y][x] = 2;
        data[(y << 16) + x] = &(++size);
        for (auto& pair : directions) {
            uint16_t ny = y + pair.first;
            uint16_t nx = x + pair.second;
            if (ny < grid.size() && nx < grid[0].size()) {
                if(grid[ny][nx] == 1)explore_island(grid, data, size, ny, nx);
                if (grid[ny][nx] == 0)water.insert({ (ny << 16) + nx , true });
            }
        }
    }

    int largestIsland(vector<vector<int>> grid) {
        static const vector<pair<short, short>> directions = { {0, 1}, {1, 0}, {-1, 0}, {0, -1} };
        unordered_map<uint32_t, int*> data;

        int answer = 1;
        for (uint16_t r = 0; r < grid.size(); ++r) {
            for (uint16_t c = 0; c < grid[0].size(); ++c) {
                if (grid[r][c] == 1) {
                    int* size = new int(0);
                    explore_island(grid, data, *size, r, c);
                    if (*size > answer)answer = *size;
                }
            }
        }
        
        static unordered_map<int*, bool> used;
        for (auto&[key,_] : water) {
            for (auto& pair : directions) {
                uint16_t ny = (key >> 16) + pair.first;
                uint16_t nx = (key & 0x0000FFFF) + pair.second;
                if (ny < grid.size() && nx < grid[0].size() && grid[ny][nx] == 2) {
                    used[data[(ny << 16) + nx]] = true;
                }
            }
            int nested = 1;
            for (auto& [f, _] : used)nested += *f;
            if (nested > answer)answer = nested;
            used.clear();
        }
        water.clear();

        return answer;
    }


    //852. Peak Index in a Mountain Array
    int peakIndexInMountainArray(vector<int> arr) {
        int l = 0, r = arr.size() - 1;
        while (l < r) {
            int middle = l + (r - l) / 2;
            if (arr[middle] > arr[middle + 1])r = middle;
            else l = middle;
        }
        return l;
    }

    //am i stupid or something?
    // unsolved...
    //57. Insert Interval
    vector<vector<int>> insert(vector<vector<int>>& intervals, vector<int>& newInterval) {
        if (intervals.size() == 0)return { newInterval };
        auto first = lower_bound(intervals.begin(), intervals.end(), newInterval, [](const auto& l, const auto& r) {return l[0] <= r[0]; });
        if (first == intervals.end()) {
            intervals.push_back(newInterval);
            return intervals;
        }
        auto last = lower_bound(first, intervals.end(), newInterval, [](const auto& l, const auto& r) {return l[1] <= r[1]; });
        auto from = first == intervals.begin() ? first : first - 1;
        if (first != intervals.end())cout << (*first)[0] << "->" << (*first)[1] << "\n";
        if (last != intervals.end())cout << (*last)[0] << "->" << (*last)[1] << "\n";
        if ((*last)[0] >= newInterval[1])newInterval[1] = (*last)[1];

        intervals.erase(first, last);
        (*from)[1] = newInterval[1];

        return intervals;
    }

    //60. Permutation Sequence
    string getPermutation(int n, int k) {
        static const int factorial[10] = { 1,1,1,2,6,24,120,720,5040,40320 };
        static vector<int> numbers(n);
        numbers = { 1,2,3,4,5,6,7,8,9 };

        k -= 1;
        string result = "";
        while (n > 0) {
            int at = k / factorial[n];
            result += char(numbers[at] + '0');
            k %= factorial[n--];
            numbers.erase(numbers.begin() + at);
        }

        return result;
    }

    //80. Remove Duplicates from Sorted Array II
    //unsolved
    int removeDuplicates(vector<int>& nums) {
        const size_t size = nums.size();
        int i = 1;
        int count = 0;
        for (int k = 1; k < size; ++k) {
            if (nums[k] != nums[k - 1])continue;
            i = k;
            for (++k; k < size && nums[k - 1] == nums[k];++k) {
                nums[++i] = nums[k];
            }
        }
    }

    //205. Isomorphic Strings
    bool isIsomorphic(string s, string t) {
        
        ios::sync_with_stdio(false);
        cin.tie(NULL);

        static short pairs[256] = {};
        static bool used[256] = {};

        memset(&pairs[0], 0, 256*2);
        memset(&used[0], 0, 256);

        for (size_t i = 0; i < s.length(); ++i) {
            if (pairs[s[i]] == 0) {
                if (used[t[i]])return false;
                used[t[i]] = pairs[s[i]] = t[i] + 1;
            }
            else if (pairs[s[i]] != t[i] + 1)return false;
        }
        return true;
    }

    //405. Convert a Number to Hexadecimal
    string toHex(int num) {
        static const char characters[16] = { '0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f' };

        if (num == 0)return "0";

        string result = "";
        uint32_t k = num;
        for (int i = 0; i < 8 && k != 0; ++i) {
            int c = k & 0xF;
            result.push_back(characters[c]);
            k >>= 4;
        }
        reverse(result.begin(), result.end());
        return result;
    }

    //1870. Minimum Speed to Arrive on Time
    int minSpeedOnTime(vector<int> dist, double hour) {
        ios::sync_with_stdio(false);
        cin.tie(NULL);

        if (hour <= dist.size() - 1)return -1;
        int l = 1, r = 10000000;

        while (l < r) {
            double middle = l + (r - l) / 2;
            double sum = 0;
            for (int i = dist.size() - 2; 0 <= i; --i) {
                sum += ceil(dist[i] / middle);
            }
            sum += dist.back() / middle;

            if (sum > hour)l = middle + 1;
            else r = middle;
        }

        return r;
    }

    //daily 
    long long maxRunTime(int n, vector<int>& batteries) {

        const size_t SIZE = batteries.size();
        long long result = 0;

        while (true) {
            sort(batteries.begin(), batteries.end());
            if (batteries[SIZE - n] == 0)break;
            ++result;
            for (int i = 1; i <= n; ++i) {
                --batteries[SIZE - i];
            }
        }

        return result;
    }

    //2365. Task Scheduler II
    long long taskSchedulerII(const vector<int> tasks, long long space) {
        static unordered_map<int, long long> schedule;
        long long days = 0;

        for (int i = 0; i < tasks.size(); ++i) {
            if (schedule.find(tasks[i]) != schedule.end()) {
                if (schedule[tasks[i]] + space > days)days = schedule[tasks[i]] + space;
            }
            schedule[tasks[i]] = ++days;
        }

        schedule.clear();

        return days;
    }

    //621. Task Scheduler
    int leastInterval(const vector<char> & tasks, int n) {
        const static size_t MAX_N = 100;
        static uint16_t data[MAX_N + 1] = {};
        static char size = 0;

        unordered_map<char, uint16_t> frequency;
        for (int i = 0; i < tasks.size();++i)++frequency[tasks[i]];

        priority_queue<uint16_t, vector<uint16_t>, less<uint16_t>> queue;
        for (auto& [_, f] : frequency)queue.push(f);

        int time = 0;
        int last = 0;
        n += 1;
        while (queue.size() != 0) {
            time += n;
            last = queue.size();
            for (int i = 0; i < n && queue.size() != 0; ++i) {
                int f = queue.top(); queue.pop();
                if (f - 1 != 0)data[size++] = f - 1;
            }
            while (size != 0) {
                queue.push(data[--size]);
            }
        }

        return time - n + last;
    }

    //377. Combination Sum IV
    int combinationSum4(vector<int> nums, int target) {
        static uint32_t dp[1001];
        memset(&dp[1], 0, target << 2);

        for (int num : nums)++dp[num];

        for (int i = 1; i <= target; ++i) {
            for (int num : nums) {
                if (i + num < 1001)dp[i + num] += dp[i];
            }
        }

        return dp[target];
    }

    //486. Predict the Winner
    bool helper(vector<int>::iterator l, vector<int>::iterator r,int difference) {
        return false;
    }


    bool PredictTheWinner(vector<int>& nums) {
        return 
            helper(nums.begin()+1,nums.end()-1, *nums.begin())
            ||
            helper(nums.begin(),nums.end()-2,*(nums.end()-1))
        ;
    }

    //315. Count of Smaller Numbers After Self
    vector<int> countSmaller(vector<int>& nums) {
        vector<int> sortedNums(nums);
        sort(sortedNums.begin(), sortedNums.end());

        FenwickTree ft(nums.size());
        vector<int> result(nums.size());

        for (int i = nums.size() - 1; 0 <= i; --i) {
            int idx = lower_bound(sortedNums.begin(), sortedNums.end(), nums[i]) - sortedNums.begin() + 1;
            result[i] = ft.query(idx - 1);
            ft.update(idx);
        }

        return result;
    }

    //493. Reverse Pairs
    //unsolved
    //indexed tree
    //fenwick tree
    int reversePairs(vector<int>& nums) {
        vector<int> sortedNums(nums.size(),0);
        for (int i = 0; i < nums.size(); ++i)sortedNums[i] = nums[i]*2;
        sort(sortedNums.begin(), sortedNums.end());

        FenwickTree ft(nums.size());

        int amount = 0;
        for (int i = nums.size() - 1; i >= 0; --i) {
            int idx = lower_bound(sortedNums.begin(), sortedNums.end(), nums[i]*2) - sortedNums.begin() + 1;
            amount += ft.query(idx - 1);
            ft.update(idx);
        }

        return amount;
    }

    //372. Super Pow
    int superPow(int a, vector<int>& b,int from = 0) {
        if (a < 2)return a;

        if (b.back() == 0 && b.size() == 1)return 1;
        if (b.back() == -1) {
            for (int i = b.size() - 2; 1 <= i; --i) {
                if (0 <= b[i])break;
                b[i] = 9;
                --b[i - 1];
            }
            if (b[0] == 0)b.erase(b.begin());
        }

        if (b.back() % 2 == 0) {
            for (int i = 0; i < b.size() - 1; ++i) {
                b[i + 1] += 10 * (b[i] & 1);
                b[i] /= 2;
            }
            b.back() /= 2;
            if (b.front() == 0)b.erase(b.begin());
            int nested = superPow(a, b);
            return (nested * nested) % 1337;
        }
        --b.back();
        return ((a % 1337) * superPow(a, b)) % 1337;
    }


};

struct TrieNode {
public:
    TrieNode* letters[26] = {};
    bool is_complete_word;

    TrieNode() {
        is_complete_word = false;
    }

    void insert(const string& word, size_t index) {
        if (index == word.length()) {
            is_complete_word = true;
            return;
        }

        if (letters[word[index] - 'a'] == nullptr) {
            letters[word[index] - 'a'] = new TrieNode();
        }

        letters[word[index] - 'a']->insert(word, index + 1);
    }

    bool find(const string& word, size_t index) {
        if (index == word.length())return is_complete_word;

        if (word[index] != '.') {
            if (letters[word[index] - 'a'] == nullptr)return false;
            return letters[word[index] - 'a']->find(word, index + 1);
        }

        for (char i = 0; i < 26; ++i) {
            if (letters[i] != nullptr && letters[i]->find(word, index + 1))return true;
        }

        return false;
    }
};

class Suggestions {
    TrieNode* root;
public:
    Suggestions() {
        root = new TrieNode();
    }

    void insert(const string& word) {
        root->insert(word, 0);
    }

    vector<string> find(const string& word) {
        const size_t MAX_WORDS = 3;
        vector<TrieNode*> queue;
    }

};

class WordDictionary {
    TrieNode* root;
public:
    WordDictionary() {
        root = new TrieNode();
    }

    void addWord(string word) {
        root->insert(word, 0);
    }

    bool search(string word) {
        return root->find(word, 0);
    }

    ~WordDictionary() {
        delete root;
    }
};

class Codec {
private:
    string to256(uint16_t num) {
        return string({ char((num >> 8) & 0xFF),char(num & 0xFF) });
    }

    uint16_t from256(const string& num, size_t from) {
        return (static_cast<unsigned char>(num[from]) << 8) + static_cast<unsigned char>(num[from + 1]);
    }

    tree_node::TreeNode* deserialize_helper(const string& data, size_t l, size_t r) {
        if (r <= l)return nullptr;

        int num = 0;
        {
            bool negative = false;
            if (data[l] == '-') negative = ++l;

            while (isdigit(data[l])) {
                num = num * 10 + data[l++] - '0';
            }

            if (negative)num *= -1;
        }
        tree_node::TreeNode* root = new tree_node::TreeNode(num);

        if (data[l] == 'f')return root;
        if (data[l] == 'l') {
            root->right = deserialize_helper(data, l + 1, r);
            return root;
        }
        if (data[l] == 'r') {
            root->left = deserialize_helper(data, l + 1, r);
            return root;
        }

        size_t offset = from256(data, l + 1) + l + 3;
        root->left = deserialize_helper(data, l + 3, offset);
        root->right = deserialize_helper(data, offset, r);

        return root;
    }
public:
    string serialize(tree_node::TreeNode* root) {
        if (root == nullptr)return "";
        string prefix = to_string(root->val);
        string left = serialize(root->left);
        string right = serialize(root->right);

        //node is leaf
        if (left.length() == 0 && right.length() == 0)return prefix + "f";
        //left empty
        if (left.length() == 0)return prefix + "l" + right;
        //right empty
        if (right.length() == 0)return prefix + "r" + left;
        return prefix + ":" + to256(left.length()) + left + right;
    }


    tree_node::TreeNode* deserialize(string data) {
        if (data.length() == 0)return nullptr;
        return deserialize_helper(data, 0, data.length() - 1);
    }
};

class BSTIterator {
    stack<tree_node::TreeNode*> path;
    char state;
public:
    BSTIterator(tree_node::TreeNode* root) {
        while (root != nullptr) {
            path.push(root);
            if (root->left != nullptr) root = root->left;
            else root = root->right;
        }
        state = 0;
    }

    int next() {
        return 0;
    }

    bool hasNext() {
        return false;
    }
};

//225. Implement Stack using Queues
//You must use only standard operations of a queue, which means that only :
//push to back, peek/pop from front, size and is empty operations are valid.
class MyStack {
    queue<int> data;
    bool isReversed;
public:
    MyStack() {
        isReversed = false;
    }

    void push(int x) {
        data.push(x);
        isReversed = false;
    }

    int pop() {
        int num = this->top();
        data.pop();
        isReversed = false;
        return num;
    }

    int top() {
        if (!isReversed)reverseData();
        isReversed = true;
        return data.front();
    }

    bool empty() {
        return data.empty();
    }
private:
    inline void reverseData() {
        const size_t SIZE = data.size();
        for (int k = 1; k < SIZE; ++k) {
            data.push(data.front());
            data.pop();
        }
    }
};

class MedianFinder {
    map<int, size_t> stream;
    size_t size = 1;
public:
    MedianFinder() {}

    void addNum(int num) {
        ++stream[num];
        ++size;
    }

    double findMedian() {
        //TODO : iterate to middle in accurate way
        size_t half = size >> 1;
        auto itr = stream.begin();
        size_t i = 0;
        
        while (i < half) {
            i += itr->second;
            ++itr;
        }

        return 0;
    }
};

struct NodeT {
    size_t index;
    int id;
    NodeT* next;
    
    NodeT(size_t index, int id) :
        index(index), id(id), next(nullptr) {};
};

struct TwitterUser {
    unordered_map<int, bool> followed;
    NodeT* root = nullptr;

    inline void pushTweet(NodeT* tweet) {
        tweet->next = root;
        root = tweet;
    }
};

class Twitter {
    unordered_map<int, TwitterUser> users;
    size_t tweetsAmount;
public:
    Twitter() {
        tweetsAmount = 0;
    }

    void postTweet(int userId, int tweetId) {
        users[userId].pushTweet(new NodeT(tweetsAmount++, tweetId));
    }

    vector<int> getNewsFeed(int userId) {
        static list<NodeT*> following;
        following.clear();

        TwitterUser& user = users[userId];
        if (user.root != nullptr)following.push_back(user.root);
        
        //fill the list
        for (auto& [f, s] : user.followed) {
            if(users[f].root != nullptr) following.push_back(users[f].root);
        }
            
        vector<int> result = {};
        int found = -1;
        while (++found < 10 && !following.empty()) {
            auto min_tweet = max_element(following.begin(), following.end(), 
                [](const auto& l, const auto& r) 
                {
                    return l->index < r->index;
                }
            );
            result.push_back((*min_tweet)->id);
            (*min_tweet) = (*min_tweet)->next;
            if (*min_tweet == nullptr)following.erase(min_tweet);
        }

        return result;
    }

    void follow(int followerId, int followeeId) {
        users[followerId].followed[followeeId] = true;
    }

    void unfollow(int followerId, int followeeId) {
        users[followerId].followed.erase(followeeId);
    }
};

class TrieNodeS {
public:
    map<char, TrieNodeS*> letters;
    bool is_complete_word;

    TrieNodeS() {
        letters = {};
        is_complete_word = false;
    }

    void insert(const string& word, size_t index) {
        if (index == word.length()) {
            is_complete_word = true;
            return;
        }

        if (letters[word[index]] == nullptr) {
            letters[word[index]] = new TrieNodeS();
        }

        letters[word[index]]->insert(word, index + 1);
    }

    bool find(const string& word, size_t index) {
        if (index == word.length())return is_complete_word;

        if (letters.find(word[index]) == letters.end())return false;
        return letters[word[index]]->find(word,index+1);
    }

    vector<string> min3(const string& prefix) {
        if (is_complete_word)return { prefix };
        
        vector<string> result;
        

        return result;
    }

};

//303. Range Sum Query - Immutable
class NumArray {
    vector<int>* sums;
public:
    NumArray(vector<int>& nums) {
        sums = &nums;
        for (int i = 1; i < nums.size(); ++i) {
            nums[i] += nums[i - 1];
        }
    }

    int sumRange(int left, int right) {
        if (left == 0)return (*sums)[right];
        return (*sums)[right] - (*sums)[left - 1];
    }
};

//304. Range Sum Query 2D - Immutable
class NumMatrix {
    vector<vector<int>>* m;
public:
    NumMatrix(vector<vector<int>>& matrix) {
        ios::sync_with_stdio(false);
        cin.tie(NULL);

        for (size_t r = 0; r < matrix.size() - 1; ++r) {
            matrix[r + 1][0] += matrix[r][0];
            for (size_t c = 1; c < matrix[0].size(); ++c) {
                matrix[r + 1][c] += matrix[r][c];
                matrix[r][c] += matrix[r][c - 1];
            }
        }

        for (size_t c = 1; c < matrix[0].size(); ++c) {
            matrix.back()[c] += matrix.back()[c - 1];
        }

        m = &matrix;
    }

    int sumRegion(int row1, int col1, int row2, int col2) {
        if (row1 == 0 && col1 == 0) return (*m)[row2][col2];
        if (row1 == 0) return (*m)[row2][col2] - (*m)[row2][col1 - 1];
        if (col1 == 0) return (*m)[row2][col2] - (*m)[row1 - 1][col2];
        return (*m)[row2][col2] - (*m)[row2][col1 - 1] - (*m)[row1 - 1][col2] + (*m)[row1 - 1][col1 - 1];
    }
};

//1286. Iterator for Combination
class CombinationIterator {
    vector<bool> bits;
    int amount = 0;
    string to_output;
    string data;
public:
    CombinationIterator(string characters, int combinationLength) {

        amount = combinations(characters.length(), combinationLength);
        data = characters;
        to_output = string(combinationLength, 0);

        bits = vector<bool>(characters.length(), true);
        while (--combinationLength != -1) {
            bits[combinationLength] = false;
        }
    }

    string next() {
        --amount;
        int j = 0;
        for (int i = 0; i < bits.size(); ++i) {
            if (!bits[i]) {
                to_output[j++] = data[i];
            }
        }
        next_permutation(bits.begin(), bits.end());
        return to_output;
    }

    bool hasNext() {
        return amount > 0;
    }
private:
    uint32_t combinations(uint16_t n, uint16_t k) {
        static vector<vector<uint32_t>> data = { {1},{1,1} };

        for (size_t current_size = data.size(); current_size <= n; ++current_size) {
            data.push_back(vector<uint32_t>(data.back().size() + 1, 1));

            for (int k = 1; k < current_size; ++k) {
                data[current_size][k] = (data[current_size - 1][k - 1] + data[current_size - 1][k]);
            }
        }

        return data[n][k];
    }
};