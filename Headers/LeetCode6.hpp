#pragma once
#include "../../../GeneralLibs/Headers/includes.hpp";
#include "DataStructures.hpp"

class Solution {
public:
    //417. Pacific Atlantic Water Flow
    vector<vector<int>> pacificAtlantic(vector<vector<int>>& heights) {
        static deque<pair<uint16_t, uint16_t>> queue;
        static unordered_map<uint32_t, short> common_points;

        //insert all pacific coast cells
        for (uint16_t r = 0; r < heights.size(); ++r)queue.push_back({ r,0 });
        for (uint16_t c = 1; c < heights[0].size(); ++c)queue.push_back({ 0,c });

        while (queue.size() != 0) {
            int size = queue.size();
            while (0 <= --size) {
                auto[y,x] = queue.front(); queue.pop_front();
                if (y + 1 < heights.size() && heights[y][x] <= heights[y + 1][x])queue.push_back({ y + 1,x });
                if (x + 1 < heights[0].size() && heights[y][x] <= heights[y][x + 1])queue.push_back({y,x + 1});
            }
        }
    }

    //441. Arranging Coins
    int arrangeCoins(int n) {
        return sqrt(double(n << 1) + 0.25) - 0.5;   
    }

    //204. Count Primes
    int countPrimes(int n) {
        static vector<int> primes = { 2,3,5,7,11 };
        if (n <= *primes.rbegin()) {
            return lower_bound(primes.begin(), primes.end(), n) - primes.begin();
        }

        for (int i = primes.back() + 1; i < n; ++i) {
            int to = sqrt(i), j = -1;
            while (primes[++j] <= to) {
                if (i % primes[j] == 0)break;
            }
            if (to < primes[j]) {
                primes.push_back(i++);
            }
        }
        return primes.size();
    }

    //227. Basic Calculator II
    int calculate2(string s) {
        static unordered_map<char, function<int(int, int)>> operators = {
            {'+',[](int l,int r) {return l + r; }},
            {'-',[](int l,int r) {return l - r; }},
            {'/',[](int l,int r) {return l / r; }},
            {'*',[](int l,int r) {return l * r; }}
        };
        static string stack[300000] = {};
        static int l, r;
        l = r = 0;

        const uint32_t LEN = s.length();
        for (uint32_t i = 0; i < LEN;++i) {
            if (s[i] == ' ')continue;
            if (!isdigit(s[i])) {
                stack[r++] = string(1, s[i]);
                continue;
            }

            string input = "";
            while (i < LEN && isdigit(s[i])) {
                input += s[i++];
            }

            if (r > 0 && (stack[r-1] == "*" || stack[r-1] == "/")) {
                char operation = stack[--r][0];
                int top = stoi(stack[--r]);
                input = to_string(operators[operation](top,stoi(input)));
            }
            stack[r++] = input;
            --i;
        }
        
        while (l+1 < r) {
            int bot = stoi(stack[l++]);
            char operation = stack[l++][0];
            int top = stoi(stack[l++]);
            stack[l - 1] = to_string(operators[operation](top,bot));
        }

        return stoi(stack[max(0, l - 1)]);
    }

    //289. Game of Life
    void gameOfLife(vector<vector<int>>& board) {
        static const pair<char, char> directions[8] = { {0,1},{1,0},{-1,0},{0,-1},{1,1},{-1,1},{1,-1},{-1,-1} };
        static deque<pair<uint32_t,int>> change = {}; //first -> (16 bits for y,16 bits for x), second -> value to insert

        for (uint16_t r = 0; r < board.size(); ++r) {
            for (uint16_t c = 0; c < board[0].size(); ++c) {
                int amount = 0;
                for (char i = 0; i < 8; ++i) {
                    if (
                        r + directions[i].first < board.size()
                        &&
                        c + directions[i].second < board[0].size()
                        )amount += board[r + directions[i].first][c + directions[i].second];
                }

                if (board[r][c] == 1 && (amount < 2 || amount > 3)) {
                    change.push_back({(r << 16) + c, 0});
                }
                else if (board[r][c] == 0 && amount == 3) {
                    change.push_back({ (r << 16) + c, 1 });
                }
            }
        }

        while (change.size() != 0) {
            auto [pos, v] = change.front(); change.pop_front();
            board[pos >> 16][pos & 0x0000FFFF] = v;
        }
    }

    //69. Sqrt(x)
    int mySqrt(int x) {
        if (x < 2)return x;
        static int l, r;
        l = 1; r = 46340;
        while (l < r) {
            int middle = l + (r - l) / 2;
            if (middle * middle >= x) r = middle;
            else l = middle + 1;
        }
        return l;
    }

    //139. Word Break
    bool wordBreak(const string& s, const vector<string>& wordDict) {
        static const int MAX_SIZE = 301;
        static bool dp[MAX_SIZE] = { true };
        memset(&dp[1], false, s.length());

        for (int i = 0; i < s.length(); ++i) {
            if (!dp[i])continue;
            for (const string& word : wordDict) {
                if (word.length() + i > s.length())continue;
                int l = -1, r = word.length();
                while (++l <= --r) {
                    if (word[l] != s[i + l] || word[r] != s[i + r])break;
                }
                if (r < l)dp[i + word.length()] = true;
            }
        }

        return dp[s.length()];
    }

    //134. Gas Station
    int canCompleteCircuit(vector<int>& gas, vector<int>& cost) {
        return -1;
    }

    int getDifference(const vector<vector<int>>& grid,int r,int c) {
        if (grid.size() <= r || grid.size() <= c)return 0;
        return grid[r][c];
    }

    //892. Surface Area of 3D Shapes
    int surfaceArea(const vector<vector<int>>& grid) {
        const uint16_t SIZE = grid.size();
        int result = SIZE * SIZE;
        for (uint16_t r = 0; r < SIZE; ++r) {
            for (uint16_t c = 0; c < SIZE; ++c) {
                result += max(0, grid[r][c] - getDifference(grid, r - 1, c));
                result += max(0, grid[r][c] - getDifference(grid, r + 1, c));
                result += max(0, grid[r][c] - getDifference(grid, r, c - 1));
                result += max(0, grid[r][c] - getDifference(grid, r, c + 1));
            }
        }
        return result;
    }

    //279. Perfect Squares
    int numSquares(int n) {
        static int dp[10001] = { 1 };
        memset(&dp[1], 0, 4 * n);

        for (int k = 0; k < n; ++k) {
            if (dp[k] == 0)continue;
            for (int sqr = 1; sqr <= 100; ++sqr) {
                int square = sqr * sqr;
                if (k + square > n)break;
                if (dp[k + square] == 0)dp[k + square] = dp[k] + 1;
                else dp[k + square] = min(dp[k + square], dp[k] + 1);
            }
        }

        return dp[n] - 1;
    }

    //598. Range Addition II
    int maxCount(int m, int n, vector<vector<int>>& ops) {
        int x = m, y = n;

        for (const vector<int>& range : ops) {
            y = min(range[1], y);
            x = min(range[0], x);
        }

        return x * y;
    }

    //331. Verify Preorder Serialization of a Binary Tree
    bool isValidSerialization(string preorder) {
        static bool stack[100] = {};
        static size_t size;
        size = 0;

        size_t i = 0;
        for (; i < preorder.length(); ++i) {
            if (preorder[i] == ',')continue;
            if (preorder[i] != '#') {
                stack[size++] = true;
                while (i < preorder.length() && preorder[i] != ',')++i;
                continue;
            }
            if (size == 0)break;
            while (size != 0 && !stack[size - 1]) {
                --size;
            }
            if (size == 0)break;
            stack[size - 1] = false;
        }

        return size == 0 && preorder.length() - i < 2;
    }

    //1615. Maximal Network Rank
    int maximalNetworkRank(int n, const vector<vector<int>>& roads) {
        static int edges[100];
        static bool connections[100][100];

        memset(&edges[0], 0, n << 2);
        for (int i = 0; i < n; ++i) {
            memset(&connections[i][0], 0, n);
        }

        for (const vector<int>& road : roads) {
            connections[road[0]][road[1]] = ++edges[road[0]];
            connections[road[1]][road[0]] = ++edges[road[1]];
        }

        int result = 0;
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i + 1; j < n; ++j) {
                result = max(result, edges[i] + edges[j] - int(connections[i][j] || connections[j][i]));
            }
        }

        return result;
    }

    //152. Maximum Product Subarray
    int maxProduct(vector<int>& nums) {
        int answer = INT_MIN;
        int current = 0;
        int prev_negative = 1;

        for (size_t num : nums) {
            if (current == 0) {
                current = 1;
                prev_negative = 1;
            }
            current *= num;
            answer = max(max(current / prev_negative, current), answer);
            if (prev_negative == 1 && num < 0)prev_negative = current;
        }

        return answer;
    }


    //2559. Count Vowel Strings in Ranges
    bool isVowel(char c) {
        return c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u';
    }
    vector<int> vowelStrings(vector<string>& words, vector<vector<int>>& queries) {
        static int sum[100001];

        vector<int> result(queries.size());
        int top = 1;
        for (size_t i = 0; i < queries.size(); ++i) {
            while (top <= queries[i][1] + 1) {
                sum[top++] = sum[top - 1] + int(isVowel(words[top - 1][0]) && isVowel(words[top - 1].back()));
            }
            result[i] = sum[queries[i][1] + 1] - sum[queries[i][0]];
        }
        return result;
    }

    //2454. Next Greater Element IV
    vector<int> secondGreaterElement(vector<int> nums) {
        ios::sync_with_stdio(false);
        cin.tie(NULL);

        static pair<int, int> stack[100000],stack2[100000];
        static int size,size2;
        size = size2 = 0;

        for (size_t i = 0; i < nums.size(); ++i) {

            while (size2 != 0 && stack2[size2 - 1].first < nums[i]) {
                nums[stack2[--size2].second] = nums[i];
            }

            while (size != 0 && stack[size - 1].first < nums[i]) {
                stack2[size2++] = stack[--size];
            }

            stack[size++] = { nums[i],i };
            nums[i] = -1;
        }

        return nums;
    }

    //unsolved
    //
    vector<int> largestDivisibleSubset(vector<int> nums) {
        int max_l = 0, max_r = 0;
        int local_min, local_max;
        int l = 0;

        for (int i = 0; i < nums.size(); ++i) {
            if (l == i) {
                local_min = local_max = nums[i];
                continue;
            }
            if (
                nums[i] % local_max == 0
                || 
                local_min % nums[i] == 0
                || 
                nums[i] % local_max == 0
            ) {
                if (nums[i] > local_max)local_max = nums[i];
                if (nums[i] < local_min)local_min = nums[i];
            }
            else {
                if (max_r - max_l < i - l) {
                    max_r = i;
                    max_l = l;
                }
                l = i + 1;
            }
        }

        return vector<int>(nums.begin() + max_l, nums.begin() + max_r);
    }

    //unsolved
    // topological sort?
    //310. Minimum Height Trees
    vector<int> findMinHeightTrees(int n, vector<vector<int>> edges) {

        return {};
    }

    //105. Construct Binary Tree from Preorder and Inorder Traversal
    //speedup?
    tree_node::TreeNode* dfs(const pair<int,int>* indexed, int l, int r) {
        if (l > r)return nullptr;
        int min_index = l;
        for (int i = l + 1; i <= r; ++i) {
            if (indexed[i].second < indexed[min_index].second)min_index = i;
        }
        return new tree_node::TreeNode(indexed[min_index].first, dfs(indexed, l, min_index - 1), dfs(indexed, min_index + 1, r));
    }

    tree_node::TreeNode* buildTree(const vector<int>& preorder, const vector<int>& inorder) {
        static pair<int, int> indexed[3000];
        unordered_map<int, int> data;
        for (int i = 0; i < preorder.size(); ++i)data[preorder[i]] = i;
        
        for (int i = 0; i < inorder.size(); ++i)indexed[i] = { inorder[i],data[inorder[i]] };

        return dfs(indexed,0,inorder.size()-1);
    }

    //767. Reorganize String
    string reorganizeString(string s) {
        static int frequency[26];
       
        if (s.length() == 1)return s;

        memset(&frequency[0], 0, 104);
        for (char c : s) ++frequency[c - 'a'];

        priority_queue<pair<int, char>> data;
        for (int i = 0; i < 26; ++i) {
            if (frequency[i] > 0)data.push({ frequency[i],i + 'a' });
        }

        string result = "";
        while (data.size() > 1) {
            auto first = data.top(); data.pop();
            auto second = data.top(); data.pop();
            result = (result + first.second) + second.second;
            if (first.first > 1)data.push({ first.first - 1,first.second });
            if (second.first > 1)data.push({ second.first - 1,second.second });
        }

        if (data.size() == 1 && data.top().first == 1 && data.top().second != result.back()) {
            result += data.top().second;
            data.pop();
        }

        return data.size() == 0 ? result : "";
    }

    //354. Russian Doll Envelopes
    //[[4,5],[4,6],[6,7],[2,3],[1,1]]
    int maxEnvelopes(vector<vector<int>> envelopes) {
        static int dp[100000];
        memset(&dp[0], 0, envelopes.size() << 2);
        
        int result = 0;

        sort(envelopes.begin(), envelopes.end(), 
            [](const auto& l, const auto& r) {
                if (l[0] == r[0])return l[1] < r[1];
                return l[0] < r[0]; 
            }
        );

        for (int i = 0; i < envelopes.size(); ++i) {
            result = max(result, dp[i]);
            int index = lower_bound(envelopes.begin() + i + 1, envelopes.end(), envelopes[i], 
                [](const auto& l, const auto& r) { return l[0] < r[0]; }
            ) - envelopes.begin();
            while (index < envelopes.size() && envelopes[index][1] < envelopes[i][1])++index;
            if (index == envelopes.size())continue;
            if (envelopes[index][0] <= envelopes[index][0])continue;
            dp[index] = max(dp[i] + 1, dp[index]);
        }
        
        return result + 1;
    }


    //787. Cheapest Flights Within K Stops
    //TLE
    //unsolved
    int dfs(const vector<vector<vector<int>*>> & data, int current, int destination, int k) {
        if (current == destination)return 0;
        if (k == 0)return -1;

        int result = -1;

        for (int i = 0; i < data[current].size(); ++i) {
            int nested = dfs(data, (*data[current][i])[1], destination, k - 1);
            if (nested == -1)continue;
            nested += (*data[current][i])[2];
            result = result == -1 ? nested : min(result, nested);
        }

        return result;
    }

    int findCheapestPrice(int n, vector<vector<int>>& flights, int src, int dst, int k) {
        vector<vector<vector<int>*>> connections(n);

        for (int i = 0; i < flights.size();++i) {
            connections[flights[i][0]].push_back(&flights[i]);
        }

        return dfs(connections, src, dst, k);
    }

    //680. Valid Palindrome II
    bool validPalindrome(string s,bool can = true) {
        int l = 0, r = s.length() - 1;
        while (l < r) {
            if (s[l] != s[r]) {
                return can ?
                    validPalindrome(s.substr(l + 1, r), false)
                    ||
                    validPalindrome(s.substr(l, r), false)
                    : false;
            }
            else {
                ++l;
                --r;
            }
        }
        return true;
    }

    //68. Text Justification
    vector<string> fullJustify(vector<string> words, int maxWidth) {
        vector<string> result;

        int j, i = 0;
        while (true) {
            int line_length = 0;
            for (j = i; j < words.size();) {
                if (words[j].length() + line_length > maxWidth)break;
                line_length += words[j++].length() + 1;
            }
            line_length += i - j;
            if (j == words.size())break;
            string line;
            int leftOver = maxWidth - line_length == (j - i - 1) ? 0 : (maxWidth - line_length) % max(1, (j - i - 1));
            const string gap = string((maxWidth - line_length) / max(1, (j - i - 1)), ' ');
            while (true) {
                line += words[i++];
                if (i == j)break;
                line += (gap + string(leftOver-- > 0 ? 1 : 0, ' '));
            }
            line += string(max(0, maxWidth - (int)line.length()), ' ');
            result.push_back(line);
        }

        result.push_back("");
        while (i < j) {
            result.back() += words[i++];
            if (result.back().length() < maxWidth)result.back() += ' ';
        }
        result.back() += string(max(0, maxWidth - (int)result.back().length()), ' ');

        return result;
    }

    //1019. Next Greater Node In Linked List
    vector<int> nextLargerNodes(ListNode* head) {
        static stack<pair<int, int>> stack;
        vector<int> answer;

        stack = {};

        int index = 0;
        while (head != nullptr) {
            while (stack.size() != 0 && stack.top().first < head->val) {
                answer[stack.top().second] = head->val;
                stack.pop();
            }
            answer.push_back(0);
            stack.push({ head->val,index++ });
            head = head->next;
        }

        return answer;
    }

    //646. Maximum Length of Pair Chain
    int findLongestChain(vector<vector<int>> pairs) {

        sort(pairs.begin(), pairs.end(), [](const auto& l, const auto& r) {return l[1] < r[1]; });

        int answer = 1;
        int prev = 0;
        for (int i = 1; i < pairs.size(); ++i) {
            if (pairs[i][0] > pairs[prev][1]) {
                prev = i;
                answer += 1;
            }
        }

        return answer;
    }

    //403. Frog Jump
    bool canCross(const vector<int>& stones) {
        unordered_map<int, unordered_map<int, bool>> dp;

        dp[1][1] = true;

        for (int stone : stones) {
            for (auto [jump, _] : dp[stone]) {
                dp[stone + jump - 1][jump - 1] = 1;
                dp[stone + jump][jump] = 1;
                dp[stone + jump + 1][jump + 1] = 1;
            }
        }

        return !dp[stones.back()].empty();
    }

    //1598. Crawler Log Folder
    int minOperations(vector<string>& logs) {
        int size = 0;

        for (const string& log : logs) {
            if (log == "../")size = max(0, size - 1);
            else if (log != "./") size++;
        }

        return size;
    }

    vector<int> findClosestElements(vector<int> arr, int k, int x) {
        int index = lower_bound(arr.begin(), arr.end(), x) - arr.begin();
        index = min(index, (int)arr.size() - 1);

        int l = index, r = index;
        while (r - l + 1 < k) {
            if (l <= 0) ++r;
            else if (r + 1 >= arr.size()) --l;
            else if ((x - arr[l]) <= (arr[r] - x))--l;
            else ++r;
        }

        vector<int> result(k);
        for (int i = 0; i < k; ++i) {
            result[i] = arr[l++];
        }

        return result;

        //prioriy method : 

        //priority_queue <pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> queue;
        //vector<int> result(k);
        //for (int i = 0; i < arr.size(); ++i) {
        //    queue.push({ abs(x - arr[i]), i });
        //}

        //for (int i = 0; i < k; ++i) {
        //    result[i] = arr[queue.top().second];
        //    queue.pop();
        //}
        //sort(result.begin(), result.end());
     
        //return result;
    }

    //2761. Prime Pairs With Target Sum
    vector<vector<int>> findPrimePairs(int n) {
        static vector<int> primes = { 2,3,5,7,11 };

        for (int i = primes.back() + 1; i <= n; ++i) {
            int to = sqrt(i), j = -1;
            while (primes[++j] <= to) {
                if (i % primes[j] == 0)break;
            }
            if (to < primes[j]) {
                primes.push_back(i++);
            }
        }

        vector<vector<int>> result = {};

        int l = 0, r = primes.size() - 1;
        while (l <= r) {
            if (primes[l] + primes[r] > n)--r;
            else if (primes[l] + primes[r] < n)++l;
            else result.push_back({ primes[l++],primes[r--] });
        }

        return result;
    }

    //929. Unique Email Addresses
    int numUniqueEmails(vector<string>& emails) {
        unordered_map<string, bool> unique;

        int answer = 0;

        for (const string& email : emails) {
            string prefix;
            bool skip = false;
            int i = 0;
            for (; email[i] != '@'; ++i) {
                if (email[i] == '.')continue;
                if (email[i] == '+')skip = true;
                if (skip)continue;
                prefix += email[i];
            }
            prefix += email.substr(i, 100);

            if (unique.find(prefix) == unique.end()) {
                ++answer;
                unique.insert({prefix,true});
            }
        }

        return answer;
    }

    //2244. Minimum Rounds to Complete All Tasks
    int minimumRounds(vector<int>& tasks) {
        unordered_map<int, int> count;
        for (int i = 0; i < tasks.size(); ++i) {
            ++count[tasks[i]];
        }

        int rounds = 0;
        for (auto [_, amount] : count) {
            if (amount == 1)return -1;
            rounds += (amount + 2) / 3;
        }
        return rounds;
    }

    //1833. Maximum Ice Cream Bars
    int maxIceCream(vector<int>& costs, int coins) {
        static int data[100001];
        int maxValue = *max_element(costs.begin(), costs.end());

        memset(&data[1], 0, maxValue << 2);

        for (int i = 0; i < costs.size(); ++i) {
            ++data[costs[i]];
        }

        int amount = 0;
        for (int i = 1; i <= maxValue; ++i) {
            if (data[i] == 0)continue;
            if (coins < i)break;

            int d = min(data[i], coins / i);
            coins -= i * d;
            amount += d;
        }
        return amount;
    }


    //224. Basic Calculator
    //first -> value, second -> exit point
    pair<int, int> eval(const string& s, int l) {

        int curr = 0, sum = 0, prev = 0;
        char op = '+';

        while (l < s.size()) {
            if (s[l] == '(') {
                pair<int, int> nested = eval(s, l + 1);
                curr = nested.first;
                l = nested.second;
            }

            if (isdigit(s[l]))curr = curr * 10 + s[l] - '0';
            if ((!isdigit(s[l]) && s[l] != ' ') || l + 1 == s.size() || s[l] == ')') {
                if (op == '+') {
                    sum += prev;
                    prev = curr;
                }
                else if (op == '-') {
                    sum += prev;
                    prev = -curr;
                }
                else if (op == '*')prev = prev * curr;
                else if (op == '/')prev = prev / curr;
                curr = 0;
                op = s[l];
            }
            if (s[l++] == ')')break;
        }
        return { sum + prev,l };
    }

    int calculate(const string& s) {
        return eval(s, 0).first;
    }

    //690. Employee Importance
    class Employee {
    public:
        int id;
        int importance;
        vector<int> subordinates;
    };

    int dfs(unordered_map<int, Employee*>& data, int id) {
        auto employee = data[id];
        int result = employee->importance;
        for (int sub_id : employee->subordinates) {
            result += dfs(data, sub_id);
        }
        return result;
    }

    int getImportance(vector<Employee*> employees, int id) {
        unordered_map<int, Employee*> data;
        for (Employee*& e : employees) {
            data[e->id] = e;
        }

        return dfs(data,id);
    }

    //473. Matchsticks to Square
    bool makesquare(vector<int>& matchsticks) {
        //sort()
        //try to get all length from max to min
        //4 ^ 15
        return false;
    }

    //823. Binary Trees With Factors
    int numFactoredBinaryTrees(vector<int>& arr) {

    }

    //764. Largest Plus Sign
    int orderOfLargestPlusSign(int n, vector<vector<int>>& mines) {

    }

    //1859. Sorting the Sentence
    string sortSentence(string s) {
        map<int, string> data;

        for (int i = 0; i < s.length(); ++i) {
            string current = "";
            int pos = 0;
            while (!isdigit(s[i]))current += s[i++];
            while (isdigit(s[i]))pos += pos * 10 + (s[i++] - '0');
            data.insert({ pos,current });
        }

        string result;
        for (auto itr = data.begin(); itr != data.end();) {
            result += itr->second;
            if (++itr != data.end())result += ' ';
        }

        return result;
    }

    //187. Repeated DNA Sequences
    vector<string> findRepeatedDnaSequences(string s) {
        //0 -> absent 
        //1 -> only 1 occurrence
        //2 -> more than 1 occurrence and already inserted
        unordered_map<int, int> data;
        vector<string> result;

        int slice = 1;
        int sum = 0;
        int r = 0, l = 0;
        while (r < 10 && r < s.length()) {
            slice += s[r] * s[r] * (r+1);
            sum += s[r] * s[r];
            ++r;
        }
        
        data[slice] = 1;
        cout << slice << " : " << sum << "\n";
        while (r < s.length()) {
            slice += (s[r]*s[r]*10) - sum; 
            sum += (s[r] * s[r]) - (s[l] * s[l]);
            cout << slice << " : " << sum << "\n";
            if (data[slice]++ == 1) {
                result.push_back(s.substr(l, 10));
            }
            l += 1;
            r += 1;
        }

        return result;
    }

    //1175. Prime Arrangements
    int numPrimeArrangements(int n) {
        static const vector<int> primes = { 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101 };
        int l = lower_bound(primes.begin(), primes.end(), n + 1) - primes.begin();
        n -= l;

        long long answer = 1;
        while (l > 1) {
            answer = (answer * l--) % 1000000007;
        }
        while (n > 1) {
            answer = (answer * n--) % 1000000007;
        }

        return answer;
    }


    //2225. Find Players With Zero or One Losses
    vector<vector<int>> findWinners(vector<vector<int>>& matches) {
        unordered_map<int, int> data;
        unordered_map<int, bool> players;

        for (const vector<int>& match : matches) {
            ++data[match[1]];
            players[match[0]] = true;
        }

        vector<vector<int>> result(2);
        for (const auto& e : data) {
            if (e.second == 1)result[1].push_back(e.first);
        }
        for (const auto& e : players) {
            if (data[e.first] == 0)result[0].push_back(e.first);
        }
        sort(result[0].begin(), result[0].end());
        sort(result[1].begin(), result[1].end());

        return result;
    }

    //438. Find All Anagrams in a String
    vector<int> findAnagrams(string s, string p) {
        static const int nums[26] = { 912673,941192,970299,1000000,1030301,1061208,1092727,1124864,1157625,1191016,1225043,1259712,1295029,1331000,1367631,1404928,1442897,1481544,1520875,1560896,1601613,1643032,1685159,1728000,1771561,1815848 };
        uint32_t p_sum = 0, s_sum = 0;
        int r = 0;

        while (r < p.length()) {
            p_sum += nums[p[r] - 'a'];
            s_sum += nums[s[r++] - 'a'];
        }

        int l = 0;
        vector<int> result;

        if (p_sum == s_sum)result.push_back(0);
        while (r < s.length()) {
            s_sum += nums[s[r++]-'a'] - nums[s[l++]-'a'];
            if (p_sum == s_sum)result.push_back(l);
        }

        return result;
    }

    //2829. Determine the Minimum Sum of a k-avoiding Array
    int sum(int s, int f) {
        return ((f + s) * (f - s + 1)) / 2;
    }
    int minimumSum(int n, int k) {
        if (k / 2 > n)return sum(1, n);
        return sum(1, k / 2) + sum(k, k + n - k / 2 - 1);
    }

    //2196. Create Binary Tree From Descriptions
    tree_node::TreeNode* createBinaryTree(vector<vector<int>> descriptions) {
        unordered_map<int, tree_node:: TreeNode* > data;
        unordered_map<tree_node::TreeNode*, bool> has_parent;
        
        for (const vector<int>& x : descriptions) {
            if (data[x[0]] == nullptr)data[x[0]] = new tree_node::TreeNode(x[0]);
            if (data[x[1]] == nullptr)data[x[1]] = new tree_node::TreeNode(x[1]);
            if (x[2])data[x[0]]->left = data[x[1]];
            else data[x[0]]->right = data[x[1]];

            has_parent[data[x[0]]] |= false;
            has_parent[data[x[1]]] |= true;

            for (const auto& [k, v] : has_parent) {
                cout << "(" << k->val << ":" << v << ")";
            }
            cout << "\n";

        }
        
        tree_node::TreeNode* root = nullptr;
        for (const auto& [k, v] : has_parent) {
            if (!v) {
                root = k;
                break;
            }
        }
        return root;
    }

    //1834. Single-Threaded CPU
    class PairComparer {
    public:
        bool operator()(const pair<int, int>& l, const pair<int, int>& r) {
            if (l.first == r.first)return l.second > r.second;
            return l.first > r.first;
        }
    };

    vector<int> getOrder(vector<vector<int>> tasks) {
        static priority_queue<pair<int, int>, vector <pair<int, int>>, PairComparer> queue;
        static vector<pair<int, vector<int>*>> sorted_data(100000);
        const int N = tasks.size();
        for (int i = 0; i < N; ++i) {
            sorted_data[i] = { i,&tasks[i] };
        }

        sort(sorted_data.begin(), sorted_data.begin() + N, [](const auto& l, const auto& r) {
            if ((*l.second)[0] == (*r.second)[0])return (*l.second)[1] < (*r.second)[1];
            return (*l.second)[0] < (*r.second)[0];
        });

        vector<int> result(tasks.size());
        int time = (*sorted_data[0].second)[0];
        int index = 0;
        
        for (int i = 0; i < N;) {
            if ((*sorted_data[i].second)[0] <= time) {
                queue.push({ (*sorted_data[i].second)[1],sorted_data[i++].first });
                continue;
            }
            if (time <= (*sorted_data[i].second)[0]) {
                if (queue.empty()) {
                    time = (*sorted_data[i].second)[0];
                    continue;
                }
                result[index++] = queue.top().second;
                time += queue.top().first;
                queue.pop();
            }
        }

        while (queue.size() != 0) {
            result[index++] = queue.top().second;
            queue.pop();
        }

        return result;
    }

    //2707. Extra Characters in a String
    int minExtraChar(string s, vector<string> dictionary) {
        static char dp[51] = {};
        memset(&dp[1], 100, s.length());

        for (int i = 0; i < s.length(); ++i) {
            for (const string& word : dictionary) {
                if (word.length() + i > s.length())continue;
                int l = -1, r = word.length();
                while (++l <= --r) {
                    if (word[l] != s[i + l] || word[r] != s[i + r])break;
                }
                if (r < l) {
                    dp[i + word.length()] = min(dp[i + word.length()], dp[i]);
                }
            }
            dp[i + 1] = min(dp[i] + 1, int(dp[i + 1]));
        }

        return dp[s.length()];
    }

    //140. Word Break II
    vector<string> assemble(const string& s, const vector<char> routes[], char index) {
        vector<string> result;

        for (char prev_index : routes[index]) {
            const string sub = s.substr(prev_index, index - prev_index);
            if (prev_index == 0) {
                result.push_back(sub);
                continue;
            }
            for (string nested : assemble(s, routes, prev_index)) {
                result.push_back(nested + ' ' + sub);
            }
        }

        return result;
    }

    vector<string> wordBreak(string s, vector<string> wordDict) {
        static vector<char> dp[21];

        for (int i = 1; i <= s.length(); ++i) {
            dp[i].clear();
        }

        for (int i = 0; i < s.length(); ++i) {
            if (i > 0 && dp[i].size() == 0)continue;

            for (const string& word : wordDict) {
                if (word.length() + i > s.length())continue;
                int l = -1, r = word.length();
                while (++l <= --r) {
                    if (word[l] != s[i + l] || word[r] != s[i + r])break;
                }
                if (r < l) {
                    dp[i + word.length()].push_back(i);
                }
            }
        }

        return assemble(s, dp, s.length());
    }

    //804. Unique Morse Code Words
    int uniqueMorseRepresentations(vector<string>& words) {
        static const string morseLetters[26] = { ".-","-...","-.-.","-..",".","..-.","--.","....","..",".---","-.-",".-..","--","-.","---",".--.","--.-",".-.","...","-","..-","...-",".--","-..-","-.--","--.." };
        unordered_map<string, bool> data;

        for (const string& word : words) {
            string combination;
            for (char c : word) {
                combination += morseLetters[c - 'a'];
            }
            data.insert({ combination,true });
        }

        return data.size();
    }

    //164. Maximum Gap
    int maximumGap(vector<int> nums) {
        static bool data[100001];

        if (nums.size() < 2)return 0;
        if (nums.size() == 2)return abs(nums[0] - nums[1]);

        auto[min,max] = minmax_element(nums.begin(), nums.end());
        memset(&data[*min], 0, *max);

        for (int i = 0; i < nums.size(); ++i) {
            data[nums[i]] = 1;
        }

        int result = 0;
        for (int i = *min; i < *max;) {
            int next = i + 1;
            while (!data[next])++next;
            result = std::max(result, next - i);
            i = next;
        }

        return result;
    }


    //
    node_138::Node* copyRandomList(node_138::Node* head) {
        static node_138::Node* new_list[1001];
        static int size = 0;
        if (head == nullptr)return nullptr;

        unordered_map<node_138::Node*, int> numbered;
        
        node_138::Node* itr = head;
        while (itr != nullptr) {
            numbered[itr] = size;
            new_list[size++] = new node_138::Node(itr->val);
            itr = itr->next;
        }

        itr = head;
        for (int i = 0; i < size; ++i) {
            new_list[i]->next = new_list[i+1];
            if (itr->random != nullptr)new_list[i]->random = new_list[numbered[itr->random]];
            itr = itr->next;
        }
        
        return new_list[0];
    }

    //174. Dungeon Game
    //[0 - 74 - 47 - 20 - 23 - 39 - 48]
    //[37 - 30  37 - 65 - 82  28 - 27]
    //[-76 - 33   7  42   3  49 - 93]
    //[37 - 41  35 - 16 - 96 - 56  38]
    //[-52  19 - 37  14 - 65 - 42   9]
    //[5 - 26 - 30 - 65  11   5  16]
    //[-60   9  36 - 36  41 - 47 - 86]
    //[-22  19 - 5 - 41 - 8 - 96 - 95]
    int calculateMinimumHP(vector<vector<int>> dungeon) {
        vector<vector<int>> min_hp(dungeon.size(), vector<int>(dungeon[0].size(), INT_MAX));

        min_hp[0][0] = dungeon[0][0];

        for (int r = 1; r < dungeon.size(); ++r) {
            dungeon[r][0] += dungeon[r - 1][0];
            min_hp[r][0] = min(dungeon[r - 1][0], dungeon[r][0]);
        }

        for (int c = 1; c < dungeon[0].size(); ++c) {
            dungeon[0][c] += dungeon[0][c - 1];
            min_hp[0][c] = min(dungeon[0][c - 1], dungeon[0][c]);
        }

        for (int r = 1; r < dungeon.size(); ++r) {
            for (int c = 1; c < dungeon[0].size(); ++c) {
                if (dungeon[r - 1][c] > dungeon[r][c - 1]) {
                    dungeon[r][c] += dungeon[r - 1][c];
                    min_hp[r][c] = min(dungeon[r][c], min_hp[r - 1][c]);
                }
                else {
                    dungeon[r][c] += dungeon[r][c - 1];
                    min_hp[r][c] = min(dungeon[r][c], min_hp[r][c - 1]);
                }
            }
        }

        return abs(min_hp.back().back()) + int(dungeon[0][0] == 0);
    }

    //741. Cherry Pickup
    int maxIncreaseKeepingSkyline(vector<vector<int>>& grid) {
        static char rows[100], columns[100];
        const int R = grid.size();
        const int C = grid[0].size();
        memset(&rows[0], 0, R);
        memset(&columns[0], 0, C);

        for (int r = 0; r < R; ++r) {
            for (int c = 0; c < C; ++c) {
                rows[r] = max(char(grid[r][c]), rows[r]);
                columns[c] = max(char(grid[r][c]), columns[c]);
            }
        }

        int result = 0;
        for (int r = 0; r < R; ++r) {
            for (int c = 0; c < C; ++c) {
                result += max(0, (min(rows[r], columns[c])) - char(grid[r][c]));
            }
        }

        return result;
    }

    bool canReach(string s, int minJump, int maxJump) {
        static bool dp[100001] = { 1 };
        const int LEN = s.length();
        memset(&dp[1], 0, LEN);
        
        //keep all next or previous reachable spots in range of [i + minJump,i + maxJump]

        for (int i = 0; i < LEN; ++i) {
            if (!dp[i])continue;
            for (int j = minJump; j <= maxJump; ++j) {
                if (j + i <= LEN && s[j + i] == '0')dp[j + i] = true;
            }
        }

        return dp[s.length() - 1];
    }

    //1390. Four Divisors
    struct RecursiveDivisor {
        int num;
        int level;
        int sum;
    };

    RecursiveDivisor* dfs(unordered_map<int, RecursiveDivisor>& memorize, int num) {
        //TODO:
        //describe way to resolve request about number
    }

    int sumFourDivisors(vector<int>& nums) {
        //defines memorization structure : 
        //  key -> the same number
        //  value :
        //          first -> level of diviors
        //          second -> prev exsiting number
        static unordered_map<int, RecursiveDivisor> memorize;

        int answer = 0;
        for (int i = 0; i < nums.size(); ++i) {
            auto res = dfs(memorize, nums[i]);
            if (res->level == 3)answer += res->sum;
        }
        return answer;
    }

    //725. Split Linked List in Parts
    int list_length(ListNode*& head) {
        ListNode* itr = head;
        int len = 0;

        while (itr != nullptr) {
            ++len;
            itr = itr->next;
        }

        return len;
    }
    vector<ListNode*> splitListToParts(ListNode* head, int k) {
        int length = list_length(head);
        int container_len = length / k;
        int leftover = length % k;

        vector<ListNode*> result(k, nullptr);

        for (int i = 0; i < k; ++i) {
            result[i] = head;
            for (int j = 1; j < container_len; ++j) {
                head = head->next;
            }
            if (leftover-- > 0 && container_len > 0) {
                head = head->next;
            }
            if (head == nullptr)break;
            ListNode* temp = head->next;
            head->next = nullptr;
            head = temp;
        }

        return result;
    }

    //962. Maximum Width Ramp
    int maxWidthRamp(vector<int>& nums) {
        map<int, int,greater<int>> data;

        int answer = 0;
        for (int i = 0; i < nums.size(); ++i) {

            auto itr = data.lower_bound(nums[i]);
            while (itr != data.end()) {
                answer = max(answer, i - itr->second);
            }

            if (data.find(nums[i]) == data.end())data[nums[i]] = i;
        }
        return answer;
    }

    //149. Max Points on a Line
    struct PairHash {
        auto operator()(const pair<float, float>& data) const {
            return hash<float>{}(data.first) ^ hash<float>{}(data.second);
        }
    };

    int maxPoints(vector<vector<int>> points) {
        unordered_map<pair<float, float>, unordered_map<int, bool>, PairHash> coofs;

        int answer = 1;

        bool is_zero = false;
        for (int i = 0; i < points.size(); ++i) {
            if (points[i][0] == 0 && points[i][1] == 0) {
                is_zero = true;
                continue;
            }
            for (int j = i + 1; j < points.size(); ++j) {
                double k = abs(points[i][1] - points[j][1] * 1.0) / abs(points[i][0] - points[j][0]);
                double b = points[i][1] - k * points[i][0];
                coofs[{k, b}][j] = coofs[{k, b}][i] = true;
                answer = max(answer, (int)coofs[{k, b}].size());
            }
        }

        return answer;
    }

    //611. Valid Triangle Number
    int triangleNumber(vector<int> nums) {
        sort(nums.begin(), nums.end());

        const int size = nums.size();
        int result = 0;

        for (int i = 0; i < size; ++i) {
            int l = i + 1, r = size - 1;
            while (l < r) {
                if (nums[l] + nums[i] < nums[r])++l;
                else break;
            }
            result += max(0, r - l);
        }

        return result;
    }

    //950. Reveal Cards In Increasing Order
    vector<int> deckRevealedIncreasing(vector<int> deck) {
        sort(deck.begin(), deck.end());

        if (deck.size() < 3)return deck;

        deque<int> data;
        int i = deck.size();
        data.push_front(deck[--i]);
        data.push_front(deck[--i]);

        while (0 < i) {
            data.push_front(data.back());
            data.pop_back();
            data.push_front(deck[--i]);
        }

        return vector<int>(data.begin(), data.end());
    }

    //838. Push Dominoes
    string pushDominoes(string& dominoes) {
        int l = 0, r = 1;
        while (r < dominoes.length()) {
            while (r + 1 < dominoes.length() && dominoes[r] == '.')++r;

            if (dominoes[l] == 'R' && dominoes[r] == 'L') {
                int len = (r - l - 1) / 2;
                memset(&dominoes[l + 1], 'R', len);
                memset(&dominoes[r - len], 'L', len);
            }
            else if (dominoes[r] == 'L') {
                memset(&dominoes[l], 'L', r - l + 1);
            }
            else if (dominoes[l] == 'R') {
                memset(&dominoes[l], 'R', r - l + 1);
            }

            l = r++;
        }

        return dominoes;
    }
    
    //1282. Group the People Given the Group Size They Belong To
    vector<vector<int>> groupThePeople(vector<int> groupSizes) {
        static unordered_map<int, vector<int>> groups;

        vector<vector<int>> result;
        for (int i = 0; i < groupSizes.size(); ++i) {
            auto group = &groups[groupSizes[i]];

            group->push_back(i);

            if (group->size() == groupSizes[i]) {
                result.push_back(*group);
                group->clear();
            }
        }
        return result;
    }
};


//432. All O`one Data Structure
struct FrequencyNode {
    uint16_t count;
    string data;
    FrequencyNode* prev;
    FrequencyNode* next;

    FrequencyNode(uint16_t count, const string& data, FrequencyNode* next = 0) :
        count(count),
        data(data),
        next(next),
        prev(0)
    {};
};

class AllOne {
    unordered_map<string, FrequencyNode*> strings;
    FrequencyNode* head, * tail;
public:
    AllOne() {
        head = tail = 0;
    }

    void inc(string key) {
        if (strings.find(key) == strings.end()) {
            auto itr = strings.insert({ key,0 }).first;
            if (head == nullptr) {
                tail = head = new FrequencyNode(1, itr->first);
            }
            else {
                head = new FrequencyNode(1, itr->first, head);
                head->next->prev = head;
            }
            itr->second = head;
            return;
        }

        ++strings[key]->count;
        resolve(key);
    }

    void dec(string key) {
        FrequencyNode* node = strings[key];
        if (--node->count == 0) {
            if (node == tail)tail = node->prev;
            if (node->prev != nullptr) {
                node->prev->next = node->next;
            }
            else {
                head = node->next;
                node->next->prev = nullptr;
            }
            strings.erase(key);
            return;
        }

        if (node->prev != nullptr) {
            resolve(strings.find(node->prev->data)->first);
        }
    }

    string getMaxKey() {
        return tail == nullptr ? "" : tail->data;
    }

    string getMinKey() {
        return head == nullptr ? "" : head->data;
    }

private:
    void resolve(const string& key) {
        FrequencyNode* node = strings[key];
        FrequencyNode* itr = node;

        while (itr->next != nullptr && itr->next->count < node->count) {
            itr = itr->next;
        }

        if (itr != node) {
            swap(strings[key], strings[itr->data]);
            swap(node->count, itr->count);
            swap(node->data, itr->data);
        }
    }
};

//1114. Print in Order
class FooBar {
private:
    int n;
    promise<void>* promise;
public:
    FooBar(int n) {
        this->n = n;
    }

    void foo(function<void()> printFoo) {

        for (int i = 0; i < n; i++) {

            // printFoo() outputs "foo". Do not change or remove this line.
            printFoo();
        }
    }

    void bar(function<void()> printBar) {

        for (int i = 0; i < n; i) {
                
            printBar();
        }
    }
};

//1381. Design a Stack With Increment Operation
class CustomStack {
    static int stack[1000];
    static int increments[1000];
    int size;
    int maxSize;
public:
    CustomStack(int maxSize) {
        memset(&increments[0], 0, maxSize << 2);
        this->maxSize = maxSize;
        size = 0;
    }

    void push(int x) {
        if (size == maxSize)return;
        stack[size++] = x;
    }

    int pop() {
        if (size == 0)return -1;
        int result = stack[--size] + increments[size];
        if (size != 0)increments[size - 1] += increments[size];
        increments[size] = 0;
        return result;
    }

    void increment(int k, int val) {
        increments[max(0, min(size - 1, k - 1))] += val;
    }
};

int CustomStack::stack[1000] = {};
int CustomStack::increments[1000] = {};

//729. My Calendar I
class MyCalendar {
    map<int, int> data;
public:
    MyCalendar() {
        data = { };
    }

    bool book(int start, int end) {
        for (auto [s, e] : data) {
            if (e < start)continue;

            if (
                s <= start && end <= e             ||
                start < s && e < end               ||
                s <= start && start < e && e < end ||
                end <= e && s < end && start < s
            )return false;

            if (end < s)break;
        }
        data[start] = end;
        return true;
    }
};