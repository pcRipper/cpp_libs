#pragma once
#include "../../../GeneralLibs/Headers/includes.hpp";
#include "DataStructures.hpp"

#define MODULO 1000000007
#define DEBUG true
#define ll long long


class Solution {
public:
    //473. Matchsticks to Square
    bool makesquare(vector<int>& matchsticks) {

        return false;
    }

    //2063. Vowels of All Substrings
    bool is_vowel(char c) {
        return c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u';
    }

    long long countVowels(string word) {
        long long result = 0;
        long long last_vowel = 0;
        long long last_diff = 0;

        const int LEN = word.length();
        for (int i = 0; i < LEN; ++i) {
            if (is_vowel(word[i])) {
                last_diff = -result;
                result += i + 1 + last_vowel;
                last_vowel = result;
                last_diff += result;
            }
            else {
                result += last_diff;
            }
        }

        return result;
    }

    //115. Distinct Subsequences
    int numDistinct(string s, string t) {
        static size_t dp[1001] = { 1 };
        memset(&dp[1], 0, 4000);

        unordered_map<char, vector<int>> ref;

        for (int i = t.length() - 1; 0 <= i; --i) {
            ref[t[i]].push_back(i);
        }

        for (int i = 0; i < s.length(); ++i) {
            for (int index : ref[s[i]]) {
                dp[index + 1] += dp[index];
            }
        }

        return dp[t.length()];
    }

    //1792. Maximum Average Pass Ratio
    class RatioCompare {
    public:
        bool operator() (const pair<int, int>& l, const pair<int, int>& r) {
            return
                static_cast<double>(l.first) / static_cast<double>(l.second)
                <
                static_cast<double>(r.first) / static_cast<double>(r.second)
                ;
        }
    };

    double maxAverageRatio(vector<vector<int>>& classes, int extraStudents) {
        static priority_queue<pair<int,int>,vector<pair<int,int>>,RatioCompare> queue;

        for (const auto& _class : classes) {
            queue.push(make_pair(_class[0], _class[1]));
        }

        return 0;
    }

    //493. Reverse Pairs
    int reversePairs(vector<int>& nums) {
        static vector<long long> sorted;
        const int SIZE = nums.size();

        FenwickTree f(SIZE);

        for (int i = 0; i < SIZE; ++i)sorted[i] = nums[i];

        sort(sorted.begin(), sorted.begin() + SIZE, greater<long long>());

        int pairs = 0;
        for (int i = 0; i < nums.size(); ++i) {
            int amount_of_greater = lower_bound(sorted.begin(), sorted.end(), static_cast<long long>(nums[i]) * 2, greater<long long>()) - sorted.begin();
            int index = lower_bound(sorted.begin(), sorted.end(), nums[i], greater<long long>()) - sorted.begin();
            pairs += f.query(amount_of_greater);
            f.update(index + 1);
        }

        return pairs;
    }

    //315. Count of Smaller Numbers After Self
    void merge(vector<pair<int,int>>& nums, vector<int>& result,int l, int r) {
        static pair<int, int> temp[100000];
        if (nums.size() < 2)return;

        int middle = l + (r - l) / 2;
        vector<pair<int, int>> left = vector<pair<int, int>>(nums.begin(), nums.begin() + middle + 1);
        vector<pair<int, int>> right = vector<pair<int, int>>(nums.begin() + middle + 1, nums.end());
    
        merge(left, result, l, middle);
        merge(right, result, middle + 1, r);

        int li = 0, ri = 0, ni = 0;
        while (li < left.size() && ri < right.size()) {
            if (left[li].first <= right[ri].first) {
                result[left[li].second] += ri;
                nums[ni] = left[li++];
            }
            else {
                nums[ni] = right[ri++];
            }
            ++ni;
        }
        while (li < left.size()) {
            result[left[li].second] += ri;
            nums[ni++] = left[li++];
        }
        while (ri < right.size()) {
            nums[ni++] = right[ri++];
        }
    }

    vector<int> countSmaller(vector<int> nums) {
        vector<int> result(nums.size());
        vector<pair<int, int>> indexed(nums.size());

        for (int i = 0; i < nums.size(); ++i) {
            indexed[i] = { nums[i], i };
        }

        merge(indexed, result,0, nums.size() - 1);

        return result;
    }

    //406. Queue Reconstruction by Height
    vector<vector<int>> reconstructQueue(vector<vector<int>>& people) {
        list<vector<int>> list;

        sort(people.begin(), people.end(),
            [](const auto& l, const auto& r) {
                return l[1] == r[1] ? (l[1] == 0 ? l[0] < r[0] : l[0] > r[0]) : l[1] < r[1];
            }
        );

        int i = 0;
        while (i < people.size() && people[i][1] == 0) {
            list.push_back(people[i++]);
        }

        while (i < people.size()) {
            auto itr = list.begin();
            int greater = 0;
            while (itr != list.end() && greater < people[i][1]) {
                if ((*itr)[0] >= people[i][0])++greater;
                ++itr;
            }
            list.insert(itr, people[i++]);
        }

        return vector<vector<int>>(list.begin(),list.end());
    }

    //
    int countTexts(string pressedKeys) {
        size_t a1[4] = { 0,0,0,1 };
        size_t a2[4] = { 0,0,0,0 };

        size_t* current = a1, * prev = a2;

        char prev_c = 0;
        int consecutive = 0;
        for (int i = 0; i < pressedKeys.size(); ++i) {
            if (pressedKeys[i] == prev_c) {
                ++consecutive;
                if (prev_c == '9' || prev_c == '7') {
                    if (consecutive > 4) prev[3] = (current[3] + current[2] + current[1] + current[0]) % 1000000007;
                    else prev[3] = (current[3] * 2) % 1000000007;
                }
                else {
                    if (consecutive > 3) prev[3] = (current[3] + current[2] + current[1]) % 1000000007;
                    else prev[3] = (current[3] * 2) % 1000000007;
                }

                prev_c = pressedKeys[i];
                memcpy(&prev[0], &current[1], 24);
                swap(prev, current);
            }
            else consecutive = 1;

            prev_c = pressedKeys[i];
        }

        return static_cast<int>(current[3]);
    }

    vector<string> findWords(vector<vector<char>> board, vector<string> words) {
        static bool used[30000];
        static const array<pair<int, int>, 4> directions = { 
            pair<int,int>(0,1),
            pair<int,int>(1,0),
            pair<int,int>(0,-1),
            pair<int,int>(-1,0)
        };
        memset(&used[0], 0, words.size());

        unordered_map<char, vector<pair<int,bool>>> word_begin;

        for (int i = 0; i < words.size(); ++i) {
            word_begin[words[i][0]].push_back(make_pair(i, 1));
            word_begin[words[i].back()].push_back(make_pair(i, 0));
        }

        const int ROWS = board.size();
        const int COLUMNS = board[0].size();
        //array of three elements : 
        //                          index of the word in the array words
        //                          index of the current char in the word
        //                          iterator for the word : 1 or -1
        vector<vector<vector<array<int, 3>>>> dp(ROWS, vector<vector<array<int, 3>>>(COLUMNS));
        
        for (int r = 0; r < ROWS; ++r) {
            for (int c = 0; c < COLUMNS; ++c) {
                //iteratre through the adjacent cells in order to continue the sequence
                for (const auto& pair : directions) {
                    int nr = r + pair.first;
                    int nc = c + pair.second;
                    if (nr < 0 || nr >= ROWS || nc < 0 || nc >= COLUMNS)continue;
                    for (const auto& prev : dp[nr][nc]) {
                        if ((prev[1] == 0 && prev[2] == -1) || (prev[1] + 1 == words[prev[0]].length() && prev[2] == 1))continue;
                        if (board[r][c] == words[prev[0]][prev[1] + prev[2]]) {
                            dp[r][c].push_back({ prev[0],prev[1] + prev[2],prev[2] });
                        }
                    }
                }
                //iterate through the first/last letters of the words
                for (const auto& pair : word_begin[board[r][c]]) {
                    if (!used[pair.first]) {
                        dp[r][c].push_back(array<int, 3>{
                            pair.first,
                                pair.second ? 0 : int(words[pair.first].length() - 1),
                                pair.second ? 1 : -1
                        });
                    }
                }
                //iterate through the current states to detect complete words
                for (const auto& state : dp[r][c]) {
                    if ((state[1] == 0 && state[2] == -1) || (state[1] + 1 == words[state[0]].length() && state[2] == 1)) {
                        used[state[0]] = true;
                    }
                }
            }
        }

        vector<string> result;
        for (int i = 0; i < words.size(); ++i) {
            if (used[i])result.push_back(words[i]);
        }

        return result;
    }

    //721. Accounts Merge
    vector<vector<string>> accountsMerge(vector<vector<string>> accounts) {
        unordered_map<string, pair<bool, vector<int>>> emails;

        for (int j = 0; j < accounts.size();++j) {
            for (int i = 1; i < accounts[j].size(); ++i) {
                if (emails.find(accounts[j][i]) == emails.end()) {
                    emails.insert({ accounts[j][i], {false,{} } });
                }
                emails[accounts[j][i]].second.push_back(j);
            }
        }

        show_unordered_map(emails,0b011);

        Unions unions(accounts.size());

        for (const auto& [_, g] : emails) {
            for (int i = 0; i < g.second.size(); ++i) {
                for (int j = i + 1; j < g.second.size(); ++j) {
                    unions.insert(g.second[i], g.second[j]);
                }
            }
        }
        
        auto ga = unions.groups_amount();
        vector<vector<string>> result;
        result.reserve(ga.second + ga.first);
        unordered_map<int, int> groups;

        for (int j = 0; j < accounts.size(); ++j) {
            int group = unions.find(emails[accounts[j][1]].second[0]);
            if (group == 0) {
                result.push_back(accounts[j]);
                continue;
            }
            if (groups.find(group) == groups.end()) {
                groups[group] = result.size();
                result.push_back({accounts[j][0]});
            }
            for (int i = 1; i < accounts[j].size(); ++i) {
                if (emails[accounts[j][i]].first)continue;
                emails[accounts[j][i]].first = true;
                result[groups[group]].push_back(accounts[j][i]);
            }
        }

        return result;
    }

    //587. Erect the Fence
    bool crossProduct(const vector<int>& l, const vector<int>& m, const vector<int>& r) {
        return (m[0] - l[0]) * (r[1] - l[1]) - (m[1] - l[1]) * (r[0] - l[0]) >= 0;
    }

    vector<vector<int>> outerTrees(vector<vector<int>>& trees) {
        static vector<pair<int, int>> sorted(3000);
        const int SIZE = trees.size();
        int lowest = min_element(trees.begin(), trees.end(),
            [](const auto& l, const auto& r) {
                return l[1] < r[1];
            }) - trees.begin();


        return {};
    }

    //1155. Number of Dice Rolls With Target Sum
    int numRollsToTarget(int n, int k, int target) {
        static int dp[931];
        if (n * k < target)return 0;
        
        dp[0] = 1;
        memset(&dp[1], 0, target << 2);

        int max_sum = 0;
        for (int i = 0; i < n; ++i) {
            for (int j = min(target, max_sum); i <= j; --j) {
                for (int l = 1; l <= k; ++l) {
                    dp[j + l] = (dp[j + l] + dp[j]) % MODULO;
                }
                dp[j] = 0;
            }
            max_sum += k;
        }

        return dp[target];
    }

    //494. Target Sum
    int findTargetSumWays(vector<int> nums, int target) {
        static unordered_map<int, int> f, s;
        
        f.clear();
        f[0] = 1;

        unordered_map<int, int>* first = &f, * second =&s;
        for (int i = 0; i < nums.size(); ++i) {
            second->clear();
            for (const auto& pair : *first) {
                (*second)[pair.first + nums[i]] += pair.second;
                (*second)[pair.first - nums[i]] += pair.second;
            }
            swap(first, second);
        }

        return (*first)[target];
    }

    //1531. String Compression II
    //unfinished
    int calcLen(const pair<int, int>& p) {
        return p.first == 0 ? 0 : p.second + (int)ceil(log10(p.first)) + 1;
    }
    int getLengthOfOptimalCompression(string s, int k) {
        //dp matrix, where map : ( last char, (amount of this chars, general length))
        vector<vector<vector<pair<char,pair<int,int>>>>> dp(
            s.length() + 1,
            vector<vector<pair<char, pair<int,int>>>>(k + 1)
        );
        dp[0][0].push_back({ -1, {0, 0} });

        for (int i = 0; i < s.length(); ++i) {
            for (int j = 0; j < dp[i].size(); ++j) {
                if (dp[i][j].size() == 0)continue;
                
                int next_len = dp[i + 1][j].size() == 0 ? 100 : calcLen(dp[i + 1][j][0].second);
                int next_len_miss = j + 1 > k ? -1 : dp[i + 1][j + 1].size() == 0 ? 100 : calcLen(dp[i+1][j+1][0].second);

                for (const auto& [c, lens] : dp[i][j]) {
                    //adding current symbol
                    pair<int, int> next_comb = c == s[i] ? 
                        make_pair(lens.first + 1, lens.second) 
                        :
                        make_pair(1 ,calcLen(lens))
                    ;
                    int current_len = calcLen(next_comb);
                    if (current_len < next_len) {
                        dp[i + 1][j].clear();
                        next_len = current_len;
                    }
                    if (next_len == current_len) {
                        dp[i + 1][j].push_back({ s[i], next_comb });
                    }

                    //skipping current symbol
                    if (j + 1 > k)continue;
                    int current_len_miss = calcLen(lens);
                    if (current_len_miss < next_len_miss) {
                        dp[i + 1][j + 1].clear();
                        next_len_miss = current_len_miss;
                    }
                    if (current_len_miss == next_len_miss) {
                        dp[i + 1][j + 1].push_back({ c, lens });
                    }
                }

                //show_vector(dp[i + 1][j],0b011);
            }
            //cout << string(10,'-');
        }

        return calcLen(dp.back().back()[0].second);
    }

    //1335. Minimum Difficulty of a Job Schedule
    int minDifficulty(vector<int>& jobDifficulty, int d) {
        static int dp[300][10];
        const int SIZE = jobDifficulty.size();

        if (SIZE < d) return -1;

        dp[0][0] = jobDifficulty[0];
        for (int i = 1; i < SIZE; ++i) {
            memset(&dp[i][1], 0b01111111, 36);
            dp[i][0] = max(dp[i - 1][0], jobDifficulty[i]);
        }

        for (int day = 1; day < d; ++day) {
            for (int i = day; i < SIZE; ++i) {
                int maxDifficulty = 0;
                for (int j = i; j >= day; --j) {
                    maxDifficulty = max(maxDifficulty, jobDifficulty[j]);
                    dp[i][day] = min(dp[i][day], dp[j - 1][day - 1] + maxDifficulty);
                }
            }
        }

        return dp[SIZE - 1][d - 1];
    }

    //845. Longest Mountain in Array
    int longestMountain(vector<int> arr) {
        int res = 0;

        const int SIZE = static_cast<int>(arr.size());
        for (int i = 0; i < SIZE; ++i) {
            int l = i + 1;
            while (l < SIZE && arr[l] > arr[l - 1])++l;
            if (l == i + 1)continue;
            if (l == SIZE)break;
            while (l < SIZE && arr[l] < arr[l - 1])++l;
            res = max(res, i - l);
            i = l - 1;
        }

        return res;
    }

//1976. Number of Ways to Arrive at Destination
#define MODULO 1000000007
#define MAX_NODES 200

    int countPaths(int n, vector<vector<int>> roads) {
        static pair<uint64_t, int> dp[MAX_NODES];
        static priority_queue<pair<uint64_t,int>,vector<pair<uint64_t, int>>, greater<pair<uint64_t, int>>> queue;
        vector<vector<pair<int, uint64_t>>> edges(n);

        memset(dp, 0xff, 3200);
        dp[0] = { 0, 1 };

        for (const auto& road : roads) {
            edges[road[0]].push_back({ road[1],road[2] });
            edges[road[1]].push_back({ road[0],road[2] });
        }
        
        queue.push({ 1, 0 });
        while (queue.size() > 0) {
            const auto [val, node] = queue.top(); queue.pop();
            if (node == n - 1)continue;
            for (const auto& pair : edges[node]) {
                uint64_t sum = val + pair.second;
                if (sum < dp[pair.first].first) {
                    queue.push({ sum, pair.first });
                    dp[pair.first] = { sum, 0 };
                }
                if (sum == dp[pair.first].first) {
                    dp[pair.first].second = (dp[pair.first].second + dp[node].second) % MODULO;
                }
            }
        }

        return dp[n-1].second;
    }

    //300. Longest Increasing Subsequence
    int lengthOfLIS(vector<int> nums) {
        map<int, int, greater<int>> dp;

        int result = 0;
        for (int i = 0; i < nums.size(); ++i) {
            auto itr = dp.lower_bound(nums[i] - 1);
            int local_max = 0;
            while(itr != dp.end()){
                local_max = max(local_max, itr->second);
                ++itr;
            }
            dp[nums[i]] = local_max + 1;
            result = max(result, local_max + 1);
            show_map(dp);
        }
        return result;
    }

    //1235. Maximum Profit in Job Scheduling
    //optimized
    int binaryJob(const pair<int, int>* data,const int SIZE, const int VALUE) {
        
        int l = 0, r = SIZE - 1;
        
        while (l < r) {
            int middle = l + (r - l) / 2;
            if (data[middle].first >= VALUE)r = middle;
            else l = middle + 1;
        }

        return data[l].first > VALUE ? l - 1 : l;
    }

#define MAX_EVENTS 50000
    int jobScheduling(vector<int> startTime, vector<int> endTime, vector<int> profit) {
        static vector<int>    indices(MAX_EVENTS);
        static pair<int, int>  dp[MAX_EVENTS] = { {1,0} };
        const int SIZE = startTime.size();

        for (int i = 0; i < SIZE; ++i) {
            indices[i] = i;
        }

        sort(indices.begin(), indices.begin() + SIZE, 
            [&endTime](int l, int r) {return endTime[l] < endTime[r]; }
        );
        
        int current_size = 1;
        int result = 0;
        for (int i = 0; i < SIZE; ++i) {
            int index = indices[i];
            if (current_size > 1 && endTime[index] == endTime[indices[i - 1]]) {
                --current_size;
            }
            dp[current_size] = {
                endTime[index],
                max(result, dp[binaryJob(dp, current_size, startTime[index])].second + profit[index])
            };
            result = max(result, dp[current_size++].second);
        }

        return result;
    }

    //2054. Two Best Non-Overlapping Events
    int maxTwoEvents(vector<vector<int>> events) {
        static vector<int>   indices(MAX_EVENTS);
        static pair<int, int> dp[MAX_EVENTS + 1] = { {1, 0} };
        const int SIZE = events.size();
        //init indices array
        for (int i = 0; i < SIZE; ++i)indices[i] = i;

        sort(indices.begin(), indices.begin() + SIZE,
            [&events](int l, int r) {return events[l][1] < events[r][1]; }
        );

        int result = 0;
        int runningMax = 0;
        int currentSize = 1;
        for (int i = 0; i < SIZE; ++i) {
            int index = indices[i];
            int prevMax = binaryJob(dp, currentSize, events[index][0]);
            runningMax = max(runningMax, dp[currentSize - 1].second);
            if (dp[currentSize - 1].first == events[index][1] + 1) {
                currentSize -= 1;
            }
            dp[currentSize++] = {
                events[index][1] + 1,
                max(events[index][2], runningMax)
            };
            result = max(result, events[index][2] + dp[prevMax].second);
        }

        return result;
    }

    //446. Arithmetic Slices II - Subsequence
    int numberOfArithmeticSlices(vector<int> nums) {
        const int SIZE = nums.size();

        //map :(
        //    key -> diff, 
        //    val -> vector of pairs where: (
        //        key -> last index,
        //        value -> general length
        //     )
        //)

        //how to 

        unordered_map<int, vector<pair<int, int>>> seqs;

        int result = 0;
        for (int i = 1; i < SIZE; ++i) {
            for (int j = i - 1; 0 <= j; --j) {
                
            }
        }

        return result;
        
    }

//834. Sum of Distances in Tree
#define MAX_N 30000
    bool used[MAX_N];

    //dp: first -> general sum, second -> count of nodes
    //routes -> matrix, where each row corresponds to node, first element is cummulative data
    pair<int, int> dfs(
        vector<vector<pair<int, int>>>& routes,
        const vector<vector<int>>& connections,
        int current
    ) {
        used[current] = true;

        int next_amount = 0;
        int next_sum = 0;
        for (int i = 0; i < connections[current].size(); ++i) {
            if (used[connections[current][i]])continue;
            const auto pair = dfs(routes, connections, connections[current][i]);
            next_sum += pair.first;
            next_amount += pair.second;
            routes[current].push_back(pair);
        }
        routes[current][0] = { next_sum, next_amount };

        used[current] = false;
        return { next_sum + next_amount + 1, next_amount + 1 };
    }

    //cummulative dfs
    void dfs2(
        const vector<vector<pair<int, int>>>& routes,
        const vector<vector<int>>& connections,
        vector<int>& answer,
        int current,
        pair<int, int> dp
    )
    {
        used[current] = true;

        answer[current] = routes[current][0].first + dp.first; // ?
        int j = 1;
        for (int i = 0; i < connections[current].size(); ++i) {
            if (used[connections[current][i]])continue;
            int next_sum = routes[current][0].first - routes[current][j].first + dp.first;
            int next_amount = routes[current][0].second - routes[current][j].second + dp.second;
            dfs2(
                routes,
                connections,
                answer,
                connections[current][i],
                { next_sum + next_amount + 1, next_amount + 1 }
            );
            j += 1;
        }

        used[current] = false;
    }

    vector<int> sumOfDistancesInTree(int n, vector<vector<int>> edges) {

        vector<vector<int>> connections(n);
        for (int i = 0; i < edges.size(); ++i) {
            connections[edges[i][1]].push_back(edges[i][0]);
            connections[edges[i][0]].push_back(edges[i][1]);
        }

        vector<vector<pair<int, int>>> routes(n, { {0,0} });
        dfs(routes, connections, 0);

        vector<int> answer(n, 0);
        dfs2(routes, connections, answer, 0, { 0,0 });

        return answer;
    }

    //768. Max Chunks To Make Sorted II
    int maxChunksToSorted(vector<int> arr) {
        map<int, int> numbers;

        const int N = arr.size();
        for (int i = 0; i < N; ++i) {
            ++numbers[arr[i]];
        }

        int prefix = 0;
        for (auto& pair : numbers) {
            int next = pair.second;
            pair.second = prefix;
            prefix += next;
        }

        show_map(numbers);

        int result = 0;

        for (int i = 0; i < N; ++i) {
            int max_index = -1;
            while (i < N) {
                max_index = max(max_index, numbers[arr[i]]++);
                if (max_index == i)break;
                i += 1;
            }
            ++result;
        }

        return result;
    }

    //2865. Beautiful Towers I || 2866. Beautiful Towers II
#define MAX_LENGTH 1000
    void helper(ll* dp, const int SIZE, const vector<int>& heights) {
        static vector<pair<ll, ll>> stack(MAX_LENGTH + 1);
        stack.clear();
        stack.push_back({ 0, -1 });
        stack.push_back({ heights[0], 0 });
        dp[0] = heights[0];

        long long stack_sum = dp[0];
        for (int i = 1; i < SIZE; ++i) {
            while (stack.back().first >= heights[i]) {
                auto top = stack.back(); stack.pop_back();
                stack_sum -= top.first * (top.second - stack.back().second);
            }
            stack_sum += heights[i] * (i - stack.back().second);
            stack.push_back({ heights[i], i });
            dp[i] = stack_sum;
            show_array(dp, i + 1);
        }
    }

    ll maximumSumOfHeights(vector<int> maxHeights) {
        static ll dp[2][MAX_LENGTH];
        const int SIZE = maxHeights.size();

        helper(dp[0], SIZE, maxHeights);
        reverse(maxHeights.begin(), maxHeights.end());
        helper(dp[1], SIZE, maxHeights);

        ll result = 0;
        for (int i = 0; i < SIZE; ++i) {
            result = max(result, dp[0][i] + dp[1][SIZE - i - 1] - maxHeights[SIZE - i - 1]);
        }

        return result;
    }
#undef MAX_LENGTH

    //880. Decoded String at Index
    string decodeAtIndex(string s, int k) {
        static vector<pair<string, uint64_t>> stack;
        stack.clear();

        for (int i = 0; i < s.length();) {
            while (i < s.length() && isdigit(s[i])) {
                stack.back().second *= s[i] - '0';
                ++i;
            }
            if (i < s.length()) {
                stack.push_back({ s.substr(i++,1),1 });
            }
            while (i < s.length() && !isdigit(s[i])) {
                stack.back().first += s[i++];
            }
        }

        uint64_t gen_len = 0;
        int i = 0;
        while (i < stack.size()) {
            gen_len = (gen_len + stack[i].first.length()) * stack[i].second;
            if (gen_len >= k)break;
            ++i;
        }

        while (0 <= i) {
            gen_len /= stack[i].second;
            k %= gen_len;
            gen_len -= stack[i].first.length();
            if (k == 0) {
                k = stack[i].first.length() - 1;
                break;
            }
            if (k > gen_len) {
                k = k - gen_len - 1;
                break;
            }
            i -= 1;
        }

        return stack[i].first.substr(k, 1);
    }

    //907. Sum of Subarray Minimums

#define MODULO 1000000007
#define MAX_SIZE 30001

    int sumSubarrayMins(vector<int> arr) {
        static vector<pair<int, int>> stack(MAX_SIZE);
        stack.clear();
        stack.emplace_back(make_pair( 0, -1 ));

        uint64_t result = 0;
        uint64_t stack_sum = 0;

        const int SIZE = arr.size();
        for (int i = 0; i < SIZE; ++i) {
            while (stack.size() > 0 && stack.back().first > arr[i]) {
                auto top = stack.back(); stack.pop_back();
                stack_sum -= top.first * (top.second - stack.back().second);
            }
            stack_sum += arr[i] * (i - stack.back().second);
            stack.emplace_back(make_pair(arr[i], i));
            result += stack_sum;
        }

        return result % MODULO;
    }

#undef MODULO
#undef MAX_SIZE
    //2092. Find All People With Secret
#define MAX_N 100000
    int find(const int* groups, int index) {
        while (index != groups[index])index = groups[index];
        return index;
    }

    vector<int> findAllPeople(int n, const vector<vector<int>>& meetings, int firstPerson) {
        static int groups[MAX_N];
        static int sortedIndices[MAX_N];
        const int SIZE = meetings.size();

        for (int i = 0; i < n; ++i) {
            sortedIndices[i] = groups[i] = i;
        }

        sort(sortedIndices, sortedIndices + SIZE, [&meetings](const auto& l, const auto& r) {
            return meetings[l][2] < meetings[r][2];
        });

        groups[firstPerson] = 0;
        static vector<int> temp;
        int i = 0;
        while (i < SIZE) {
            int currentTime = meetings[sortedIndices[i]][2];
            temp.clear();
            while (i < SIZE && meetings[sortedIndices[i]][2] == currentTime) {
                int g1 = find(groups, meetings[sortedIndices[i]][0]);
                int g2 = find(groups, meetings[sortedIndices[i]][1]);
                groups[max(g1, g2)] = min(g1, g2);
                temp.push_back(meetings[sortedIndices[i]][0]);
                temp.push_back(meetings[sortedIndices[i]][1]);
                ++i;
            }
            for (int j = 0; j < temp.size(); ++j) {
                if (find(groups, temp[j]) != 0) {
                    groups[temp[j]] = temp[j];
                }
            }
        }

        vector<int> result;
        for (int j = 0; j < n; ++j) {
            if (find(groups, j) == 0)result.push_back(j);
        }

        return result;
    }
#undef MAX_N

    //1395. Count Number of Teams

#define MAX_SIZE 1000
    int numTeams(vector<int>& rating) {
        static int sorted[MAX_SIZE];

        const int SIZE = rating.size();
        FenwickTree tree(SIZE);

        for (int i = 0; i < SIZE; ++i) sorted[i] = rating[i];
        sort(sorted, sorted + SIZE);

        int result = 0;
        for (int i = 0; i < SIZE; ++i) {
            int index = tree.query(lower_bound(sorted, sorted + SIZE, rating[i]) - sorted);
            tree.update(index + 1);
            cout << i << ":" << index << "\n";

        }

        return result;
    }
#undef MAX_SIZE

    //1239. Maximum Length of a Concatenated String with Unique Characters
#define MAX_LEN 16
#define MAX_VARS pow(2, 16)
    int maxLength(vector<string> arr) {
        static vector<pair<int, int>> asNums(MAX_LEN);
        asNums.clear();

        for (const auto& str : arr) {
            int num = 0;
            bool add = true;
            for (char c : str) {
                int offset = 1 << (c - 'a');
                if ((num & offset) == 0) {
                    num |= offset;
                }
                else {
                    add = false;
                    break;
                }
            }
            if (add)asNums.push_back({ num, str.length() });
        }

        static vector<pair<int, int>> seqs(MAX_VARS);
        seqs.clear();
        seqs.push_back({ 0,0 });

        int result = 0;
        for (const auto& pair : asNums) {
            for (int j = seqs.size() - 1; 0 <= j; --j) {
                if ((seqs[j].first & pair.first) == 0) {
                    seqs.push_back({ seqs[j].first | pair.first , seqs[j].second + pair.second });
                    result = max(result, seqs[j].second + pair.second);
                }
            }
        }

        return result;
    }
#undef MAX_LEN

    //368. Largest Divisible Subset
#define MAX_SIZE 1000
    vector<int> largestDivisibleSubset(vector<int> nums) {
        static int dp[2][MAX_SIZE];
        
        const int SIZE = nums.size();
        int result = 0;

        memset(dp[0], 0, SIZE * 4);
        memset(dp[1], 255, SIZE * 4);
        sort(nums.begin(), nums.end());
        
        for (int i = 0; i < SIZE; ++i) {
            for (int j = 0; j < i; ++j) {
                if (nums[i] % nums[j] != 0)continue;
                if (dp[0][j] < dp[0][i])continue;
                dp[0][i] = dp[0][j] + 1;
                dp[1][i] = j;
            }
            if (dp[0][result] < dp[0][i]) {
                result = i;
            }
        }

        vector<int> subset;
        subset.reserve(dp[0][result]);

        int index = result;
        while (index != -1) {
            subset.push_back(nums[index]);
            index = dp[1][index];
        }

        return subset;
    }
#undef MAX_SIZE

    //2157. Groups of Strings
#define MAX_SIZE 20000
    vector<int> groupStrings(vector<string> words) {
        static int groups[MAX_SIZE];

        unordered_map<int, int> used;
        const int SIZE = words.size();
        for (int i = 0; i < SIZE; ++i) {
            groups[i] = i;

            int num = 0;
            for (int j = 0; j < words[i].length(); ++j) {
                num |= 1 << (words[i][j] - 'a');
            }

            int bitMask = 1 << 25;
            while (bitMask > 0) {
                auto itr = used.find(bitMask ^ num);
                if (itr != used.end()) {
                    groups[itr->second] = i;
                }
                else {
                    used.insert({ bitMask ^ num,i });
                }
                bitMask /= 2;
            }
            used[num] = i;
        }

        used.clear();
        int msg = 0;
        for (int i = 0; i < SIZE; ++i) {
            msg = max(msg, ++used[find(groups, groups[i])]);
        }

        return { static_cast<int>(used.size()), msg };
    }

    //1354. Construct Target Array With Multiple Sums
    bool isPossible(vector<int> target) {

        priority_queue<ll> queue;
        const int SIZE = target.size();
        
        ll sum = 0;
        for (int i = 0; i < SIZE; ++i) {
            queue.push(static_cast<ll>(target[i]));
            sum += target[i];
        }

        //handle big numbers?
        while (queue.top() > 1) {
#if DEBUG == true
            show_priority_queue(queue);
#endif
            ll top = queue.top(); queue.pop();
            ll prev = top * 2 - sum;
            if (prev < 1)return false;
            queue.push(top % prev == 0 ? prev : top % prev);
            sum = top;
        }

        return queue.top() == 1;
    }

#undef MAX_SIZE

    //576. Out of Boundary Paths
#define MAX_ROWS 50
#define MAX_COLUMNS 50
#define MODULO 1000000007

    int findPaths(const int rows, const int columns, int maxMove, int startRow, int startColumn) {
        static const int directions[4][2] = { {0,1},{1,0},{-1,0},{0,-1} };
        static int f[MAX_ROWS][MAX_COLUMNS], s[MAX_ROWS][MAX_COLUMNS];

        //init 
        for (int i = 0; i < rows; ++i) {
            memset(f[i], 0, MAX_COLUMNS * 4);
        }
        f[startRow][startColumn] = 1;

        //calc
        int (*first)[MAX_COLUMNS] = f, (*second)[MAX_COLUMNS] = s;
        int result = 0;
        for (int i = 0; i < maxMove; ++i) {
            for (int j = 0; j < rows; ++j) {
                memset(second[j], 0, MAX_COLUMNS * 4);
            }

            for (int r = 0; r < rows; ++r) {
                for (int c = 0; c < columns; ++c) {
                    if (first[r][c] == 0)continue;
                    for (int d = 0; d < 4; ++d) {
                        int nr = r + directions[d][0];
                        int nc = c + directions[d][1];
                        if (nr < 0 || nr == rows || nc < 0 || nc == columns) {
                            result = (result + first[r][c]) % MODULO;
                        }
                        else {
                            second[nr][nc] = (second[nr][nc] + first[r][c]) % MODULO;
                        }
                    }
                }
            }
            swap(first, second);
        }

        return result;
    }
#undef MAX_ROWS
#undef MAX_COLUMNS
#undef MODULO
    //410. Split Array Largest Sum
#define MAX_LEN 1000
#define MAX_K 50

    int splitArray(vector<int> nums, int k) {
        //first -> general max, second -> current sum
        static int dp[MAX_LEN][MAX_K];
        dp[0][0] = nums[0];

        const int SIZE = nums.size();
        for (int i = 1; i < SIZE; ++i) {
            
            dp[i][0] = nums[i] + dp[i - 1][0];
            
            const int LIMIT = min(i, k - 1);
            for (int j = 1; j <= LIMIT; ++j) {
                dp[i][j] = INT_MAX;

                int sum = 0;
                for (int b = i; j <= b; --b) {
                    if (sum > dp[i][j])break;
                    sum += nums[b];
                    dp[i][j] = min(dp[i][j], max(sum, dp[b-1][j-1]));
                }
            }
        }

        return dp[SIZE - 1][k - 1];
    }
#undef MAX_LEN
#undef MAX_K

//1074. Number of Submatrices That Sum to Target
#define MAX_COLUMNS 100
    int numSubmatrixSumTarget(vector<vector<int>>& matrix, ll target) {
        static ll row[MAX_COLUMNS];
        
        unordered_map<ll, int> memo{ {0,1} };

        const register int ROWS = matrix.size();
        const register int COLUMNS = matrix[0].size();

        memset(row, 0, COLUMNS * 8);

        int result = 0;
        for (int r = 0; r < ROWS; ++r) {
            ll prefix = 0;
            for (int c = 0; c < COLUMNS; ++c) {
                prefix += matrix[r][c];
                result += memo[matrix[r][c] + row[c] - target];
            }
        }

        return result;
    }

    //475. Heaters
    int findRadius(vector<int> houses, vector<int> heaters) {
        //iterate for each heater
        //  find the middle house between i-th heater and (i+1)heater
        //  set radius as half of distance
        //  if there is no houses(between two heaters) then keep max radius in previous state

        sort(houses.begin(), houses.end());
        sort(heaters.begin(), heaters.end());

        int radius = max(0, heaters[0] - houses[0]);

        int from = 0;

        const int SIZE2 = houses.size();
        const int SIZE = heaters.size() - 1;
        for (int i = 0; i < SIZE; ++i) {

            while (from < SIZE2 && houses[from] < heaters[i])++from;
            int appMiddle = heaters[i] + (heaters[i + 1] - heaters[i]) / 2;
            int middle = houses[from];
            while (from < SIZE2 && houses[from] < heaters[i + 1]) {
                if (abs(appMiddle - houses[from]) < abs(appMiddle - middle)) {
                    middle = houses[from];
                }
                ++from;
            }

            radius = max(radius, min(middle - heaters[i], heaters[i + 1] - middle));

            if (from == SIZE2)break;
        }

        return max(radius, max(0, houses.back() - heaters.back()));
    }

//static const auto speedup = []() {
//    std::ios::sync_with_stdio(false);
//    std::cin.tie(nullptr);
//    std::cout.tie(nullptr);
//     return 0;
//}();
};


//2353. Design a Food Rating System
class PairComparer {
public:
    bool operator()(const pair<int, string>& l, const pair<int, string>& r) {
        if (l.first == r.first)return l.second > r.second;
        return l.first < r.first;
    }
};

class FoodRatings {
    unordered_map<string, priority_queue<pair<int, string>, vector<pair<int, string>>, PairComparer>> data;
    unordered_map<string, int> rating;
    unordered_map<string, string> food_type;
public:
    FoodRatings(vector<string>& foods, vector<string>& cuisines, vector<int>& ratings) {
        for (int i = 0; i < foods.size(); ++i) {
            data[cuisines[i]].push(make_pair(ratings[i], foods[i]));
            rating[foods[i]] = ratings[i];
            food_type[foods[i]] = cuisines[i];
        }
    }

    void changeRating(string food, int newRating) {
        rating[food] = newRating;
        data[food_type[food]].push(make_pair(newRating, food));
    }

    string highestRated(string cuisine) {
        auto& queue = data[cuisine];
        while (queue.size() != 0) {
            auto top = queue.top();
            
            if (rating[top.second] == top.first) {
                return top.second;
            }

            queue.pop();

        }
        return "";
    }
};
