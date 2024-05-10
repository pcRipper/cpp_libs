#pragma once
#include "../../../GeneralLibs/Headers/includes.hpp";
#include "DataStructures.hpp"

class Solution {
public:
    bool checkValidGrid(const vector<vector<int>> grid) {
        static vector<pair<int, pair<int, int>>> data(49);
        const int N = grid.size();
        const int LAST = N * N - 1;

        for (int r = 0, i = 0; r < N; ++r) {
            for (int c = 0; c < N; ++c) {
                data[i++] = { grid[r][c],{r,c} };
            }
        }

        sort(data.begin(), data.begin() + LAST + 1, [](const auto& l, const auto& r) {return l.first < r.first; });
        
        for (int i = 0; i < LAST; ++i) {
            int x = abs(data[i].second.first - data[i + 1].second.first);
            int y = abs(data[i].second.second - data[i + 1].second.second);
            if ((x == 2 && y == 1) || (x == 1 && y == 2))continue;
            return false;
        }

        return true;
    }

    //667. Beautiful Arrangement II
    vector<int> constructArray(int n, int k) {
        //split numbers from range[1;n] in half
        // k is factor of reversing sequence of bigger part of numbers
        // for example n = 6, k = 4
        // 1,2,3,4,5,6 -> 1,2,3 and 4,5,6
        // k/2 -> 2 and it will be 5,4,6, mean that only 2 first elements will be reversed
        
        return {};
    }

    //2514. Count Anagrams
    //unsolved!
    int countAnagrams(string s) {
        static char letters[26];
        unordered_map<string,pair<int,int>> data;

        //count as factorial
        //each repeated sequence of chars si also factorial,but devide it by main multiplier

        for (int i = 0; i < s.length(); ++i) {

            memset(&letters[0], 0, 26);

            int r = i;
            while (r < s.length() && s[r] != ' ') {
                letters[s[r++] - 'a'] = 1;
            }

            int unique = 0;
            for (int j = 0; j < 26; ++j) {
                unique += letters[j];
            }

            string cut = s.substr(i, r - i);
            sort(cut.begin(), cut.end());
            data.insert({ cut,{unique,0} });
            ++data[cut].second;
            i = r;
        }

        show_unordered_map(data, 0b011);

        return 0;
    }

    //1220. Count Vowels Permutation
    int countVowelPermutation(int n) {
        static uint32_t dp[20000][5]{ {1,1,1,1,1} };

        for (int i = 1; i < n; ++i) {
            dp[i][0] = dp[i - 1][1];
            dp[i][1] = (dp[i - 1][0] + dp[i - 1][2]) % 1000000007;
            dp[i][2] = (dp[i - 1][0] + dp[i - 1][1] + dp[i - 1][3] + dp[i - 1][4]) % 1000000007;
            dp[i][3] = (dp[i - 1][2] + dp[i - 1][4]) % 1000000007;
            dp[i][4] = dp[i - 1][0];
        }

        uint32_t sum = 0;
        for (int i = 0; i < 5; ++i) {
            sum = (sum + dp[n - 1][i]) % 1000000007;
        }

        return sum;
    }

    //779. K-th Symbol in Grammar
    int kthGrammar(int n, int k) {
        static const string mutate[2] = { "01","10" };
        if (n == 1)return 0;
        k -= 1;
        int bit_mask = 1 << (n - 2);
        string init = "01";
        while (bit_mask > 1) {
            init = mutate[init[bit_mask & k ? 1 : 0] - '0'];
            bit_mask = bit_mask >> 1;
        }

        return init[k & 1] - '0';
    }

    //1061. Lexicographically Smallest Equivalent String
    int find(char pos, const int letters[26]) {
        while (letters[pos] != pos) {
            pos = letters[pos];
        }
        return pos;
    }

     string smallestEquivalentString(string s1, string s2, string baseStr) {
        static int letters[26], min_letters[26];
        for (int i = 0; i < 26; ++i) {
            min_letters[i] = letters[i] = i;
        }

        const int LEN = s1.length();
        for (int i = 0; i < LEN; ++i) {
            letters[find(s1[i] - 'a', letters)] = find(s2[i] - 'a', letters);
        }

        for (int i = 0; i < 26; ++i) {
            int group = find(letters[i], letters);
            min_letters[i] = min(min_letters[group],min_letters[i]);
            min_letters[group] = min(min_letters[group], i);
        }

        for (int i = 0; i < baseStr.length(); ++i) {
            baseStr[i] = min_letters[baseStr[i] - 'a'] + 'a';
        }

        return baseStr;
    }

     //823. Binary Trees With Factors
     //not accurate
     int numFactoredBinaryTrees(vector<int> arr) {
        
         const int SIZE = arr.size(); 
         unordered_map<int, int> nums;
         for (int i = 0; i < SIZE; ++i) {
             nums[arr[i]] = i;
         }

         vector<vector<int>> dp(1,vector<int>(SIZE,1));
         int sum = 0,prev_sum = 0;
         int current_level = 0;
         do {
             prev_sum = sum;
             dp.push_back(vector<int>(SIZE, 0));
             for (int j = 0; j < SIZE; ++j) {
                 for (int level = 0; level <= current_level; ++level) {
                     for (int i = level == 0 && current_level != 0 ? 0 : j; i < SIZE; ++i) {
                         int next_node = arr[j] * arr[i];
                         if (nums.find(next_node) == nums.end())continue;
                         dp[current_level + 1][nums[next_node]] += dp[current_level][j] * dp[level][i] * (arr[j] == arr[i] ? 1 : 2);
                     }
                 }
                 sum += dp[current_level][j];
             }
             current_level += 1;
         } while (prev_sum != sum);

         return sum;
     }

     //76. Minimum Window Substring
     bool isSmaller(const string& s, pair<int, int> l, pair<int, int> r) {
         if (l.second - l.first < r.second - r.first)return true;
         if (r.second - r.first < l.second - l.first)return false;
         while (l.first != l.second) {
             if (s[l.first++] > s[r.first++])return false;
         }
         return true;
     }

     string minWindow(string s, string t) {
         vector<int> letters(58, 0);
         const int LEN = s.length();

         if (t.length() > LEN)return "";

         int r = 0;
         while (r < t.length()) {
             --letters[t[r] - 'A'];
             ++letters[s[r++] - 'A'];
         }
         int diffs = 0;
         for (int i = 0; i < 58; ++i) {
             diffs += letters[i] < 0;
         }

         //first -> start index, second -> length
         pair<int, int> result;
         int l = 0;
         
         while (l < LEN) {
             while (r < LEN && diffs != 0) {
                 if (letters[s[r] - 'A']++ < 0) {
                     --diffs;
                 }
                 ++r;
             }
             if (--letters[s[l]-'A'] < 0) {
                 if (result.second == 0 && diffs == 0)result = { l,r - l };
                 else if (result.second > r - l && diffs == 0)result = { l,r - l };
                 ++diffs;
             }
             ++l;
         }

         return s.substr(result.first,result.second);
     }

     //502. IPO
     int findMaximizedCapital(int k, int w, vector<int>& profits, vector<int>& capital) {
         return 0;
     }

     //354. Russian Doll Envelopes
     int maxEnvelopes(vector<vector<int>>& envelopes) {
         sort(envelopes.begin(), envelopes.end(), [](const auto& l, const auto& r) {
             if (l[0] == r[0])return l[1] < r[1];
             return l[0] < r[0];
         });

         auto prev = envelopes[0];
         int steps = 1;
         for (int i = 1; i < envelopes.size(); ++i) {
             if (envelopes[i][0] > prev[0] && envelopes[i][1] > prev[1]) {
                 ++steps;
                 prev = envelopes[i];
             }
             else if (envelopes[i][1] < prev[1]) {
                 prev = envelopes[i];
             }
         }
         return steps;
     }

     //51. N-Queens
     //52. N-Queens II(allmost the same)
     void dfs(vector<vector<string>>& answer, vector<string>& board, int row, const int N) {
         static vector<int> queen_pos;

         if (row == N) {
             for (int y = 0; y < queen_pos.size(); ++y) {
                 board[y][queen_pos[y]] = 'Q';
             }
             answer.push_back(board);
             for (int y = 0; y < queen_pos.size(); ++y) {
                 board[y][queen_pos[y]] = '.';
             }
             return;
         }

         for (int i = 0; i < N; ++i) {
             bool can_place = true;
             for (int y = 0; y < queen_pos.size(); ++y) {
                 if (i == queen_pos[y] || abs(y - row) == abs(queen_pos[y] - i)) {
                     can_place = false;
                     break;
                 }
             }
             if (!can_place)continue;
             queen_pos.push_back(i);
             dfs(answer, board, row + 1, N);
             queen_pos.pop_back();
         }
     }

     vector<vector<string>> solveNQueens(int n) {
         vector<vector<string>> answer;
         vector<string> board(n, string(n, '.'));
         dfs(answer, board, 0, n);
         return answer;
     }

     //679. 24 Game
     bool judgePoint24(vector<int>& cards) {
         return dfs({ double(cards[0]),double(cards[1]),double(cards[2]),double(cards[3]) });
     }
     bool dfs(vector<double> cards) {
         static const vector<function<double(double, double)>> funcs{
             [](double l,double r) {return l + r; },
             [](double l,double r) {return l - r; },
             [](double l,double r) {return l / r; },
             [](double l,double r) {return l * r; }
         };
         if (cards.size() == 1)return round(cards[0] * 10000) / 10000 == 24.0;

         for (int f = 0; f < 4; ++f) {
             for (int l = 0; l < cards.size(); ++l) {
                 for (int r = 0; r < cards.size(); ++r) {
                     if (l == r)continue;
                     if (f == 2 && cards[r] == 0)continue;
                     vector<double> next{ funcs[f](cards[l],cards[r]) };
                     for (int i = 0; i < cards.size(); ++i) {
                         if (i != l && i != r)next.push_back(cards[i]);
                     }
                     if (dfs(next))return true;
                 }
             }
         }

         return false;
     }

     //1420. Build Array Where You Can Find The Maximum Exactly K Comparisons
     int numOfArrays(int n, int m, int k) {
         static const int modulo = 1000000007;
         static uint64_t dp[50][50][100];

         //if (k == 0 || m < k)return 0;

         for (int v = 0; v < m; ++v) {
             dp[0][0][v] = 1;
         }

         for (int l = 1; l < n; ++l) {
             for (int s = 0; s < k; ++s) {
                 for (int v = 0; v < m; ++v) {
                     dp[l][s][v] = (dp[l-1][s][v] * (v+1)) % modulo;
                     if (s == 0)continue;
                     for (int i = 0; i < v; ++i) {
                         dp[l][s][v] = (dp[l][s][v] + dp[l-1][s-1][i]) % modulo;
                     }
                 }
             }
         }

         int sum = 0;
         for (int i = 0; i < m; ++i) {
             sum = (sum + dp[n-1][k - 1][i]) % modulo;
         }

         return sum;
     }

     //unsolved
     long long minIncrementOperations(vector<int> nums, int k) {

     }

     //738. Monotone Increasing Digits
     int monotoneIncreasingDigits(int n) {

         uint32_t i = 10, j = 1;
         while (i <= n) {
             if ((n / i) % 10 > (n / j) % 10) {
                 n -= (n % i) + 1;
             }
             j = i;
             i *= 10;
         }

         return n;
     }

     //1395. Count Number of Teams
     //unsolved
     int numTeams(vector<int> rating) {
        //first -> falling sequence
        //second -> increasing sequnce
        static pair<int,int> dp[1000]{ {0,0} };
          
        int sum = 0;
        for (int i = 1; i < rating.size(); ++i) {
            dp[i] = { 0,0 };
            for (int j = 0; j < i; ++j) {
                if (rating[i] < rating[j]) {
                    dp[i].first = true;
                    sum += dp[j].first;
                }
                else {
                    dp[i].second = true;
                    sum += dp[j].second;
                }
            }
        }
             
        return sum;
     }

     //2915. Length of the Longest Subsequence That Sums to Target
     int lengthOfLongestSubsequence(vector<int> nums, int target) {
         static int dp[1001];
         for (int i = 0; i <= target; ++i) {
             dp[i] = 0;
         }

         for (int i = 0; i < nums.size(); ++i) {
             if (nums[i] > target)continue;
             for (int j = target - nums[i]; j > 0; --j) {
                 if (dp[j] == 0)continue;
                 dp[nums[i] + j] = max(dp[nums[i] + j], dp[j] + 1);
             }
             dp[nums[i]] = max(dp[nums[i]], 1);
         }

         return dp[target] == 0 ? -1 : dp[target];
     }

     //2009. Minimum Number of Operations to Make Array Continuous
     int removeDuplicates(vector<int>& data) {
         const int SIZE = data.size();
         int l = 1;
         for (int r = 1; r < SIZE; ++r) {
             if (data[r - 1] == data[r])continue;
             data[l++] = data[r];
         }
         return l;
     }

     int minOperations(vector<int> nums) {
         sort(nums.begin(), nums.end());
         const int OLD_SIZE = nums.size();
         const int SIZE = removeDuplicates(nums);
         const auto END = nums.begin() + SIZE;

         int answer = 100000;
         for (int i = 0; i < SIZE; ++i) {
             int if_min = lower_bound(nums.begin() + i, END, nums[i] + OLD_SIZE - 1) - nums.begin();
             if_min = min(SIZE - 1, if_min);
             if (nums[if_min] > nums[i] + OLD_SIZE - 1)--if_min;
             answer = min(answer, SIZE - 1 - if_min + i);
         }

         return answer + OLD_SIZE - SIZE;
     }

     //2433. Find The Original Array of Prefix Xor
     vector<int> findArray(vector<int> pref) {
         if (pref.size() < 2)return pref;

         const int SIZE = pref.size() - 1;
         int l = pref[0], r = pref[1];
         for (int i = 1; i < SIZE; ++i) {
             int next = pref[i + 1];
             pref[i] = l ^ r;
             l = r;
             r = next;
         }
         pref.back() = l ^ r;
         return pref;
     }

     //2401. Longest Nice Subarray
     int longestNiceSubarray(vector<int> nums) {
         const int SIZE = nums.size();

         int len = 0;
         int l = 0, r = 0;
         int current = 0;
         while (r < SIZE) {
             while (r < SIZE && (current & nums[r]) == 0) {
                 current |= nums[r++];
             }
             len = max(len, r - l);
             current ^= nums[l++];
         }

         return len;
     }

     //670. Maximum Swap
     int maximumSwap(int num) {
         static int digits[9];
         if (num < 12)return num;

         int size = floor(log10(num)) + 1;
         for (int i = size - 1; 0 <= i; --i) {
             digits[i] = num % 10;
             num /= 10;
         }

         for (int i = 0; i < size; ++i) {
             if (digits[i] == 9)continue;
             int index = i;
             for (int j = i + 1; j < size; ++j) {
                 if (digits[index] <= digits[j]) {
                     index = j;
                 }
             }
             if (digits[i] == digits[index])continue;
             swap(digits[i], digits[index]);
             break;
         }

         for (int i = 0; i < size; ++i) {
             num = num * 10 + digits[i];
         }
         return num;
     }

     //1610. Maximum Number of Visible Points
     int visiblePoints(vector<vector<int>> points, int angle, vector<int> location) {
         static vector<double> angles(100000);
         static const double PI = 3.1415926535897932384626433832795;
         static const double DPI = 6.283185307179586476925286766559;

         int additional = 0;
         for (int i = 0; i < points.size(); ++i) {
             if (points[i] == location) {
                 ++additional;
                 continue;
             }
             angles[i - additional] = atan2(location[1] - points[i][1], location[0] - points[i][0]) + PI;
         }
         const int SIZE = points.size() - additional;

         sort(angles.begin(), angles.begin() + SIZE);

         double radianAngle = angle * PI / 180;
         int l = 0, r = 0;
         int result = 0;
         while (r < SIZE) {
             while (r < SIZE && abs(angles[r] - angles[l]) <= radianAngle) {
                 ++r;
             }
             result = max(result, r - l++);
         }

         r = 0;
         l = min(max(0, l - 1), SIZE - 1);
         while (l < SIZE && r < l) {
             while (r < l && abs(angles[r] + (DPI - angles[l])) <= radianAngle) {
                 ++r;
             }
             result = max(result, SIZE - l++ + r);
         }

         return result + additional;
     }

     //1944. Number of Visible People in a Queue
     vector<int> canSeePersonsCount(vector<int> heights) {
         static int stack[100000];
         static int size;

         size = 0;

         for (int i = heights.size() - 1; 0 <= i; --i) {
             int counter = 0;
             while (size != 0 && stack[size-1] < heights[i]) {
                 ++counter;
                 --size;
             }
             stack[size++] = heights[i];
             heights[i] = counter + int(size > 1);
         }

         return heights;
     }

     //962. Maximum Width Ramp
     int maxWidthRamp(vector<int>& nums) {
         static int stack[50000];
         static int size;

         const int SIZE = nums.size();
         size = 0;

         int result = 0;
         for (int i = 0; i < SIZE; ++i) {
             if (size == 0 || nums[stack[size - 1]] > nums[i]) {
                 stack[size++] = i;
                 continue;
             }
             int j = size;
             while (0 < --j) {
                 if (nums[stack[j - 1]] > nums[i]) {
                     break;
                 }
             }
             result = max(result, i - stack[j]);
         }

         return result;
     }

     //316. Remove Duplicate Letters
    //1081. Smallest Subsequence of Distinct Characters
     string removeDuplicateLetters(string s) {
         static int  last_seen[26];
         static char increasing[26];
         static int size;

         for (int i = 0; i < 26; ++i)last_seen[i] = -1;
         size = 0;

         const int LEN = s.length();
         for (int i = 0; i < LEN; ++i) {
             while (0 < size && increasing[size - 1] > s[i]) {
                 last_seen[increasing[--size] - 'a'] = i;
             }
         }

         return "";
     }

     //1743. Restore the Array From Adjacent Pairs
     vector<int> restoreArray(vector<vector<int>> adjacentPairs) {
         unordered_map<int, vector<int>> data;
         for (const auto& pair : adjacentPairs) {
             data[pair[0]].push_back(pair[1]);
             data[pair[1]].push_back(pair[0]);
         }

         auto itr = data.begin();
         while (itr != data.end()) {
             if (itr->second.size() == 1)break;
             ++itr;
         }

         vector<int> result;
         result.reserve(data.size());

         result.push_back(itr->first);
         int current = data[itr->first][0];
         data.erase(itr->first);
         while (data[current].size() != 1) {
             result.push_back(current);
             int next = data.find(data[current][0]) == data.end() ? data[current][1] : data[current][0];
             data.erase(current);
             current = next;
         }
         result.push_back(current);

         return result;
     }

     //1631. Path With Minimum Effort
     int minimumEffortPath(vector<vector<int>> heights) {
         static const vector<pair<int, int>> directions = { {0,1},{1,0},{-1,0},{0,-1} };
         static priority_queue<pair<int, pair<int, int>>, vector<pair<int, pair<int, int>>>, greater<pair<int, pair<int, int>>>> queue;
         static int diffs[100][100];

         const int ROWS = heights.size();
         const int COLUMNS = heights[0].size();
         if (ROWS == 1 && COLUMNS == 1)return 0;


         for (int r = 0; r < ROWS; ++r) {
             for (int c = 0; c < COLUMNS; ++c) {
                 diffs[r][c] = INT_MAX;
             }
         }
            
         queue.push({ 0,{0,0} });

         while (not queue.empty()) {
             auto [val,point] = queue.top(); queue.pop();
             if (diffs[point.first][point.second] <= val)continue;

             diffs[point.first][point.second] = val;
             for (const auto& [y, x] : directions) {
                 uint16_t ny = y + point.first;
                 uint16_t nx = x + point.second;
                 if (ny < ROWS && nx < COLUMNS){
                     queue.push({ max(abs(heights[point.first][point.second] - heights[ny][nx]),val),{ny,nx} });
                 }
             }
         }

         return diffs[ROWS-1][COLUMNS-1];
     }

     //815. Bus Routes
     int numBusesToDestination(vector<vector<int>> routes, int source, int target) {
         static bool used[500];
         if (source == target)return 0;
         unordered_map<int, vector<int>> stops;

         int bus_num = 0;
         for (const auto& route : routes) {
             for (int i = 0; i < route.size(); ++i) {
                 stops[route[i]].push_back(bus_num);
             }
             ++bus_num;
         }

         memset(&used[0], 0, routes.size());

         deque<int> queue;
         for (const int bus : stops[source]) {
             queue.push_back(bus);
             used[bus] = true;
         }

         int steps = 1;
         while (not queue.empty()) {
             int size = queue.size();
             while (size-- > 0) {
                 int top = queue.back(); queue.pop_back();

                 for (int i = 0; i < routes[top].size(); ++i) {
                     if (routes[top][i] == target)return steps;
                     for (int bus : stops[routes[top][i]]) {
                         if (used[bus]) continue;
                         queue.push_front(bus);
                         used[bus] = true;
                     }
                 }
             }
             ++steps;
         }

         return -1;
     }

     //
     vector<int> findMinHeightTrees(int n, vector<vector<int>> edges) {
         static bool visited[20001] = { 1 };
         vector<vector<int>> connections(n);

         memset(&visited[1], 0, n);
         for (const auto& edge : edges) {
             connections[edge[0]].push_back(edge[1]);
             connections[edge[1]].push_back(edge[0]);
         }

         vector<vector<int>> levels;
         levels.reserve(n);
         
         vector<int> level{ 0 };
         int count = 1;
         int l = 0, r = 1;
         while (count < n) {
             vector<int> next_level;
             for (int node : level) {
                 for (int next_node : connections[node]) {

                 }
             }
         }

         show_vector(levels);

         return {};
     }

     //2909. Minimum Sum of Mountain Triplets II
     int minimumSum(vector<int> nums) {
         static int sorted[100000];
         const int SIZE = nums.size() - 1;
         int min_right = INT_MAX;
         for (int i = SIZE; 1 < i; --i) {
             min_right = min(min_right, nums[i]);
             sorted[i] = min_right;
         }

         int result = INT_MAX;
         int left_min = nums[0];
         for (int i = 1; i < SIZE; ++i) {
             if (left_min < nums[i] && sorted[i+1] < nums[i]) {
                 result = min(result, nums[i] + left_min + sorted[i + 1]);
             }
             left_min = min(left_min, nums[i]);
         }

         return result == INT_MAX ? -1 : result;
     }

     //410. Split Array Largest Sum
     //recursion?
     int splitArray(vector<int>& nums, int k) {
         const int SIZE = nums.size() - 1;

         uint64_t sum = 0;
         for (int i = 0; i <= SIZE; ++i) {
             sum += nums[i];
         }

         uint64_t max_sum = 0;
         uint64_t current_sum = 0;
         uint64_t precise_part = sum / k;
         for (int i = 0; i < SIZE; ++i) {
             if (current_sum + nums[i] > precise_part && current_sum != 0) {
                 max_sum = max(current_sum, max_sum);
                 current_sum = 0;
             }
             current_sum += nums[i];
         }

         if (current_sum == precise_part)return max((uint64_t)nums[SIZE], max(precise_part, max_sum));
         return max(max_sum, current_sum + nums[SIZE]);
     }

     int countPalindromicSubsequence(string s) {
         unordered_map<char, int> left;
         unordered_map<char, int> right;
         unordered_map<string, bool> memo;

         const int SIZE = s.length() - 1;
         left[s[0]] = 1;
         for (int i = 2; i <= SIZE; ++i) {
             ++right[s[i]];
         }

         for (int i = 1; i < SIZE; ++i) {
             for (const auto&x : left) {
                 if (right.find(x.first) == right.end())continue;
                 memo[string(1, x.first) + s[i] + x.first] = true;
             }

             ++left[s[i]];
             if (--right[s[i + 1]] == 0)right.erase(s[i + 1]);
         }

         return memo.size();
     }

     //1980. Find Unique Binary String
     int from_binary(const string& num) {
         int result = 0;
         for (int i = 0; i < num.length(); ++i) {
             result = result * 2 + num[i] - '0';
         }
         return result;
     }

     string to_binary(int num) {
         string result;

         while (num != 0) {
             result += (num & 1) + '0';
             num /= 2;
         }

         reverse(result.begin(), result.end());
         return result;
     }

     string findDifferentBinaryString(vector<string>& nums) {

         sort(nums.begin(), nums.end());

         int num = 0;
         for (int i = 0; i < nums.size(); ++i) {
             int current = from_binary(nums[i]);
             if (num == current) {
                 ++num;
                 continue;
             }
             string result = to_binary(num);
             return string(nums[0].length() - result.length(), '0') + result;
         }

         return "";
     }

     //2391. Minimum Amount of Time to Collect Garbage
     int garbageCollection(vector<string>& garbage, vector<int>& travel) {
         static int prefix_sum[100000] = { 0 };
         int GlassLastIndex = 0;
         int PaperLastIndex = 0;
         int MetalLastIndex = 0;
         int sum = 0;

         const int SIZE = garbage.size();
         for (int i = 0; true; ++i) {
             for (char gt : garbage[i]) {
                 switch (gt) {
                 case('G'):
                     GlassLastIndex = i;
                     break;
                 case('M'):
                     MetalLastIndex = i;
                     break;
                 case('P'):
                     PaperLastIndex = i;
                     break;
                 default:
                     break;
                 }
             }
             sum += garbage[i].length();
             if (i + 1 == SIZE)break;
             prefix_sum[i + 1] = prefix_sum[i] + travel[i];
         }

         return sum + prefix_sum[GlassLastIndex] + prefix_sum[PaperLastIndex] + prefix_sum[MetalLastIndex];
     }

     //233. Number of Digit One
     int countDigitOne(int n) {
         static const int values[] = { 0,1,20,300,4000,50000,600000,7000000,80000000,900000000 };
         int sum = 0;
         int i = 0;
         int right_side = 0;
         long power = 1;
         while (n > 0) {
             sum += (n % 10) * values[i];
             if (n % 10 == 1)sum += right_side + 1;
             else if (n % 10 > 1)sum += power;
             right_side = right_side + (n % 10) * power;
             n /= 10;
             power *= 10;
             ++i;
         }
         return sum;
     }
    
     #define MAX_N_VALUE 25
     //688. Knight Probability in Chessboard
     double knightProbability(int n, int k, int row, int column) {
         static const pair<int, int> directions[8] = {{2,1},{1,2},{-1,2},{-2,1},{-2,-1},{2,-1},{1,-2},{-1,-2}};
         static double   f[25][25], s[25][25];
         
         for (int i = 0; i < n; ++i)memset(f[i], 0, 200);
         f[row][column] = 1;

         double (*first)[25] = f, (*second)[25] = s;
         while (k-- > 0) {
             for (int i = 0; i < n; ++i)memset(second[i], 0, 200);
             
             for (int r = 0; r < n; ++r) {
                 for (int c = 0; c < n; ++c) {
                     if (first[r][c] == 0)continue;
                     for (int i = 0; i < 8; ++i) {
                         int nx = c + directions[i].first;
                         int ny = r + directions[i].second;
                         if (nx < 0 || ny < 0 || nx >= n || ny >= n)continue;
                         second[ny][nx] += first[r][c] / 8;
                     }
                 }
             }
             swap(first, second);
         }
         
         double sum = 0;
         for (int r = 0; r < n; ++r) {
             for (int c = 0; c < n; ++c) {
                 sum += first[r][c];
             }
         }

         return sum;
     }

     //416. Partition Equal Subset Sum
     bool canPartition(vector<int>& nums) {
         static bool dp[10001][200] = { {1} };
         const int SUM = reduce(nums.begin(), nums.end());
         const int HALF = SUM / 2;
         const int SIZE = nums.size() - 1;
         if (SUM & 1)return false;

         for (int i = 1; i <= HALF; ++i) {
             memset(&dp[i][0], 0, 200);
         }

         int max_sum = 0;
         for (int num = 0; num <= SIZE; ++num) {
             for (int i = min(HALF - nums[num], max_sum); 0 <= i; --i) {
                 for (int l = min(SIZE - 1, num); 0 <= l; --l) {
                     if (!dp[i][l])continue;
                     if (i + nums[num] == HALF)return true;
                     dp[i + nums[num]][l + 1] = true;
                 }
             }
             max_sum += nums[num];
         }

         return false;
     }

     //85. Maximal Rectangle
     int maximalRectangle(vector<vector<char>>& matrix) {
         const int ROWS = matrix.size();
         const int COLUMNS = matrix[0].size();
         int result = 0;

         for (int r = 0; r < ROWS; ++r) {
             if (r + 1 != ROWS) {
                 for (int c = 0; c < COLUMNS; ++c) {
                     if (matrix[r][c] == '0')continue;
                     matrix[r + 1][c] += matrix[r][c] - 48;
                 }
             }

             for (int c = 0; c < COLUMNS; ++c) {
                 int l = c + 1;
                 int current_min = INT_MAX;
                 for (int i = c; 0 <= i; --i) {
                     if (matrix[r][i] == '0') {
                         l = i;
                         continue;
                     }
                     current_min = min(current_min, matrix[r][i] - 48);
                     result = max(result, current_min * (l - i));
                 }
             }
         }

         return result;
     }

     //1277. Count Square Submatrices with All Ones
     int countSquares(vector<vector<int>>& matrix) {
         const int ROWS = matrix.size();
         const int COLUMNS = matrix[0].size();
         int result = 0;
         for (int r = 0; r < ROWS; ++r) {
             int l = -1;
             for (int c = 0; c < COLUMNS; ++c) {
                 if (r + 1 != ROWS && matrix[r + 1][c] != 0)matrix[r + 1][c] += matrix[r][c];
                 if (matrix[r][c] == 0) {
                     l = c;
                     continue;
                 }
                 ++result;
                 if (matrix[r][c] == c - l) {
                     result += pow(max(matrix[r][c] - 1, 0), 2);
                 }
             }
         }
         return result;
     }

     //2147. Number of Ways to Divide a Long Corridor
     int numberOfWays(string corridor) {
         const int LEN = corridor.length();
         int general_amount = 0;
         uint64_t prev = 1, current = 0;
         int i = 0;
         while (i < LEN) {
             int local = 0;
             while (i < LEN && local < 2) {
                 if (corridor[i] == 'S')++local;
                 ++i;
             }
             general_amount += local;
             int l = i - 1;
             while (i < LEN&& corridor[i] != 'S') {
                 ++i;
             }
             current = (prev * (i - l)) % 1000000007;
             if (i == LEN)break;
             prev = current;
         }

         return general_amount > 0 && general_amount % 2 == 0 ? prev : 0;
     }

     //221. Maximal Square
     int maximalSquare(vector<vector<char>> matrix) {
         static int dp[300][300];
         const int ROWS = matrix.size();
         const int COLUMNS = matrix[0].size();
         int result = 0;

         for (int r = 0; r < ROWS; ++r) {
             for (int c = 0; c < COLUMNS; ++c) {
                 if (matrix[r][c] == '0') {
                     dp[r][c] = 0;
                 }
                 else {
                     dp[r][c] = matrix[r][c] - 48 + (r == 0 ? 0 : dp[r - 1][c]);
                 }
             }

             int local_min = 0;
             int l = -1;
             for (int c = 0; c < COLUMNS; ++c) {
                 if (local_min < dp[r][c]) {
                     local_min = dp[r][c];
                     l = c - 1;
                 }
                 else local_min = min(local_min, dp[r][c]);
                 if (c - l >= local_min)result = max(result, local_min);
             }
         }

         return result * result;
     }

     //2948. Make Lexicographically Smallest Array by Swapping Elements
     vector<int> lexicographicallySmallestArray(vector<int> nums, const int limit) {
         static vector<pair<int, int>> sorted(100000);
         static vector<int> indices(100000);
         const int SIZE = nums.size();

         for (int i = 0; i < SIZE; ++i) {
             sorted[i] = { nums[i],i };
         }

         sort(sorted.begin(), sorted.begin() + SIZE);

         int i = 0;
         while (i < SIZE) {
             int s = 1;
             int l = i;
             indices[0] = sorted[i++].second;
             while (i < SIZE) {
                 if (sorted[i].first - sorted[i - 1].first > limit)break;
                 indices[s++] = sorted[i++].second;
             }
             sort(indices.begin(), indices.begin() + s);
             for (int j = 0; j < s; ++j) {
                 nums[indices[j]] = sorted[l++].first;
             }
         }

         return nums;
     }

     //480. Sliding Window Median
     double insert_and_balance(
         priority_queue<pair<int, int>, vector<pair<int, int>>, less<pair<int, int>>>& left,
         priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>>& right,
         const int min_index,
         const pair<int, int> value
     ) {
         while (left.size() > 0 && left.top().second < min_index) {
             left.pop();
         }
         while (right.size() > 0 && right.top().second < min_index) {
             right.pop();
         }
         
         left.push(value);
         while (left.size() > right.size() + 1) {
             const auto top = left.top(); left.pop();
             if (top.second < min_index)continue;
             right.push(top);
         }
         while (right.size() > left.size() + 1) {
             const auto top = right.top(); right.pop();
             if (top.second < min_index)continue;
             left.push(top);
         }

         if (left.size() == right.size())return (left.top().first + right.top().first) / 2;
         return left.size() > right.size() ? left.top().first : right.top().first;
     }

     vector<double> medianSlidingWindow(vector<int> nums, int k) {
         priority_queue<pair<int, int>, vector<pair<int, int>>, less<pair<int, int>>> left;
         priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> right;
         k -= 1;
         
         int i = 0;
         while (i < k) {
             insert_and_balance(left, right, 0, {nums[i],i});
             i += 1;
         }
         
         vector<double> result;
         result.reserve(nums.size() - k);
         
         const int SIZE = nums.size();
         while (i < SIZE) {
             result.push_back(insert_and_balance(left, right, i - k, { nums[i], i }));
             i += 1;
         }
         
         return result;
     }

     //1611. Minimum One Bit Operations to Make Integers Zero
     string as_bit(int n) {
         string result;

         while (n > 0) {
             result += n & 1;
             n /= 2;
         }

         return result;
     }

     int minimumOneBitOperations(int n) {
         //TODO : calculate amount of operations to reduce the rightmost bit to and add to this 2^(amount of bits)
         //if bit == 1 than sum = bit(as num) - sum
         return 0;
     }

     //474. Ones and Zeroes
#define MAX_LEN_STRS_474 601

     struct PairHash {
         auto operator()(const pair<int, int>& data) const {
             return hash<int>{}(data.first) ^ hash<int>{}(data.second);
         }
     };

     int findMaxForm(const vector<string> strs, int m, int n) {
         //first -> max length, second -> (first -> zeroes, second -> ones)
         static unordered_map<pair<int, int>, bool, PairHash> dp[MAX_LEN_STRS_474] = { {{{0,0},1}} };
         
         const int SIZE = strs.size();
         for (int i = 0; i < SIZE; ++i) {
             dp[i+1].clear();

             //transform
             int zeroes = 0;
             int ones = 0;
             for (int j = 0; j < strs[i].length(); ++j) {
                 if (strs[i][j] == '0')zeroes += 1;
                 else ones += 1;
             }

             //calculate using other sequences
             for (int j = i; 0 <= j; --j) {
                 for (const auto&[v, _] : dp[j]) {
                     if (v.first + zeroes <= m && v.second + ones <= n) {
                         dp[j + 1][make_pair(v.first + zeroes, v.second + ones)] = true;
                     }
                 }
             }
         }

         for (int i = SIZE; 0 < i; --i) {
             if (dp[i].size() != 0)return i;
         }

         return 0;
     }

     //135. Candy
     size_t sum(size_t n) {
         return (n * n + n) / 2;
     }
     int candy(const vector<int>& ratings) {

         int result = 0;
         int prev = 0;
         const int SIZE = ratings.size() - 1;
         for (int i = 0; i < SIZE; ++i) {
             int next = 1;
             int r = i + 1;
             if (ratings[r] == ratings[i]) {
                 result -= i == 0 ? 1 : 2;
             }
             else if (ratings[r] > ratings[i]) {
                 while (r <= SIZE && ratings[r] > ratings[r - 1])++r;
                 next = r - i;
                 r -= 1;
                 result -= min(prev, next);
             }
             else {
                 while (r <= SIZE && ratings[r] < ratings[r - 1])++r;
                 result -= min(prev, r - i);
                 r -= 1;
             }
             result += sum(r - i + 1);
             prev = next;
             i = r - 1;
         }

         return max(1, result);
     }

     //1547. Minimum Cost to Cut a Stick
     int calculate(const int LEN, const vector<int>& cuts, int from, int to) {
         static vector<int> ranged;
         int sum = 0;

         while (from < to) {
             ++from;
         }

         return sum;
     }

     int minCost(int n, vector<int>& cuts) {
         static vector<int> sequence(100);
         static vector<int> temp(100);

         const int SIZE = cuts.size();
         for (int t = 0; t < SIZE; ++t) {
             for (int j = 0; j <= t; ++j) {
                 //calculate for each inserting position
             }
         }

         return calculate(n, sequence,0, SIZE - 1);
     }

     //2468. Split Message Based on Limit
     vector<string> splitMessage(string message, int limit) {
         if (limit < 6)return {};
         //min place for suffix
         limit -= 3;

         //binary search on : parts = (message.length() / limit) - 1
         // calc if to_string(parts).length() will be smaller than limit


     }

     //95. Unique Binary Search Trees II
     static vector<tree_node::TreeNode*> memo[16][16];

     void dfs(int from, int to) {
         static vector<tree_node::TreeNode*> empty = { nullptr };

         if (memo[from][to].size() != 0)return;

         for (int i = from; i <= to; ++i) {
             vector<tree_node::TreeNode*>* left = &empty, * right = &empty;
             if (from <= i - 1) {
                 dfs(from, i - 1);
                 left = &memo[from][i - 1];
             }
             if (i + 1 <= to) {
                 dfs(i + 1, to);
                 right = &memo[i + 1][to];
             }

             for (int l = 0; l < left->size(); ++l) {
                 for (int r = 0; r < right->size(); ++r) {
                     memo[from][to].push_back(new tree_node::TreeNode(i, (*left)[l], (*right)[r]));
                 }
             }
         }
     }

     vector<tree_node::TreeNode*> generateTrees(int n) {
         dfs(1, n);
         return memo[1][n];
     }

     //static const auto speedup = []() {
     //    std::ios::sync_with_stdio(false);
     //    std::cin.tie(nullptr);
     //    std::cout.tie(nullptr);
     //     return 0;
     //}();
};

vector<tree_node::TreeNode*> Solution::memo[16][16] = {};


class BL {
    vector<pair<int, int>> ranges;
    int current_num;
    int current_itr;
    const int N;
public:
    BL(int n, vector<int>& blacklist):N(n),current_num(0),current_itr(0) {
        sort(blacklist.begin(), blacklist.end());
        if (blacklist.size() == 0)return;
        ranges.push_back({ blacklist[0],blacklist[0] });
        for (int i = 1; i < blacklist.size(); ++i) {
            if (ranges.back().second + 1 == blacklist[i]) {
                ++ranges.back().second;
            }
            else ranges.push_back({ blacklist[i],blacklist[i] });
        }
    }

    int pick() {
        while (ranges.size() != 0 && ranges[current_itr].first == current_num) {
            current_num = ranges[current_itr++].second + 1;
            if (current_num >= N)current_num = 0;
            if (current_itr >= ranges.size())current_itr = 0;

        }
        int result = current_num;
        current_num = (current_num + 1) % N;
        return result;
    }
};

//2642. Design Graph With Shortest Path Calculator
class Graph {
    vector<vector<pair<int, int>>> connections;
public:
    Graph(int n, vector<vector<int>>& edges) {
        connections = vector<vector<pair<int, int>>>(n);
        for (const auto& edge : edges) {
            addEdge(edge);
        }
    }

    inline void addEdge(const vector<int>& edge) {
        connections[edge[0]].push_back({ edge[2],edge[1] });
    }

    int shortestPath(int node1, int node2) {
        if (node1 == node2)return 0;

        vector<long long> shortest(connections.size(), LONG_MAX);
        priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> queue;

        for (const auto& edge : connections[node1]) {
            queue.push(edge);
        }

        while (not queue.empty()) {
            auto top = queue.top(); queue.pop();
            if (shortest[top.second] <= top.first)continue;
            if (top.second == node2)return top.first;

            shortest[top.second] = top.first;
            for (const auto& edge : connections[top.second]) {
                queue.push({ top.first + edge.first,edge.second });
            }
        }

        return -1;
    }
};