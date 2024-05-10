#pragma once
#include "../../../GeneralLibs/Headers/includes.hpp";
#include "DataStructures.hpp"

class Solution {
public:
	//19. Remove Nth Node From End of List
	int length(ListNode* head)
	{
		int length = 0;

		for (; head != nullptr; head = head->next)
		{
			length++;
		}

		return length;
	}

	ListNode* removeNthFromEnd(ListNode* head, int n)
	{
		int size = this->length(head);
		ListNode* iter = head;

		for (int k = 1; k < size - n; k++) iter = iter->next;

		if (head == iter)
		{
			if (size == n && size > 1)head = head->next;
			else if (size > 1)head->next = head->next->next;
			else head = nullptr;
		}
		else
		{
			iter->next = iter->next->next;
		}

		return head;
	}

	ListNode* removeNthFromEnd2(ListNode* head, int n) {
		ListNode* prev = nullptr;
		ListNode* gap = head,*itr = head;

		while (n--)gap = gap->next;

		if (gap == nullptr)return itr->next;

		while (gap != nullptr) {
			prev = head;
			itr = itr->next;
			gap = gap->next;
		}

		prev->next = itr->next;

		return head;
	}

	//
	ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {

		ListNode* head = new ListNode();
		ListNode* iter = head;

		for (int next = 0; l1 != nullptr || l2 != nullptr;)
		{
			int f = (l1 == nullptr) ? 0 : l1->val;
			int s = (l2 == nullptr) ? 0 : l2->val;

			iter->val = next + (f + s) % 10;
			next = (f + s) / 10;

			l1 = (l1 == nullptr) ? nullptr : l1->next;
			l2 = (l2 == nullptr) ? nullptr : l2->next;

			iter = iter->next = new ListNode();
		}

		return head;
	}

	ListNode* addTwoNumbers_2(ListNode* l1, ListNode* l2) {

		ListNode* head = new ListNode();
		ListNode* iter = head;
		int next = 0;

		while (true)
		{
			int num = l1->val + l2->val + next;

			iter->val = num % 10;
			next = num / 10;

			l1 = l1->next;
			l2 = l2->next;

			if (l1 != nullptr && l2 != nullptr)break;

			iter = iter->next = new ListNode();
		}

		for (; l1 != nullptr; l1 = l1->next) {
			iter->val = (l1->val + next) % 10;
			next = (l1->val + next) / 10;
			iter = iter->next = new ListNode();
		}

		for (; l2 != nullptr; l2 = l2->next) {
			iter->val = (l2->val + next) % 10;
			next = (l2->val + next) / 10;
			iter = iter->next = new ListNode();
		}

		return head;
	}

	ListNode* find_min(vector<ListNode*>& lists)
	{
		if (lists.size() == 0)return nullptr;

		auto itr = min_element(lists.begin(), lists.end(), [](const ListNode* l, const ListNode* r) {return l->val < r->val; });
		auto result = *itr;

		if (*itr != nullptr)*itr = (*itr)->next;
		if (*itr == nullptr)lists.erase(itr);

		return result;
	}

	ListNode* mergeKLists(vector<ListNode*>& lists) {
		vector<ListNode*> filtered_lists = {};
		copy_if(lists.begin(), lists.end(), back_inserter(filtered_lists), [](const ListNode* node) {return node != nullptr; });

		ListNode* head = this->find_min(filtered_lists);
		ListNode* iterator = head;

		while (filtered_lists.size() > 0)
		{
			iterator = iterator->next = this->find_min(filtered_lists);
		}

		return head;
	}

	ListNode* mergeKLists_2(vector<ListNode*>& lists)
	{
		if (lists.size() == 0)
			return nullptr;

		ListNode* result = lists[0];
		for (int k = 1; k < lists.size(); k++)result = this->mergeTwo(result, lists[k]);

		return result;
	}

	ListNode* mergeTwo(ListNode* l, ListNode* r)
	{
		if (l == nullptr)
			return r;
		if (r == nullptr)
			return l;

		if (l->val < r->val)
		{
			l->next = mergeTwo(l->next, r);
			return l;
		}
		else
		{
			r->next = mergeTwo(l, r->next);
			return r;
		}
	}

	int firstMissingPositive(vector<int>& nums)
	{
		sort(nums.begin(), nums.end());
		auto itr = find(nums.begin(), nums.end(), 1);

		if (itr == nums.end())return 1;

		for (int k = 1; itr != nums.end(); itr++)
		{
			if (itr != nums.begin() && *itr == *(itr - 1))continue;
			if ((*itr) - k != 0)return k;
			k++;
		}

		return nums[nums.size() - 1] + 1;
	}

	int firstMissingPositive_2(vector<int>& nums)
	{
		int positive_size = count_if(nums.begin(), nums.end(), [](const int x) {return x > 0; });
		bool* present = new bool[positive_size];

		for (int num : nums)
		{
			if (num > 0 && num < positive_size + 1)
			{
				present[num - 1] = false;
			}
		}

		for (int k = 0; k < positive_size; k++)
		{
			if (present[k])
			{
				delete present;
				return k + 1;
			}
		}

		delete[] present;
		return positive_size + 1;
	}

	bool add(string& to, string& from, size_t pos) {
		if (pos < from.length()) {
			to += from[pos];
			return true;
		}
		return false;
	}

	string convert(string s, int numRows) {
		const size_t length = s.length();
		string result = "";

		if (numRows == 1)return s;

		size_t letter = 0, row = 0;
		for (size_t i = 0; i < length;)
		{
			i += add(result, s, letter);
			letter += (numRows - 1) * 2;

			if (row < numRows - 1 && row > 0)
			{
				i += add(result, s, letter - row * 2);
			}

			if (letter > length)
			{
				letter = ++row;
			}
		}

		return result;
	}

	ListNode* swapPairs(ListNode* head)
	{
		if (head == nullptr || head->next == nullptr)return head;

		ListNode* f_head = head;
		ListNode* s_head = head->next;

		for (ListNode* iter1 = f_head, *iter2 = s_head; iter1->next != nullptr;)
		{
			iter1 = iter1->next = iter2->next;
			if (iter2->next == nullptr)break;
			iter2 = iter2->next = iter1->next;
		}

		ListNode* iter1 = s_head, * iter2 = f_head;
		for (ListNode* n1, *n2; iter1 != nullptr; iter1 = n1, iter2 = n2)
		{
			n1 = iter1->next;
			n2 = iter2->next;

			iter1->next = iter2;
			if (n1 != nullptr)iter2->next = n1;
		}

		return s_head;
	}

	long long zeroFilledSubarray(vector<int> nums)
	{
		int start = 0;
		long long amount = 0;
		nums.push_back(1);

		for (int k = 0, start = 0; k < nums.size(); k++)
		{
			if (nums[k] == 0) amount += ++start;
			else start = 0;
		}

		return amount;
	}

	//Minimum Score of a Path Between Two Cities
	void paths(vector<vector<int>>& links, vector<bool>& visited, int edge) {
		if (visited[edge]) return;

		visited[edge] = true;

		for (int link : links[edge])paths(links, visited, link);
	}

	int minScore(int n, vector<vector<int>> roads) {

		vector<vector<int>> links(n + 1);

		for (auto& road : roads) {
			links[road[0]].push_back(road[1]);
			links[road[1]].push_back(road[0]);
		}

		vector<bool> root_linked(n + 1, false);

		paths(links, root_linked, 1);

		int min = INT_MAX;
		for (auto& road : roads)
			if ((root_linked[road[0]] || root_linked[road[1]]) && road[2] < min)
				min = road[2];

		return min;
	}

	//best liniar solution, previous does not work for each case
	int maxSubArray(vector<int>& nums)
	{
		int max = nums[0];
		int current_max = max;

		for (int k = 1; k < nums.size(); k++)
		{
			current_max += (current_max + nums[k] < nums[k]) ? -current_max + nums[k] : nums[k];

			if (current_max > max)max = current_max;
		}

		return max;
	}

	//Longest cycle in a Graph
	int path(vector<int>& edges, unordered_map<int, int>& visited, int edge, int k = 0) {
		if (edges[edge] == -1)
			return -1;

		auto it = visited.find(edge);
		if (it != visited.end())
			return k - it->second + 1;

		visited[edge] = k + 1;
		int res = path(edges, visited, edges[edge], k + 1);
		edges[edge] = -1;

		return res;
	}

	int longestCycle(vector<int>& edges) {
		unordered_map<int, int> visited;
		visited.reserve(edges.size());

		int max_len = -1;
		for (int i = 0; i < edges.size(); i++) {

			int len = path(edges, visited, i);

			if (len > max_len) {
				max_len = len;
			}
		}

		return max_len;
	}

	//Minimum Path Sum
	int minPathSum(vector<vector<int>> grid) {

		const int r = grid.size();
		const int c = grid[0].size();
		const size_t v = r * c;

		int n = 0, m = 1;
		for (int k = 1; k < v; k++) {

			if (n == 0)grid[n][m] += grid[n][m - 1];
			else if (m == 0)grid[n][m] += grid[n - 1][m];
			else grid[n][m] += min(grid[n - 1][m], grid[n][m - 1]);

			if (n + 1 == r || m == 0) {
				int pm = m;
				m = min(n + m + 1, c - 1) + 1;
				n = (c <= (pm + n + 1)) ? pm + n - c + 1 : -1;
			}

			m--;
			n++;

		}

		return grid[r - 1][c - 1];
	}

	void handle_diagonal(int& n, int& m, const int N) {

		if (n + 1 == N || m == 0) {
			int pm = m;
			m = min(n + m + 1, N - 1) + 1;
			n = (N <= (pm + n + 1)) ? pm + n - N + 1 : -1;
		}

		m--;
		n++;
	}

	int isPalindrome(int x) {

		int binary_first = 30;

		while ((1 ^ (x >> binary_first)) && --binary_first > 0);

		return binary_first;
	}

	int getRange(const vector<int>& days, int l, int r, int len) {

		int result = l;

		for (const int aprox = days[l] + len; result <= r && days[result] < aprox; result++);

		return result;
	}

	int helper(const vector<int>& days, const vector<int>& costs, int l, int r) {

		if (l > r) return 0;
		if (l == r) return min(costs[2], min(costs[0], costs[1]));

		int best = (r - l + 1) * costs[0];

		int offset = best;
		for (int k = 1; k + l < r; k++) {
			int nested_result = costs[0] * k + helper(days, costs, l + k, r);
			if (nested_result < offset)offset = nested_result;
		}

		int week = costs[1] + helper(days, costs, getRange(days, l, r, 7), r);
		int month = costs[2] + helper(days, costs, getRange(days, l, r, 30), r);

		if (best > week)best = week;
		if (best > month)best = month;
		if (best > offset)best = offset;

		return best;
	}

	int mincostTickets(vector<int> days, vector<int> costs) {
		return helper(
			days,
			costs,
			0,
			days.size() - 1
		);
	}

	int compress(vector<char>& chars) {

		const size_t size = chars.size();
		int local_count = 1;
		int write_on = 0;

		string count = "";

		for (int k = 0; k < size; k++) {

			if (k + 1 == size || chars[k] != chars[k + 1]) {

				chars[write_on++] = chars[k];

				if (local_count > 1) {
					count = to_string(local_count);
					memcpy(&chars[write_on], &count[0], count.length());
					write_on += count.length();
				}

				local_count = 0;
			}

			local_count++;
		}

		return write_on;
	}

	ListNode* detectCycle(ListNode* head) {
		unordered_map<ListNode*, bool> cycle = {};

		int k = 0;
		while (head != nullptr) {

			if (cycle[head])break;

			cycle[head] = 1;
			head = head->next;
		}

		return head;
	}

	//Spells and Potions
	vector<int> successfulPairs(vector<int> spells, vector<int> potions, long long success) {

		vector<int> result = vector<int>(spells.size());

		sort(potions.begin(), potions.end());

		for (int k = 0; k < spells.size(); k++) {
			uint64_t div = (success + spells[k] - 1) / spells[k];
			auto pos = lower_bound(potions.begin(), potions.end(), div);
			result[k] = (pos == potions.end()) ? 0 : potions.size() - distance(potions.begin(), pos);
		}

		return result;

	}

	//Boats to Save People
	int numRescueBoats(vector<int> people, int limit) {
		sort(people.begin(), people.end());

		int count = 0;
		for (int r = people.size() - 1, l = 0; l <= r; r--) {
			if ((people[r] + people[l]) <= limit) {
				l++;
			}
			count++;
		}

		return count;
	}

	//Optimal Partition of String

	int partitionString(string s) {
		int count = 0;
		int letters = 0;

		for (int k = 0; k < s.length(); k++) {
			int bit = 1 << (s[k] - 97);
			if (letters & bit) {
				count++;
				letters = 0;
			}
			letters |= bit;
		}

		return count + 1;
	}


	//2316. Count Unreachable Pairs of Nodes in an Undirected Graph <- solved
	inline uint64_t sum(uint64_t n) {
		return (n > 1) ? n * (n - 1) / 2 : 0;
	}

	long long countPairs(int n, vector<vector<int>>& edges) {

		vector<int> grouped(n, 0);

		int groups = 0;
		for (auto& pair : edges) {
			int g1 = grouped[pair[0]];
			int g2 = grouped[pair[1]];
			if (bool(g1) ^ bool(g2)) {
				grouped[pair[0]] = grouped[pair[1]] = (g1 > g2) ? g1 : g2;
			}
			else if (g1 == 0) {
				grouped[pair[0]] = grouped[pair[1]] = ++groups;
			}
			else if (g1 != g2) {
				for (int k = 0; k < n; k++)if (grouped[k] == g2)grouped[k] = g1;
			}
		}

		if (groups == 0)return sum(n);

		vector<size_t> g_amount(++groups);
		for (int k = 0; k < n; k++)
			g_amount[grouped[k]]++;

		long long result = 0;
		for (int k = 1; k < groups; k++) {
			result += (n - g_amount[k]) * g_amount[k];
			n -= g_amount[k];
		}

		return result + sum(g_amount[0]);
	}

	//Minimize Maximum of Array
	int minimizeArrayValue(vector<int> nums) {

		//use such algorithm but add a sliced part,that sooner will be added to parts,that are smaller,than biggest number
		//select first pick,
		//example 3 8 20 11 1, pick 20,because 8 is less
		//int r = nums.size()-1;
		//while (r > 0 && nums[r] < nums[r - 1])r--;
		//
		const size_t size = nums.size();
		int local_max = 0;
		int sliced = 0;

		for (int r = size - 1, k = size - 2; 0 <= k; k--) {

			size_t diff = abs(nums[k] - nums[k + 1]);

			if (nums[k] < nums[k + 1]) {

				if (diff <= sliced) {
					nums[k] = nums[k + 1];
					sliced -= diff;
				}
				else {
					sliced += diff * (r - k);
				}

			}
			else {
				nums[k] = nums[k + 1];
				sliced += diff;
			}
		}

		return local_max;
	}

	//Number of Closed Islands

	void markIsland(vector<vector<int>>& grid, size_t x, size_t y, pair<int, int>& count) {

		static const vector<pair<int, int>> directions = {
			{1,0},
			{0,1},
			{-1,0},
			{0,-1}
		};

		count.first++;
		grid[y][x] = 1;

		for (auto& dir : directions) {
			size_t ny = y + dir.first;
			size_t nx = x + dir.second;

			if (nx < grid[0].size() && ny < grid.size() && !grid[ny][nx]) {
				markIsland(grid, nx, ny, count);
			}
			else if (x == 0 || y == 0 || (x + 1) == grid[0].size() || (y + 1) == grid.size()) {
				count.second = INT_MIN;
			}
			else {
				count.second++;
			}
		}
	}

	int closedIsland(vector<vector<int>> grid) {

		const size_t row = grid.size();
		const size_t col = grid[0].size();

		int count = 0;
		pair<int, int> local = { 0,0 };
		for (size_t r = 0; r < row; r++)
			for (size_t c = 0; c < col; c++)
				if (!grid[r][c]) {
					markIsland(grid, c, r, local);
					if (local.first < (local.second >> 1))count++;
					local = { 0,0 };
				}

		return count;
	}

	//Decode Ways

	int numDecodings(string s) {

		if (s.length() == 0) return 0;

		vector<uint32_t> fib(46, 1);
		for (int i = 2; i < 46; i++)
			fib[i] = fib[i - 1] + fib[i - 2];

		size_t size = s.length();
		size_t l = 0;
		int count = 1;

		for (int k = 0; k < size; k++) {
			if (s[k] == '0') return 0;

			int range = 0;

			if (s[k] > '2' || (k + 1) == size) {
				range = k - l;
				if (k > 0 && (s[k - 1] == '1' || (s[k - 1] == '2' && s[k] < '7'))) {
					range++;
				}
				l = k + 1;
			}
			else if (k + 1 < size && s[k + 1] == '0') {
				range = k - l;
				l = ++k + 1;
			}

			if (range > 1) count *= fib[range];
		}

		return count;
	}

	//Number of Enclaves
	void sink_land(vector<vector<int>>& grid, size_t x, size_t y) {
		if (!grid[x][y])return;

		grid[y][x] = 0;

		if (x + 1 < grid[0].size())sink_land(grid, x + 1, y);
		if (x - 1 < grid[0].size())sink_land(grid, x - 1, y);
		if (y + 1 < grid.size())sink_land(grid, x, y + 1);
		if (y - 1 < grid.size())sink_land(grid, x, y - 1);

	}

	int numEnclaves(vector<vector<int>>& grid) {
		const size_t row = grid.size();
		const size_t col = grid[0].size();

		for (size_t r = 0; r < row; r++) {
			if (r == 0 || r + 1 == row)
				for (int c = 0; c < col; c++)sink_land(grid, c, r);
			else {
				sink_land(grid, 0, r);
				sink_land(grid, col - 1, r);
			}
		}

		int count = 0;
		for (size_t k = 0, c = row * col; k < c; k++)
			count += grid[k / col][k % col];

		return count;
	}

	//Clone graph

	node_1::Node* helper(std::vector<node_1::Node*>& created, node_1::Node* current) {

		const int val = current->val - 1;

		if (created.size() <= val)created.resize(val + 1, nullptr);
		node_1::Node* copy = new node_1::Node(val + 1, vector<node_1::Node*>(current->neighbors.size(), nullptr));

		created[val] = copy;

		int k = 0;
		for (auto& neighbor : current->neighbors)
			copy->neighbors[k++] = (neighbor->val <= created.size() && created[neighbor->val - 1] != nullptr) ? created[neighbor->val - 1] : helper(created, neighbor);

		return created[val] = copy;
	}

	node_1::Node* cloneGraph(node_1::Node* node) {
		if (node == nullptr)return nullptr;

		std::vector<node_1::Node*> created = {};
		return helper(created, node);
	}

	//Climb Stairs
	int climbStairs(int n) {
		size_t x = 1, y = 2;
		while (n--) {
			y += x;
			x = y - x;
		}
		return x;
	}

	//95. Unique Binary Search Trees II

	inline tree_node::TreeNode* copy_tree(tree_node::TreeNode* root) {
		return root == nullptr ? nullptr : new tree_node::TreeNode(root->val,copy_tree(root->left),copy_tree(root->right));
	}

	tree_node::TreeNode* insert(tree_node::TreeNode* root, int val) {
		if (root == nullptr)return new tree_node::TreeNode(val);
		if (root->val < val)root->right = insert(root->right, val);
		else if (root->val > val)root->left = insert(root->left, val);
		return root;
	}

	vector<tree_node::TreeNode*> generateTrees(int n) {
		vector<tree_node::TreeNode*> result = {};

		for (int k = 1; k <= n; ++k) {
			//generate possible left and right pairs into vectors below
			vector<tree_node::TreeNode*> left, right;
			for (int i = 0; i < left.size(); ++i)
				for (auto node : right)
					result.push_back(new tree_node::TreeNode(k, copy_tree(left[i]), copy_tree(node)));
		}

		return result;
	}


	//Maximum Number of Events That Can Be Attended II
	int maxValue(vector<vector<int>> events, int k) {
		
		//TODO : process all events adding them and checking if count of present events <= k and if events[i] be profitable replaced with present
		// O(n^2) complexety

		//Sort events by value/spending time and start with top
		//O(log2(n)) + (O(n)) -> O(log2(n))

		std::sort(events.begin(), events.end(), [](vector<int> & l, vector<int> & r) {return (l[1] - l[0])/double(l[2]) < (r[1] - r[0]) / double(r[2]); });

		return -1;

	}

	using u16 = unsigned short int;

	void path_search(string colors, vector<vector<u16>>& visited,int node) {

	}

	int largestPathValue(string colors, vector<vector<int>>& edges) {

		//vector of fixed arrays(26) to mark count of colors for each road
		vector<vector<u16>> roads;
		//vector-representation of fixed size (count of nodes) for which path does node belong
		vector<vector<u16>> visited(colors.length());

		sort(edges.begin(), edges.end(), [](vector<int>& l, vector<int>& r) {return l[0] < r[0]; });



	}

	//20. Valid Parentheses
	bool isValid(string s) {
		const size_t len = s.length();
		string stack = string(len/2, ' ');
		size_t index = 0;

		if (len & 1)return false;

		for (int k = 0; k < len; k++) {

			if (s[k] == ')' || s[k] == '}' || s[k] == ']') {
				if (index == 0)return false;
				if (s[k] - 2 != stack[index - 1] && s[k] - 1 != stack[index - 1])return false;
				index--;
			}
			else {
				if (index == len / 2)return false;
				stack[index++] = s[k];
			}
		}

		return index==0;
	}

	//21. Merge Two Sorted Lists
	ListNode* mergeTwoLists(ListNode* list1, ListNode* list2) {
		if (list1 == nullptr)return list2;
		if (list2 == nullptr)return list1;

		ListNode* head;

		if (list1->val > list2->val) {
			head = list2;
			list2 = list2->next;
		}
		else {
			head = list1;
			list1 = list1->next;
		}

		ListNode* iter = head;
		while (list1 != nullptr && list2 != nullptr) {
			if (list1->val < list2->val) {
				iter = iter->next = list1;
				list1=list1->next;
			}
			else {
				iter = iter->next = list2;
				list2 = list2->next;
			}
		}

		if (list1 == nullptr)iter->next = list2;
		else iter->next = list1;

		return head;
	}
	//145. Binary Tree Postorder Traversal
	void helper(tree_node::TreeNode* current, vector<int>& result) {
		if (current == nullptr)return;
		helper(current->right, result);
		result.push_back(current->val);
		helper(current->left, result);
	}

	vector<int> postorderTraversal(tree_node::TreeNode* root) {
		vector<int> result = {};
		helper(root, result);
		return result;
	}

	//521. Longest Uncommon Subsequence I
	int findLUSlength(string a, string b) {
		return (a == b) ? -1 : max(a.length(), b.length());
	}

	//522. Longest Uncommon Subsequence II
	int findLUSlength(vector<string>& strs) {
		
		if (strs.size() == 0)return -1;

		size_t size = strs.size();
		string current = strs[0];
		for (int k = 1; k < size; k++) {
			if (current == strs[k])
				current = "";
			else if (current.length() < strs[k].length())
				current = strs[k];
		}


		return (current.length() == 0) ? -1 : current.length();
	}

	//2390. Removing Stars From a String
	string removeStars(string s) {
		string result = "";
		int stars = 0;

		for (int k = s.length() - 1; 0 <= k; k--) {
			if (s[k] == '*') {
				stars++;
			}
			else if(stars == 0){
				result = s[k] + result;
			}
			else {
				stars--;
			}
		}

		return (stars > 0) ? string(stars,'*') : result;
	}

	//2390. Removing Stars From a String
	string simplifyPath(string path) {
		vector<string> instructions = {};

		string current;
		for (int k = 0; k < path.length(); k++) {
			if (path[k] == '/') {
				if(current.length() > 0 && current != ".")instructions.push_back(current);
				current = "";
			}
			else {
				current += path[k];
			}
		}
		if (current != ".")instructions.push_back(current);

		int skip = 0;
		string result = "";
		for (int k = instructions.size() - 1; 0 <= k; k--) {
			if (instructions[k] == "..")
				k--;
			else if (skip == 0)
				result = "/" + instructions[k] + result;
			else
				skip--;
		}

		return result;
	}

	string simplifyPath2(string path) {
		string current;
		string result = "";
		int skip = 0;
		for (int k = path.length()-1; 0 <= k; --k) {
			if (path[k] == '/') {
				if (current.length() > 0 && current != ".") {
					if (current == "..") skip++;
					else if (skip == 0) result = "/" + current + result;
					else skip--;
				}
				current = "";
			}
			else {
				current = path[k] + current;
			}
		}

		return (result == "") ? "/" : result;
	}

	//946. Validate Stack Sequences
	bool validateStackSequences(vector<int> pushed, vector<int> popped) {

		stack<int> stack = {};

		const size_t size = popped.size();
		int push = 0, pop = 0;
		while (pop < size) {
			if (push == size)return false;
			if (push < size)stack.push(pushed[push++]);
			while (stack.size() > 0 && stack.top() == popped[pop]) {
				stack.pop();
				pop++;
			}
		}

		return true;
	}
	//516. Longest Palindromic Subsequence

	int longestPalindromeSubseq(string s) {
		const size_t len = s.length();
		int max = 1;
		for (int k = 1; k < len; k++) {

			bool mono = true;
			int l = k-1;
			int r = k+1;
			string palindrome = string(1, s[k]);

			while (0 <= l && r < len) {
				if (s[l] == s[r]) {
					mono = palindrome[0] == s[l];
					palindrome = s[l] + palindrome + s[r];
				}
				else if (mono && s[l] == palindrome[0]) {
					palindrome += s[l];
				}
				else if (mono && s[r] == palindrome[0]) {
					palindrome += s[r];
				}
				l--;
				r++;
			}

			int plen = palindrome.length();

			if (mono) {
				for (; 0 <= l; l--)
					if (s[l] == palindrome[0])plen++;
				for (; r < len; r++)
					if (s[r] == palindrome[0])plen++;
			}

			max = (max < plen)? plen : max;
		}
		return max;
	}


	//2218. Maximum Value of K Coins From Piles
	int maxValueOfCoins(vector<vector<int>>& piles, int k) {
		return -1;
	}

	//1431. Kids With the Greatest Number of Candies
	vector<bool> kidsWithCandies(vector<int>& candies, int extraCandies) {
		vector<bool> result(candies.size(),false);
		int max = *max_element(candies.begin(), candies.end());
		size_t size = candies.size();

		for (int k = 0; k < size; k++)
			result[k] = max <= (candies[k] + extraCandies);
		
		return result;
	}

	//1768. Merge Strings Alternately
	string mergeAlternately(string word1, string word2) {
		const size_t len1 = word1.length();
		const size_t len2 = word2.length();
		string result = "";

		size_t l = 0, r = 0;
		while (l < len1 && r < len2)
			result += string(1,word1[l++]) + word2[r++];
		
		if (l < len1)
			result += word1.substr(l, len1 - l + 1);
		if (r < len2)
			result += word1.substr(r, len2 - r + 1);
		
		return result;
	}

	//5. Longest Palindromic Substring

	string finder(const string& s, size_t l, size_t r, bool mono = true) {

		if (mono) {
			if (r + 1 < s.length() && 0 < l && s[l - 1] == s[r + 1]) {
				return finder(s, l - 1, r + 1, s[l - 1] == s[l]);
			}
			if (r + 1 < s.length() && s[r] == s[r + 1]) {
				return finder(s, l, r + 1, true);
			}
			if (0 < l && s[l] == s[l - 1]) {
				return finder(s, l - 1, r, true);
			}
		}
		else if (0 < l && r + 1 < s.length() && s[l - 1] == s[r + 1]) {
			return finder(s, l - 1, r + 1, false);
		}
		
		return s.substr(l,r-l+1);
	}

	string longestPalindrome(string s) {
		const size_t len = s.length();
		string max = "";
		for (int k = 0; k < len; k++) {
			string nested = finder(s, k, k);
			if (nested.length() > max.length())max = nested;
		}
		return max;
	}

	//120. Triangle

	int dropper(const vector<vector<int>>& triangle,int row = 0,int index = 0) {
		if (row == triangle.size())return 0;
		return triangle[row][index] + min(dropper(triangle,row+1,index),dropper(triangle,row+1,index+1));
	}

	//int minimumTotal(vector<vector<int>> triangle) {
	//	return dropper(triangle);
	//}

	int minimumTotal(vector<vector<int>> triangle) {

		const size_t rows = triangle.size();

		for (int k = 1; k < rows; k++) {
			triangle[k][0] += triangle[k - 1][0];
			triangle[k][k] += triangle[k - 1][k - 1];
		}

		for (int k = 1; k < rows; ++k)
			for (int j = 1; j < k; ++j)
				triangle[k][j] += min(triangle[k - 1][j],triangle[k - 1][j - 1]);

		return *min_element(triangle[rows - 1].begin(), triangle[rows - 1].end());
	}


	//375. Guess Number Higher or Lower II

	int spliter(int n) {
		if (n < 4)return n - 1;
		if (n < 9)return 2*n - 4;
		return n - 4 + max(spliter(n - 5), n - 1);
	}

	int getMoneyAmount(int n) {
		return spliter(n);
	}

	//392. Is Subsequence
	bool isSubsequence(string s, string t) {
		if (s.length() > t.length())return false;

		const size_t len = t.length();
		int index = 0;
		for (int k = 0; k < len; ++k)
			if (s[index] == t[k])index++;

		return index == s.length();
		
	}


	//1372. Longest ZigZag Path in a Binary Tree

	//state meanings :

	int zigZager(tree_node::TreeNode* current, int length,bool state) {
		if (current == nullptr)return length;

		int l = zigZager(current->left, (state) ? length + 1 : 0, 0);
		int r = zigZager(current->right, (!state) ? length + 1 : 0, 1);

		return (l > r) ? l : r;
	}

	int longestZigZag(tree_node::TreeNode* root) {
		return max(
			zigZager(root->left, 0, 0),
			zigZager(root->right, 0, 1)
		);
	}

	int widthOfBinaryTree(tree_node::TreeNode * root) {
		int maxWidth = 1;
		queue<pair<tree_node::TreeNode*, size_t>> q;
		q.push(make_pair(root, 1));

		while (!q.empty()) {
			int size = q.size();
			size_t left = q.front().second, right = left;

			for (int i = 0; i < size; i++) {
				tree_node::TreeNode* node = q.front().first;
				right = q.front().second;
				
				q.pop();

				if (node->left) q.push({ node->left, right * 2 });
				if (node->right) q.push({node->right, right * 2 + 1});
			}

			maxWidth = max(maxWidth, int(right - left + 1));
		}

		return maxWidth;
	}

	//338. Counting Bits
	vector<int> countBits(int n) {	
		if (n == 0)return { 0 };

		static vector<int> previous = { 0,1 };
		vector<int> result(n + 1);
		result[1] = 1;

		int bit_overflow = 1;
		for (int k = 2; k <= n; k++) {
			result[k] = previous[k ^ (1 << bit_overflow)] + 1;
			if (previous.size() <= k)
				previous.push_back(result[k]);
			if (result[k] == bit_overflow)
				++bit_overflow;
		}

		return result;
	}

	//118. Pascal's Triangle
	vector<vector<int>> generate(int numRows) {
		vector<vector<int>> result(numRows);

		result[0] = { 0 };

		for (int k = 1; k < numRows; k++) {
			result[k] = vector<int>(k + 1);
			for (int c = 0; c <= k; c++)
				result[k][c] = (c == 0 || c == k) ? 1 : (result[k - 1][c] + result[k - 1][c - 1]);
		}

		return result;
	}

	//879. Profitable Schemes
	int profitableSchemes(int n, int minProfit, vector<int>& group, vector<int>& profit) {
		return -1;
	}

	//8. String to Integer(atoi)
	int myAtoi(string s) {
		static const int max_div = 214748364;
		int sign = 0;
		int index = 0;
		size_t len = s.length();
		while (s[index] == ' ' && index+1 < len)index++;

		if (s[index] == '+') {
			sign = 1;
			index++;
		}
		else if (s[index] == '-') {
			sign = -1;
			index++;
		}
		else if (isdigit(s[index])) {
			sign = 1;
		}
		else return 0;

		while (s[index] == '0')index++;

		int result = 0;
		while (isdigit(s[index]) && index < len) {
			if (result > max_div)return (sign > 0) ? INT_MAX : INT_MIN;
			if (result == max_div) {
				if (s[index] > '8' && sign < 0)return INT_MIN;
				if (s[index] > '7' && sign > 0)return INT_MAX;
				len = index + 1;
			}
			result = result * 10 + s[index++] - '0';
		}

		return result * sign;
	}

	//1312. Minimum Insertion Steps to Make a String Palindrome
	int minInsertions(string s) {
		//1)parse string in order to find all palindromic subsequences 
		//2)calculate len between them and sum it up

		int min = s.length();

		size_t len = s.length()-1;
		for (int k = 1; k < len; k++) {
			int local = len + 1;
		}

		return min;
	}

	//1416. Restore The Array
	int numberOfArrays(string s, int k) {
		static const int modulo = 10e9 + 7;
		const size_t N = s.length();
        vector<size_t> vars(N + 1, 0);
        vars[N] = 1;

		for (int k = 0; k < N; k++)s[k] -= 48;

        for (int i = N - 1; i >= 0; i--) {
            if (!s[i]) continue;
            size_t num = 0;
            for (int j = i; j < N; j++) {
                num = num * 10 + s[j];
                if (num > k) break;
                vars[i] = (vars[i] + vars[j + 1]) % modulo;
            }
        }

        return vars[0];
	}

	//1046. Last Stone Weight
	int lastStoneWeight(vector<int>& stones) {
		vector<int> additional(stones.size() / 2);

		int max2 = 0;
		int size1 = stones.size();
		int size2 = 0;
		do {

			sort(stones.begin(), stones.end());

			if (stones[--size1] == 1)return !(size1 % 2);

			while (size1 > 0) {
				if (stones[size1] < max2)break;
				if (stones[size1] != stones[size1 - 1]) {
					int diff = stones[size1] - stones[size1 - 1];
					additional[size2++] = diff;
					if (diff > max2)max2 = diff;
				}
				size1 -= 2;
			}

			if (size1 + size2 < 0)return 0;

			memcpy(&stones[size1 + 1], &additional[0], size2 * 4);
			size1 += size2 + 1;
			max2 = size2 = 0;

		} while (size1 > 1);

		return stones[0];
	}

	int lastStoneWeight2(vector<int>& stones) {
		priority_queue<int> queue(stones.begin(), stones.end());

		while (queue.size() > 1) {
			int max1 = queue.top(); queue.pop();
			int max2 = queue.top(); queue.pop();
			if (max1 != max2) {
				queue.push(max1 - max2);
			}
		}

		return (queue.empty()) ? 0 : queue.top();
	}

	int lastStoneWeight3(vector<int>& stones) {
		sort(stones.begin(), stones.end());

		while (stones.size() > 1) {
			int size = stones.size();
			int diff = stones[size - 1] - stones[size - 2];
			stones.pop_back();
			stones.pop_back();
			if (diff > 0) {
				auto it = lower_bound(stones.begin(), stones.end(), diff);
				stones.insert(it, diff);
			}
		}

		return stones.empty() ? 0 : stones[0];
	}

	//4. Median of Two Sorted Arrays
	double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
		size_t median = nums1.size() + nums2.size();
		size_t half = median / 2;

		size_t n = 0;
		size_t m = 0;
		while (n < nums1.size() && m < nums2.size()  && n + m < median) {
			if (nums1[n] > nums2[m])m++;
			else n++;
		}

		cout << n << "|" << m << "\n";

		return (median%2)? (nums1[n]+nums2[m])/2 : nums1[n];
	}

	//29. Divide Two Integers
	int divide(int dividend, int divisor) {
		int result = 0;
		int sign = (dividend > 0 ^ divisor > 0) ? -1 : 1;
		while (divisor <= dividend) {
			dividend -= divisor;
			result++;
		}
		return result * sign;
	}

	//11. Container With Most Water
	int maxArea(vector<int> height) {

		function<int(int, int)> calc = [height](int l, int r) {return min(height[l], height[r]) * (r - l); };

		size_t size = height.size();
		
		size_t l = 0;
		size_t r = 1;

		for (int k = 2; k < size; ++k) {
			if (height[l] < height[k]) {
				if (height[r] < height[l])r = l;
				l = k;
			}
			else if (height[r] < height[k]) {
				r = k;
			}
		}

		if (l > r)swap(l, r);
		
		for (int k = l-1; 0<= k; --k)
			if (calc(l, r) < calc(k, r))l = k;

		for (int k = r; k < size; k++)
			if (calc(l, r) < calc(l, k))r = k;

		return calc(l,r);
	}

	//1. Two Sum
	vector<int> twoSum(vector<int> & nums, int target) {

		const size_t size = nums.size();
		vector<pair<int, int>> pairs(size);

		for (int k = 0; k < size; k++)
			pairs[k] = { nums[k],k };

		sort(pairs.begin(), pairs.end());

		int l = 0;
		int r = size-1;
		while (l < r) {
			int sum = pairs[l].first + pairs[r].first;
			if (sum > target)r--;
			else if (sum < target)l++;
			else return { pairs[l].second,pairs[r].second};
		}
		return {};
	}

	//14. Longest Common Prefix
	string longestCommonPrefix(vector<string>& strs) {

		if (strs.size() == 0)return "";

		sort(strs.begin(), strs.end(), [](string& l, string& r) {return l.length() < r.length(); });

		string result = "";
		for (int k = 0; k < strs[0].length(); k++) {
			for (int c = 1; c < strs.size(); c++) {
				if (strs[0][k] != strs[c][k])return result;
			}
		}
		return result;
	}

	//258. Add Digits
	inline int addDigits(int num) {
		return (num > 9)? addDigits(num%10 + num/10) : num;
	}

	//278. First Bad Version
	bool isBadVersion(int n) { return true; }

	int firstBadVersion(int n) {
		uint64_t l = 1;
		while (l + 1 < n) {
			uint64_t middle = (n + l) / 2;
			if (isBadVersion(middle)) n = middle;
			else l = middle;
		}
		return (isBadVersion(l)) ? l : n;
	}

	//35. Search Insert Position
	int searchInsert(vector<int>& nums, int target) {
		return distance(nums.begin(),lower_bound(nums.begin(), nums.end(), target));
	}

	//12. Integer to Roman

	string intToRoman(int num) {

		static unordered_map<int, char> toRoman = {
			{1,'I'},
			{5,'V'},
			{10,'X'},
			{50,'L'},
			{100,'C'},
			{500,'D'},
			{1000,'M'}
		};

		string result = "";
		for (int power = 1; num > 0;) {
			int digit = num % 10;

			string local = "";
			if (3 < digit && digit < 9) {
				local = toRoman[5 * power];
				if (digit == 4)local = toRoman[1 * power] + local;
				else local += string(digit - 5, toRoman[1 * power]);
			}
			else if (digit < 4) local = string(digit, toRoman[1 * power]);
			else local = string(1,toRoman[1 * power]) + toRoman[1 * power * 10];

			result = local + result;
			num /= 10;
			if (power < 10e4)power *= 10;
		}

		return result;
	}

	//1432. Max Difference You Can Get From Changing an Integer
	int maxDiff(int num) {
		
		//find most left digit,that is grater than 1 (if it first change it to 1,else change it to 0)
		//find most left digit,that is lower than 9 (change it to 9)
		string s = to_string(num);
		
		char most_digit = 0;
		for (int k = 0; k < s.length();k++) {
			if (s[k] < '9') {
				most_digit = s[k];
				break;
			}
		}
		char least_digit = 0;
		for (int k = 0; k < s.length(); k++) {
			if (s[k] > '1') {
				least_digit = k;
				break;
			}
		}
		int upd = (least_digit > 0) ? 0 : 1;
		least_digit = s[least_digit];

		int min = 0;
		int max = 0;
		for (int k = 0; k < s.length(); k++) {
			max = max * 10 + int((s[k] == most_digit) ? 9 : s[k]-48);
			min = min * 10 + int((s[k] == least_digit) ? upd : s[k]-48);
		}

		return max - min;
	}

	//319. Bulb Switcher
	int bulbSwitch(int n) {
		//test
		vector<bool> lamps(n, 0);
		for (int k = 0; k < n; k++) {

			for (int c = k; c < n; c += k + 1) {
				lamps[c] = !lamps[c];
			}

			size_t on = 0;
			for (bool c : lamps) {
				on += c;
				cout << c;
			}
			cout << "|" << on << "\n";
		}

		return sqrt(n);
	}

	//977. Squares of a Sorted Array
	vector<int> sortedSquares(vector<int>& nums) {
		register const size_t size = nums.size();
		vector<int> result(size);
		int plus = distance(nums.begin(),lower_bound(nums.begin(), nums.end(), 0));
		int minus = plus - 1;

		size_t index = 0;
		while (0 <= minus && plus < size) {
			if (-1*nums[minus] > nums[plus]) {
				result[index] = nums[plus] * nums[plus];
				plus++;
			}
			else {
				result[index] = nums[minus] * nums[minus];
				minus--;
			}
			index++;
		}
		
		while (0 <= minus) {
			result[index++] = nums[minus] * nums[minus];
			minus--;
		}

		while (plus < size) {
			result[index++] = nums[plus] * nums[plus];
			plus++;
		}

		return result;
	}

	//189. Rotate Array
	void rotate(vector<int>& nums, int k) {
		
		k %= nums.size();

		if (k > 0) {
			int diff = nums.size() - k;
			void* temp = malloc(min(diff, k)<<2);
			if (diff < k) {
				memmove(temp, &nums[0], diff << 2);
				memmove(&nums[0], &nums[diff], k << 2);
				memmove(&nums[k], temp, diff << 2);
			}
			else {
				memmove(temp, &nums[diff], k << 2);
				memmove(&nums[k], &nums[0], diff << 2);
				memmove(&nums[0],temp, k << 2);
			}
		}

	}

	void rotate2(vector<int>& nums, int k) {
		const size_t size = nums.size();
		k %= size;

		if (k < size - k) {
			int left = size - 1;
			int next = nums[left];
			nums[left--] = nums[k--];
			while (0 <= k) {
				swap(next, nums[k--]);
				swap(next, nums[left--]);
			}
		}
		else {
			k = size - k;
			int right = 0;
			int next = nums[0];
			nums[right++] = nums[k++];
			while (k < size) {
				swap(next, nums[k++]);
				swap(next, nums[right++]);
			}
		}
	}

	//283. Move Zeroes
	void moveZeroes(vector<int>& nums) {
		const size_t size = nums.size();
		int left = -1;
		for (int k = 0; k < size; k++) {
			if (nums[k] == 0 && left < 0) {
				left = k;
			}
			else if (left != -1) {
				swap(nums[k], nums[left]);
				left = (nums[left + 1] == 0) ? left + 1 : -1;
			}
		}
	}
	//167. Two Sum II - Input Array Is Sorted
	vector<int> twoSum2(vector<int> numbers, int target) {
		static vector<int> answer(2);
		int l = 0, r = numbers.size() - 1;
		while (l < r) {
			if (numbers[l] + numbers[r] == target) {
				answer[0] = l + 1;
				answer[1] = r + 1;
				break;
			}
			else if (numbers[l] + numbers[r] < target)l++;
			else r--;
		}
		return answer;
	}

	//839. Similar String Groups
	bool isSimilar(string& l, string& r) {
		const size_t len = l.length();

		if (len != r.length())return false;
		
		int count = 0;
		for (int k = 0; k < len; ++k) {
			count += l[k] != r[k];
			if (count > 2)return false;
		}
		return true;
	}

	int numSimilarGroups(vector<string> strs) {

		const size_t size = strs.size();
		
		vector<int> grouped(size, 0);
		
		size_t single = 0;
		int groups = 0;
		for (int k = 0; k < size-1; ++k) {
			for (int j = k + 1; j < size; ++j) {
				if (isSimilar(strs[k], strs[j])) {
					int g1 = grouped[k];
					int g2 = grouped[j];
					if (bool(g1) ^ bool(g2)) {
						grouped[k] = grouped[j] = max(g1, g2);
						single++;
					}
					else if (g1 == 0) {
						grouped[k] = grouped[j] = ++groups;
						single += 2;
					}
					else if (g1 != g2) {
						for (int c = 0; c < size; ++c)if (grouped[c] == g2)grouped[c] = g1;
					}
				}
			}
		}

		set<int> result(grouped.begin(), grouped.end());
		return result.size() + (single == size)? -1 : 0;
	}

	//1697. Checking Existence of Edge Length Limited Path
	
	//req - {end_edge,limit_value};
	
	void add(vector<unordered_map<int, int>>& p, int f, int t, int v) {
		if (p[f].find(t) == p[f].end()) {
			p[f][t] = v;
		}
		else if(p[f][t] > v) {
			p[f][t] = v;
		}
	}

	inline static bool comparer(const vector<int>& l, const vector<int>& r) {
		return l[2] < r[2];
	}

	vector<bool> distanceLimitedPathsExist(int n, vector<vector<int>> edgeList, vector<vector<int>> queries) {
		
		size_t k = 0;
		for (auto& query : queries) {
			query.push_back(k++);
		}
		
		sort(edgeList.begin(), edgeList.end(), Solution::comparer);
		sort(queries.begin(), queries.end(), Solution::comparer);

		size_t node_used = 0;
		auto query = queries.begin();
		vector<bool> result(queries.size(), false);
		
		Groups group(n);

		for (auto& edge : edgeList) {

			if ((*query)[2] <= edge[2]) {
				query++;
				if (query == queries.end())break;
			}
			
			group.insert(edge[0], edge[1]);

			int g1 = group[(*query)[0]];
			int g2 = group[(*query)[1]];
			if (g1 == g2 && g1 > 0) {
				result[(*query)[3]] = true;
				query++;
				if (query == queries.end())break;
			}
			else if (g1 > 0 && g2 > 0 && g1 != g2) {
				query++;
				if (query == queries.end())break;
			}
		}

		return result;
	}

	//557. Reverse Words in a String III
	void reverse(string& s, int l, int r) {
		if (r <= l)return;

		while (l < r)
			swap(s[l++], s[r--]);
	}

	string reverseWords(string s) {
		const size_t size = s.length();
		int l = 0;
		for (int k = 0; k < size; k++) {
			if (s[k] == ' ') {
				reverse(s, l, k - 1);
				l = k + 1;
			}
		}
		reverse(s, l, size - 1);
		return s;
	}

	//1579. Remove Max Number of Edges to Keep Graph Fully Traversable
	int maxNumEdgesToRemove(int n, vector<vector<int>> edges) {
		static const pair<size_t, size_t> best = { 1,0 };
		const size_t size = edges.size();
		size_t to_remove = 0;

		Unions Bob(n),Alice(n);
		
		for (auto& edge : edges) {
			if (edge[0] == 3) {
				int l = edge[1] - 1;
				int r = edge[2] - 1;
				if (Bob.groups_amount() != best) {
					to_remove += !Bob.insert(l, r);
					Alice.insert(l, r);
				}
				else to_remove++;
			}
		}

		for (auto& edge : edges) {
			if(edge[0] != 3){
				int l = edge[1] - 1;
				int r = edge[1] - 1;
				if (edge[0] == 1) {
					if (Bob.groups_amount() != best) {
						to_remove += !Bob.insert(l, r);
					}
					else to_remove++;
				}
				else {
					if (Alice.groups_amount() != best) {
						to_remove += !Alice.insert(l, r);
					}
					else to_remove++;
				}
			}
		}
		
		return (Bob.groups_amount() == best && Alice.groups_amount() == best)? to_remove : -1;
	}

	//876. Middle of the Linked List
	ListNode* middleNode(ListNode* head) {
		ListNode* middle = head;
		bool move = 1;
		
		while (head->next != nullptr) {
			if (move)middle = middle->next;
			head = head->next;
		}

		return middle;
	}

	//10. Regular Expression Matching
	bool isMatch(string s, string p) {

	}

	//26. Remove Duplicates from Sorted Array
	int removeDuplicates(vector<int>& nums) {
		const size_t size = nums.size();
		int count = 0;
		for (int k = 0; k < size; ++k) {
			while (nums[k] == nums[k + 1] && k+1 < size)++k;
			nums[count++] = nums[k];
		}
		return count;
	}
	//27. Remove Element
	int removeElement(vector<int>& nums, int val) {
		const size_t size = nums.size();
		int count = 0;
		int i = -1;
		for (int k = 0; k < size; k++) {
			if (nums[k] == val) {
				if (i < 0)i = k;
				++count;
			}
			else if (i > 0) {
				nums[i++] = nums[k];
			}
		}
		return size - count;
	}

	//31. Next Permutation
	void nextPermutation(vector<int>& nums) {

	}
	//33. Search in Rotated Sorted Array
	int helper(const vector<int>& nums, int target, int l, int r) {
		while (l < r) {
			int middle = l + (r - l) / 2;
			if (target == nums[middle])return middle;
			if (nums[l] > nums[r]) {
				//recursive branching on rotation detection
				return max(
					helper(nums, target, l, middle - 1),
					helper(nums, target, middle + 1, r)
				);
			}
			else {
				//standard binary search
				if (nums[middle] >= target)r = middle;
				else l = middle + 1;
			}
		}

		return nums[l] == target ? l : -1;
	}

	int search(const vector<int>& nums, int target) {
		return helper(nums, target, 0, nums.size() - 1);
	}

	//1971. Find if Path Exists in Graph
	bool validPath(int n, vector<vector<int>> edges, int source, int destination) {

		Unions un(n);

		for (auto& edge : edges) un.insert(edge[0], edge[1]);

		return un.find(source) == un.find(destination);
	}

	//17. Letter Combinations of a Phone Number
	vector<string> keyboard = {
		"abc",
		"def",
		"ghi",
		"jkl",
		"mno",
		"pqrs",
		"tuv",
		"wxyz"
	};

	void composer(vector<string>& result, string digits,string current) {
		if (digits.empty()) {
			result.push_back(current);
			return;
		}
			
		for(char c : keyboard[digits.front()-'2'])
			composer(result, digits.substr(1, digits.length() - 1), current + c);
	}

	vector<string> letterCombinations(string digits) {
		vector<string> result;
		
		result.reserve(pow(3, digits.length()));

		if (digits.length()) {
			composer(result, digits, "");
		}
		return result;
	}

	//39. Combination Sum
	void helper(unordered_map<uint64_t, bool>& sums, const vector<int>& candidates, vector<int>& current, vector<vector<int>>& result, int target) {
		if (target == 0) {
			if (sums.insert({ accumulate(current.begin(), current.end(), 0,
					[](const uint64_t& l, const uint64_t& r) {
						return l + (r + 37) * (r + 37);
					}
				),true }).second)
			{
				result.push_back(current);
			}
		}

		for (int c : candidates) {
			if (c <= target) {
				current.push_back(c);
				helper(sums, candidates, current, result, target - c);
				current.pop_back();
			}
		}
	}

	vector<vector<int>> combinationSum(vector<int> candidates, int target) {
		ios::sync_with_stdio(false);
		cin.tie(NULL);

		static unordered_map<uint64_t, bool> sums;
		static vector<int> current;
		vector<vector<int>> result;

		sums.clear();

		helper(sums, candidates, current, result, target);

		return result;
	}

	//1491. Average Salary Excluding the Minimum and Maximum Salary
	double average(vector<int>& salary) {
		int max = std::max(salary[0], salary[1]);
		int min = std::min(salary[0],salary[1]);
		int64_t sum = 0;

		const size_t size = salary.size();
		for (int k = 2; k < size; k++) {
			if (salary[k] > max) {
				sum += max;
				max = salary[k];
			}
			else if (salary[k] < min) {
				sum += min;
				min = salary[k];
			}
			else {
				sum += salary[k];
			}
		}

		return sum / (size - 2);
	}

	//3. Longest Substring Without Repeating Characters
	int lengthOfLongestSubstring(string s) {

		vector<bool> used(256, false);
		const size_t len = s.length();
		
		used[s[0]] = true;
		int max_len = 1;
		int l = 0, r = 0;
		while (r < len-1) {
			if (!used[s[r+1]]) {
				used[s[++r]] = true;
				max_len = max(max_len,r - l + 1);
			}
			else {
				used[s[l++]] = false;
			}
		}

		return max_len;
	}

	//567. Permutation in String
	bool checkInclusion(string s1, string s2) {
		const size_t len1 = s1.length();
		const size_t len2 = s2.length();

		if (len2 < len1)return false;

		vector<int> letters = vector<int>(26, 0);
		for (char c : s1)++letters[c - 'a'];

		int iterations = len2 - len1;
		for (int k = 0; k <= iterations; k++) {
			if (letters[s2[k] - 'a'] > 0) {
				vector<int> to_check(letters.begin(), letters.end());

				int len = len1;
				while (0 <= --len) {
					if (--to_check[s2[k + len] - 'a'] < 0)break;
				}

				if (len == -1)return true;
			}
		}

		return false;
	}

	//509. Fibonacci Number
	int fib(int n) {
		int x = 0, y = 1;
		while (n--) {
			y += x;
			x = y - x;
		}
		return x;
	}

	//1137. N-th Tribonacci Number
	int tribonacci(int n) {
		if (n < 1)return 0;
		int x = 0, y = 1, z = 1;
		while (--n > 1) {
			z += x + y;
			y = z - (x + y);
			x = z - (x + y);
		}
		return z;
	}

	//1822. Sign of the Product of an Array
	int arraySign(vector<int>& nums) {
		sort(nums.begin(), nums.end());
		auto itr = lower_bound(nums.begin(), nums.end(), 0);
		if (itr == nums.end())return -1;
		if ((*itr) == 0)return 0;
		return distance(nums.begin(), itr) & 1 ? -1 : 1;
	}
	int arraySign2(vector<int>& nums) {
		bool is_negative = false;
		for (int num : nums) {
			if (num == 0) {
				return 0;
			}
			if (num < 0) {
				is_negative = !is_negative;
			}
		}
		return is_negative ? -1 : 1;
	}

	//733. Flood Fill
	vector<vector<int>> floodFill(vector<vector<int>>& image, int sr, int sc, int color) {
		static const vector<pair<size_t, size_t>> directions = { {0,1},{1,0},{-1,0},{0,-1} };
		
		int current_color = image[sr][sc];
		image[sr][sc] = color;

		if (current_color == color)return image;

		for (auto& p : directions) {
			size_t x = sc + p.first;
			size_t y = sr + p.second;
			if (x < image[0].size() && y < image.size() && image[y][x] == current_color) {
				floodFill(image, y, x, color);
			}
		}

		return image;
	}

	//695. Max Area of Island
	int search(vector<vector<int>>& grid, pair<size_t, size_t> point) {
		static const vector<pair<size_t, size_t>> directions = { {0,1},{1,0},{-1,0},{0,-1} };

		int result = 1;

		grid[point.first][point.second] = 0;

		for (auto& p : directions) {
			size_t y = point.first + p.first;
			size_t x = point.second + p.second;
			if (x < grid[0].size() && y < grid.size() && grid[y][x] == 1) {
				result += search(grid, { y,x });
			}
		}

		return result;
	}

	int maxAreaOfIsland(vector<vector<int>>& grid) {
		int max = 0;

		const size_t rows = grid.size();
		const size_t columns = grid[0].size();

		for (int r = 0; r < rows; ++r) {
			for (int c = 0; c < columns; ++c) {
				if (grid[r][c] == 1)
					max = std::max(max,search(grid, { r,c }));
			}
		}

		return max;
	}
	//88. Merge Sorted Array
	void merge(vector<int>& nums1, int m, vector<int>& nums2, int n) {
		int k = 0;
		while (k < n) nums1[m++] = nums2[k++];
		sort(nums1.begin(), nums1.end());
	}

	//746. Min Cost Climbing Stairs
	int minCostClimbingStairs(vector<int>& cost) {
		
		for (int k = cost.size() - 3; 0 <= k; --k)
			cost[k] += min(cost[k + 1], cost[k + 2]);

		return cost[0] < cost[1] ? cost[0] : cost[1];
	}

	//290. Word Pattern
	bool wordPattern(string pattern, string s) {
		vector<string> depend(26, "");
		set<string> words;
		
		const size_t len = s.length();
		const size_t len2 = pattern.length();

		int letter = 0;
		for (int k = 0; k < len; ++k) {
			if (letter == len2)return false;
			int index = pattern[letter++] - 'a';
			int left = k;

			while (s[k] != ' ' && k < len)++k;

			string sub = s.substr(left, k - left);

			if (depend[index] == "") {
				depend[index] = sub;
				int size = words.size();
				words.insert(sub);
				if (size == words.size())return false;
			}
			else if (depend[index] != sub)return false;
		}

		return true;
	}

	int rob(vector<int>& nums) {
		const size_t size = nums.size();

		if (size == 1)return nums[0];

		int prev = nums[size - 2];
		for (int k = size - 3; 0 <= k; --k) {
			int curr = nums[k];
			nums[k] += max(nums[k + 2], nums[k + 1] - prev);
			prev = curr;
		}

		return max(nums[0], nums[1]);
	}

	//2215. Find the Difference of Two Arrays
	vector<vector<int>> findDifference(vector<int> nums1, vector<int> nums2) {
		sort(nums1.begin(), nums1.end());
		sort(nums2.begin(), nums2.end());

		vector<vector<int>> result(2);

		const size_t s1 = nums1.size();
		const size_t s2 = nums2.size();
		int l = 0, r = 0;
		while (l < s1 && r < s2) {
			while (l + 1 < s1 && nums1[l] == nums1[l + 1])++l;
			while (r + 1 < s2 && nums2[r] == nums2[r + 1])++r;
			if (nums1[l] < nums2[r]) {
				result[0].push_back(nums1[l++]);
			}
			else if (nums2[r] < nums1[l]) {
				result[1].push_back(nums2[r++]);
			}
			else {
				++l;
				++r;
			}
		}

		while (l < s1) {
			while (l + 1 < s1 && nums1[l] == nums1[l + 1])++l;
			result[0].push_back(nums1[l++]);
		}

		while (r < s2) {
			while (r + 1 < s2 && nums2[r] == nums2[r + 1])++r;
			result[1].push_back(nums2[r++]);
		}

		return result;
	}

	//350. Intersection of Two Arrays II
	vector<int> intersect(vector<int>& nums1, vector<int>& nums2) {
		vector<int> result = {};

		sort(nums1.begin(), nums1.end());
		sort(nums2.begin(), nums2.end());

		const size_t s1 = nums1.size();
		const size_t s2 = nums2.size();
		int l = 0, r = 0;

		while (l < s1 && r < s2) {
			if (nums1[l] < nums2[r])++l;
			else if (nums2[r] < nums1[l])++r;
			else {
				result.push_back(nums1[l++]);
				++r;
			}
		}

		return result;
	}

	//121. Best Time to Buy and Sell Stock
	int maxProfit(vector<int>& prices) {
		const size_t size = prices.size();
		int max = 0;
		int max_value = 0;
		int min_value = 0;

		for (int k = 0; k < size; k++) {
			if (min_value < k) max = std::max(max, prices[k] - prices[min_value]);
			if (prices[min_value] > prices[k])min_value = k;
		}

		return max;
	}

	//617. Merge Two Binary Trees
	tree_node::TreeNode* mergeTrees(tree_node::TreeNode* root1, tree_node::TreeNode* root2) {
		if (root2 == nullptr)return root1;
		if (root1 == nullptr)return nullptr;

		root1->val += root2->val;
		root1->left = mergeTrees(root1->left, root2->left);
		root1->right = mergeTrees(root1->right, root2->right);

		return root1;
	}

	//116. Populating Next Right Pointers in Each Node
	tree_node2::Node* connect(tree_node2::Node* root) {
		static vector<tree_node2::Node*> *current = new vector<tree_node2::Node*>(4096, nullptr);
		static vector<tree_node2::Node*> *next = new vector<tree_node2::Node*>(4096, nullptr);

		(*current)[0] = root;
		int level_size = 1;
		while ((*current)[0] != nullptr) {
			int nls = -1;
			for (int k = 0; k < level_size; ++k) {
				(*current)[k]->next = (*current)[k + 1];
				(*next)[++nls] = (*current)[k]->left;
				(*next)[++nls] = (*current)[k]->right;
			}
			swap(current, next);
			level_size = nls + 1;
		}
		
		return root;
	}
	//740. Delete and Earn
	int deleteAndEarn(vector<int>& nums) {
		int n = 10001;
		vector<int> values(n, 0);
		for (int num : nums)
			values[num] += num;

		int take = 0, skip = 0;
		for (int i = 0; i < n; i++) {
			int takei = skip + values[i];
			int skipi = max(skip, take);
			take = takei;
			skip = skipi;
		}
		return max(take, skip);
	}

	//235. Lowest Common Ancestor of a Binary Search Tree
	tree_node::TreeNode* lowestCommonAncestor(tree_node::TreeNode* root, tree_node::TreeNode* p, tree_node::TreeNode* q) {
		const int pv = p->val;
		const int qv = q->val;

		while (root) {
			int val = root->val;
			if (val == pv || val == qv)break;
			if (val < pv && val > qv)break;
			if (val > pv && val < qv)break;
			root = val < qv ? root->right : root->left;
		}
		return root;
	}
};

//2336. Smallest Number in Infinite Set
class SmallestInfiniteSet {
private:
	vector<int> pushed;
	int max;
public:
	SmallestInfiniteSet() {
		max = 1;
		pushed = {};
	}

	int popSmallest() {
		if (pushed.size() > 0) {
			int result = pushed[0];
			pushed.erase(pushed.begin());
			return result;
		}
		return max++;
	}

	void addBack(int num) {
		if (num >= max)return;

		if (pushed.size() > 0 && *lower_bound(pushed.begin(), pushed.end(), num) != num) {
			pushed.push_back(num);
			sort(pushed.begin(), pushed.end());
		}
	}
};

//146. LRU Cache
class LRUCache
{
	list<pair<int, int>> l;
	unordered_map<int, list<pair<int, int>>::iterator> m;
	int size;
public:
	LRUCache(int capacity) :
		size(capacity)
	{};

	int get(int key)
	{
		if (m.find(key) == m.end())return -1;

		l.splice(l.begin(), l, m[key]);
		
		return m[key]->second;
	}

	void put(int key, int value)
	{
		if (m.find(key) != m.end())
		{
			l.splice(l.begin(), l, m[key]);
			m[key]->second = value;
			return;
		}
		if (l.size() == size)
		{
			auto d_key = l.back().first;
			l.pop_back();
			m.erase(d_key);
		}
		l.push_front({ key,value });
		m[key] = l.begin();
	}
};