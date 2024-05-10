#pragma once
#include "../../../GeneralLibs/Headers/includes.hpp"
#include "DataStructures.hpp"

class Solution {
public:
	//213. House Robber II
	int rob1(vector<int>& nums) {
		const size_t size = nums.size();

		if (nums.size() == 1)return nums[0];

		int prev = nums[size - 2];
		for (int k = size - 3; 0 <= k; --k) {
			int curr = nums[k];
			nums[k] += max(nums[k + 2], nums[k + 1] - prev);
			prev = curr;
		}

		return max(nums[0], nums[1]);
	}

	int rob(vector<int>& nums) {
		if (nums.size() == 1)return nums[0];
		vector<int> nums2(nums.begin() + 1, nums.end());
		nums.back() = 0;
		return max(rob1(nums2), rob1(nums));
	}

	//789. Escape The Ghosts
	bool escapeGhosts(vector<vector<int>>& ghosts, vector<int>& target) {
		function<int(int, int)> dist = [&target](int x,int y) { return abs(target[0] - x) + abs(target[1] - y); };

		uint64_t most = dist(0,0);
		for (auto& p : ghosts) {
			if (dist(p[0], p[1]) <= most)return false;
		}

		return true;
	}

	//566. Reshape the Matrix
	vector<vector<int>> matrixReshape(vector<vector<int>>& mat, int r, int c) {
		
		const size_t ic = mat[0].size();

		if (mat.size() * ic != r * c)return mat;

		vector<vector<int>> result(r, vector<int>(c));

		int index = 0;
		for (int k = 0; k < r; ++k) {
			for (int j = 0; j < c; ++j) {
				result[k][j] = mat[index / ic][index % ic];
				++index;
			}
		}

		return result;
	}

	//542. 01 Matrix

	int dfs(vector<vector<bool>>&visited,vector<vector<int>> & mat,pair<size_t,size_t> pos) {

		static const vector<pair<char, char>> directions = { {1,0},{0,1},{-1,0},{0,-1} };

		if (pos.first >= mat.size() || pos.second >= mat[0].size())
			return INT_MAX;
		if (visited[pos.first][pos.second])
			return mat[pos.first][pos.second];

		visited[pos.first][pos.second] = true;

		int min = INT_MAX;
		for (auto& p : directions) {
			size_t y = pos.first + p.first;
			size_t x = pos.second + p.second;
			if (y < mat.size() && x < mat[0].size())min = std::min(min, mat[y][x]);
		}

		if(min != 0) {
			for (auto& p : directions) {
				min = std::min(min, dfs(visited, mat, {pos.first + p.first,pos.second + p.second}));
			}
		}
		
		mat[pos.first][pos.second] += min;

		return mat[pos.first][pos.second];
	}

	vector<vector<int>> updateMatrix(vector<vector<int>> mat) {
		
		const size_t rows = mat.size();
		const size_t columns = mat[0].size();

		vector<vector<bool>> visited(rows, vector<bool>(columns, false));

		for (int r = 0; r < rows; ++r) {
			for (int c = 0; c < columns; ++c) {
				if (mat[r][c] != 0 && !visited[r][c]) {
					dfs(visited, mat, { r,c });
				}
			}
		}

		return mat;
	}

	//994. Rotting Oranges
	int orangesRotting(vector<vector<int>>& grid) {
		static const vector<pair<char, char>> directions = { {1,0},{0,1},{-1,0},{0,-1} };

		const size_t rows = grid.size();
		const size_t columns = grid[0].size();
		size_t fresh = 0;

		queue<pair<size_t, size_t>> rotting;

		for (int r = 0; r < rows; ++r) {
			for (int c = 0; c < columns; ++c) {
				if (grid[r][c] == 1)++fresh;
				if (grid[r][c] == 2)rotting.push({ r,c });
			}
		}

		int iterations = 0;
		while (rotting.size()) {
			int size = rotting.size();
			for (int k = 0; k < size; ++k) {
				pair<size_t, size_t> pos = rotting.front(); rotting.pop();
				for (auto& p : directions) {
					size_t y = p.first + pos.first;
					size_t x = p.second + pos.second;
					if (y < rows && x < columns && grid[y][x] == 1) {
						grid[y][x] = 2;
						fresh--;
						rotting.push({ y,x });
					}
				}
			}
			++iterations;
		}

		return (fresh == 0)? iterations : -1;
	}

	bool canJump(vector<int>& nums) {
		
		const size_t last = nums.size() - 1;
		
		int max_right = 0;
		for (int k = 0; k < last; ++k) {
			if (nums[k] + k > max_right)max_right = nums[k] + k;
			if (max_right == k)return false;
		}

		return true;
	}

	int jump(vector<int> nums) {
    	const int size = nums.size();
        nums.back() = 0;
        for (int k = size - 2; 0 <= k; --k) {
            if (nums[k] != 0)
                nums[k] = *min_element(&nums[k + 1], &nums[min(k + nums[k] + 1,size)]) + 1;
            else 
                nums[k] = 10e6;
        }
		return nums[0];
	}

	//649. Dota2 Senate
	string predictPartyVictory(string senate) {
		const size_t len = senate.length();
		queue<int> dire;
		queue<int> radiant;

		for (int k = 0; k < len; ++k) {
			if (senate[k] == 'D')
				dire.push(k);
			else
				radiant.push(k);
		}

		while (dire.size() && radiant.size()) {
			int d = dire.front(); dire.pop();
			int r = radiant.front(); radiant.pop();

			if (d < r) {
				dire.push(d + len);
			}
			else {
				radiant.push(r + len);
			}
		}

		return dire.size() ? "Dire" : "Radiant";
	}
	
	//206. Reverse Linked List
	ListNode* reverseList(ListNode* head) {
		if (head == nullptr)return head;

		ListNode* prev = nullptr;
		ListNode* next = head->next;

		while (true) {
			head->next = prev;
			if (next == nullptr)break;
			prev = head;
			head = next;
			next = next->next;
		}

		return head;
	}
	//77. Combinations

	//int combinations(size_t n, size_t k) {
	//	if (n <= k)return 1;
	//	double result = 1;
	//	size_t diff = k - n;
	//	
	//	if (diff > k) swap(k, diff);

	//	while (k <= n) {
	//		result *= k / diff--;
	//	}

	//	return result;
	//}

	vector<vector<int>> combine(int n, int k) {

		vector<vector<int>> result = {};

		vector<bool> bitmask(n - k, false);
		bitmask.resize(n, true);

		do {
			vector<int> subarray;
			for (int i = 0; i < n; ++i) {
				if (bitmask[i]) {
					subarray.push_back(i+1);
				}
			}
			result.push_back(subarray);

		} while (next_permutation(bitmask.begin(), bitmask.end()));

		return result;
	}
	//46. Permutations
	vector<vector<int>> permute(vector<int>& nums) {

		if (nums.size() == 1)return { {nums[0]} };
		if (nums.size() == 2)return { {nums[0],nums[1]},{nums[1],nums[0]}};

		vector<vector<int>> result = {};

		for (int k = 0; k < nums.size(); ++k) {
			vector<int> nums2(nums.begin(),nums.end());
			nums2.erase(nums2.begin()+k);
			for (auto& v : permute(nums)) {
				v.push_back(nums2[k]);
				result.push_back(v);
			}
		}

		return result;
	}
	//784. Letter Case Permutation

	vector<string> letterCasePermutation(string s) {
		
		vector<int> indexes(12,0);
		int len = 0;

		for (int k = 0; k < s.length(); ++k) {
			if (isalpha(s[k]))indexes[len++] = k;
		}

		vector<string> result;
		size_t bit = 1 << len;
		
		for (size_t k = 0; !(k & bit); ++k) {
			string l = s;
			for (int c = 0; c < len; c++) {
				if (k & (1 << c))l[indexes[c]] = isupper(l[indexes[c]]) ? tolower(l[indexes[c]]) : toupper(l[indexes[c]]);
			}
			result.push_back(l);
		}

		return result;
	}

	//1456. Maximum Number of Vowels in a Substring of Given Length
	int maxVowels(string s, int k) {
		static const vector<bool> isVowel = { 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0 };
		const size_t len = s.length();
		int l = 0, r = 0;
		int max = 0;
		
		while (r < k)
			if (isVowel[s[r++]-'a'])++max;

		int vowels = max;
		while (r < len) {
			if (isVowel[s[l++] - 'a'])vowels--;
			if (isVowel[s[r++] - 'a'])vowels++;
			if (vowels > max)max = vowels;
		}

		return max;
	}
	//74. Search a 2D Matrix
	inline int get(const vector<vector<int>>& matrix, int element) {
		return matrix[element / matrix[0].size()][element % matrix[0].size()];
	}

	bool searchMatrix(vector<vector<int>>& matrix, int target) {
		int l = 0, r = matrix.size() * matrix[0].size() - 1;

		while (l <= r) {
			int m = l + (r - l) / 2;
			int element = get(matrix, m);

			if (element == target)
				return true;
			if (element < target)
				l = m + 1;
			else
				r = m - 1;
		}

		return false;
	}

	//36. Valid Sudoku
	bool isValidSudoku(vector<vector<char>>& board) {
		char block[9][9];
		char row[9][9];
		char column[9][9];

		const size_t rows = board.size();
		const size_t columns = board[0].size();

		for (int r = 0; r < rows; ++r) {
			for (int c = 0; c < columns; ++c) {
				if (board[r][c] != '.') {
					int digit = board[r][c] - '1';
					if (++block[(r/3) * 3 + c / 3][digit] > 1) return false;
					if (++row[r][digit] > 1) return false;
					if (++column[c][digit] > 1) return false;
				}
			}
		}

		return true;
	}

	//918. Maximum Sum Circular Subarray
	int maxSubarraySumCircular(vector<int>& nums) {
		const size_t size = nums.size();
		
		if (size == 1)return nums[0];

		int max = nums[0] + nums[size-1];
		int current_max = max;
		for (int l = 0, r = size - 1; l < r;) {
		
			if (current_max > max)max = current_max;
		}

		return max;
	}
	//85. Maximal Rectangle

	//int isRectangle(const vector<vector<char>>& matrix,pair<size_t, size_t> vert,pair<size_t,size_t> horizontal) {
	//	
	//	int max = 0;

	//	while (vert.first <= vert.second) {
	//		size_t x = horizontal.first;
	//		while (x <= horizontal.second) {
	//			if (!matrix[vert.first][x])
	//			++x;
	//		}
	//		++vert.first;
	//		max += horizontal.second - horizontal.first + 1;
	//	}

	//	return max;
	//}

	int maximalRectangle(vector<vector<char>>& matrix) {
		//unions?
		//const size_t rows = 0;
		//return -1;
		
		//form a vector row-based in size,where each row described by it's sublines of ones(1)

		const size_t rows = matrix.size();
		const size_t columns = matrix[0].size();
		vector<vector<pair<int, int>>> pieces(rows);

		for (size_t r = 0; r < rows; ++r) {
			for (size_t c = 0; c < columns; ++c) {
				if (matrix[r][c]) {
					int from = c;
					while (matrix[r][c] && c < columns)++c;
					pieces[r].push_back({ from,c == columns ? c-1 : c});
				}
			}
		}

		//glue up pieces together,but how?

		


	}

	int numSubseq(vector<int>& nums, int target) {
		//count of subsequences = len of substring^2 - 1
		static const size_t modulo = 1000000007;
		const size_t size = nums.size();
		static vector<size_t> powers = {1,2};
		
		if (powers.size() < nums.size()) {
			int prev = powers.size();
			powers.resize(nums.size());
			for (int k = prev-1; k < size; ++k) {
				powers[k] = (powers[k - 1] * 2) % modulo;
			}
		}

		sort(nums.begin(), nums.end());

		size_t answer = 0;
		int l = 0;
		int r = nums.size() - 1;
		while (l <= r) {
			if (nums[l] + nums[r] <= target)
				answer = (answer + powers[r - l++]) % modulo;
			else 
				--r;
		}
		return answer;
	}

	//387. First Unique Character in a String
	int firstUniqChar(string s) {
		vector<pair<int, int>> letters(26);

		for (int k = 0; k < s.length(); ++k) {
			letters[s[k] - 'a'].first++;
			letters[s[k] - 'a'].second = k;
		}

		int index = s.length();
		for (int k = 0; k < 26; ++k) {
			if (letters[k].first == 1 && index > letters[k].second) {
				index = letters[k].second;
			}
		}

		return index == s.length() ? -1 : index;
	}

	int maxProduct(vector<int>& nums) {
		
		vector<int> nums2(nums.begin(), nums.end());

		int result = nums2[0];
		for (int k = 1; k < nums.size(); ++k) {
			if (nums[k - 1] != 0) {
				nums2[k] = max(nums2[k]*nums[k - 1],nums2[k]*nums[k]);
			}
			if (nums[k] > result)result = nums[k];
		}

		return result;
	}

	//141. Linked List Cycle
	bool hasCycle(ListNode* head) {
		if (head == nullptr)return false;
		ListNode* fast = head->next;

		while (head != fast) {
			if (fast == nullptr) return false;
			if (fast->next == nullptr) return false;
			fast = fast->next->next;
			head = head->next;
		}

		return true;
	}

	//203. Remove Linked List Elements
	ListNode* removeElements(ListNode* head, int val) {
		ListNode* iterator = head;
		ListNode* prev = nullptr;

		while (iterator != nullptr) {
			if (iterator->val == val) {
				if (prev == nullptr)
					iterator = head = iterator->next;
				else
					iterator = prev->next = iterator->next;
			}
			else {
				prev = iterator;
				iterator = iterator->next;
			}
		}

		return head;
	}

	//122. Best Time to Buy and Sell Stock II
	int maxProfit(vector<int>& prices) {
		int sum = 0;
		int start = 0;

		prices.push_back(0);

		for (int k = 1; k < prices.size(); k++) {
			if (prices[k - 1] > prices[k]) {
				sum += max(0, prices[k - 1] - prices[start]);
				start = k;
			}
		}
		return sum;
	}

	//1964. Find the Longest Valid Obstacle Course at Each Position
	vector<int> longestObstacleCourseAtEachPosition(vector<int>& obstacles) {

	}

	//1014. Best Sightseeing Pair
	int maxScoreSightseeingPair(vector<int> values) {
		
		int max = values[0];
		int wave = values[0] - 1;
		for (int k = 1; k < values.size(); ++k) {
			if (values[k] + wave > max)max = values[k] + wave;
			if (wave > values[k])wave = values[k];
			--wave;
		}

		return max;
	}

	//191. Number of 1 Bits
	int hammingWeight(uint32_t n) {
		int k = 0;
		while (n > 0) {
			k += n & 1;
			n = n >> 1;
		}
		return k;
	}

	inline int hammingWeight2(uint32_t n) {
		return (n & 1) + (n > 0 ? hammingWeight(n >> 1) : 0);
	}

	//231. Power of Two
	bool isPowerOfTwo(int n) {
		int k = 0;
		while (n > 0) {
			k += n & 1;
			n = n >> 1;
		}
		return k < 2;
	}

	//136. Single Number
	int singleNumber(vector<int>& nums) {
		return accumulate(nums.begin(), nums.end(), 0, [](int l, int r) {return l ^ r; });
	}

	int diagonalSum(vector<vector<int>>& mat) {
		const size_t size = mat.size();
		int sum = 0;
		for (int k = 0; k < size; ++k) {
			sum += mat[k][k] + mat[size - k - 1][k];
		}
		return sum - (size & 1 ? mat[size/2][size/2] : 0);
	}

	//83. Remove Duplicates from Sorted List
	ListNode* deleteDuplicates(ListNode* head) {

		ListNode* itr = head;

		if (head == nullptr)return nullptr;

		while (itr->next != nullptr) {
			if (itr->val == itr->next->val)itr->next = itr->next->next;
			else itr = itr->next;
		}

		return head;
	}

	//714. Best Time to Buy and Sell Stock with Transaction Fee
	int maxProfit2(vector<int> prices, int fee) {
		int sum = 0;
		int local_max = prices.back();
		int local_sum = 0;
		for (int k = prices.size() - 2; 0 <= k; --k) {
			if (local_max - prices[k] - fee < 1) {
				local_max = prices[k];
				sum += local_sum;
				local_sum = 0;
				continue;
			}
			local_sum = max(local_sum, local_max - prices[k] - fee);
		}

		return sum + local_sum;
	}

	//25. Reverse Nodes in k-Group
	ListNode* reverseKGroup(ListNode* head, int k) {
		
		if (head == nullptr || k < 2)return head;

		ListNode* left_part = nullptr;
		ListNode* part_begin = head;
		while (part_begin != nullptr) {
			int c = k;
			ListNode* part_end = part_begin;
			while (--c && part_end->next != nullptr) {
				part_end = part_end->next;
			}
			if (c > 0)break;
			ListNode* next_part = part_end->next;
			if (left_part != nullptr)left_part->next = nullptr;
			part_end->next = nullptr;

			reverseList(part_begin);

			if (left_part == nullptr)head = part_end;
			else left_part->next = part_end;
			part_begin->next = next_part;
			left_part = part_begin;
			part_begin = next_part;
		}

		return head;
	}

	//54. Spiral Matrix
	inline bool in_range(pair<size_t,size_t>& range, size_t x) { return range.first <= x && x <= range.second; }

	vector<int> spiralOrder(vector<vector<int>>& matrix) {
		static vector<pair<int, int>> directions = { {0,1},{1,0},{-1,0},{0,-1} };

		const size_t rows = matrix.size();
		const size_t columns = matrix[0].size();
		vector<int> result(rows*columns);

		size_t dir = 0;
		size_t x = 0, y = 0;
		pair<size_t, size_t> row_range = { 0,rows - 1 }, column_range = { 0,columns - 1 };
		for (int k = 0; k < result.size(); ++k) {

			result[k] = matrix[y][x];

			if (!in_range(row_range,x + directions[dir].second) || !in_range(column_range, y + directions[dir].first)) {
				if (dir == 0)row_range.first++;
				else if (dir == 1)column_range.second--;
				else if (dir == 2)row_range.second--;
				else column_range.first++;
				dir = (dir + 1) % 4;
			}
			y += directions[dir].first;
			x += directions[dir].second;
		}

		return result;
	}

	//59. Spiral Matrix II
	vector<vector<int>> generateMatrix(int n) {
		static vector<pair<int, int>> directions = { {0,1},{1,0},{0,-1},{-1,0} };

		const uint64_t N = n * n;
		vector<vector<int>> matrix(n, vector<int>(n));

		size_t dir = 0;
		size_t x = 0, y = 0;
		pair<size_t, size_t> row_range = { 0,n - 1 }, column_range = { 0,n - 1 };
		for (int k = 0; k < N; ++k) {

			matrix[y][x] = k + 1;

			if (!in_range(row_range, y + directions[dir].first) || !in_range(column_range, x + directions[dir].second)) {
				if (dir == 0)row_range.first++;
				else if (dir == 1)column_range.second--;
				else if (dir == 2)row_range.second--;
				else column_range.first++;
				dir = (dir + 1) % 4;
			}
			y += directions[dir].first;
			x += directions[dir].second;
		}

		return matrix;
	}

	//144. Binary Tree Preorder Traversal
	//94. Binary Tree Inorder Traversal
	void inorder(vector<int>& data, tree_node::TreeNode* current) {
		if (current == nullptr)return;
		inorder(data, current->left);
		data.push_back(current->val);
		inorder(data, current->right);
	}

	void preorder(vector<int>& data, tree_node::TreeNode* current) {
		if (current == nullptr)return;
		data.push_back(current->val);
		preorder(data, current->left);
		preorder(data, current->right);
	}

	vector<int> preorderTraversal(tree_node::TreeNode* root) {
		vector<int> result;
		preorder(result, root);
		return result;
	}

	//162. Find Peak Element
	inline int64_t getElement(const vector<int>& nums, size_t index) { return nums.size() <= index ? INT_FAST64_MIN : nums[index] ; }

	int findPeakElement(vector<int>& nums) {
		for (int k = 0; k < nums.size(); ++k) {
			if (nums[k] > nums[k + 1])return k;
		}
		return -1;
	}

	int search(vector<int>& nums, size_t left, size_t right) {
		size_t middle = (left + right) / 2;
		if (left == right)
			return middle + 1;
		if (nums[middle] < nums[middle + 1])
			return search(nums, middle, right);
		else
			return search(nums, left, middle);
	}

	int findPeakElement_dc(vector<int> nums) {
		return search(nums, 0, nums.size() - 1);
	}


	//153. Find Minimum in Rotated Sorted Array
	int findMin(vector<int>& nums) {
		int l = 0, r = nums.size() - 1;
		while (l != r) {
			int middle = (l + r) >> 1;
			if (nums[r] < nums[middle])l = middle + 1;
			else r = middle;
		}
		return nums[l];
	}

	//33. Search in Rotated Sorted Array
	int search(vector<int>& nums, int target) {
		int l = 0, r = nums.size() - 1;
		while (l != r) {
			int middle = (l + r) >> 1;
			if (nums[middle] == target)return middle;
			if (nums[r] < nums[middle] && target < nums[middle])l = middle + 1;
			else r = middle;
		}
		return -1;
	}

	//82. Remove Duplicates from Sorted List II
	ListNode* deleteDuplicates_2(ListNode* head) {

		ListNode* prev = nullptr;
		ListNode* itr = head;
		while (itr != nullptr && itr->next != nullptr) {
			
			if (itr->val == itr->next->val) {
				ListNode* last = itr->next;
				while (last != nullptr && last->val == itr->val) {
					last = last->next;
				}
				if (!prev) {
					itr = head = last;
				}
				else {
					itr = prev->next = last;
				}
			}
			else {
				prev = itr;
				itr = itr->next;
			}
		}

		return head;
	}

	//201. Bitwise AND of Numbers Range
	int rangeBitwiseAnd(int left, int right) {
		if (left == right)return left;

		int k = 0;
		while ((right >> k) > 0)++k;
		
		int power = 1 << --k;
		int result = 0;

		while ((right & power) == (left & power)) {
			result += right & power ? power : 0;
			power = power >> 1;
		}
		
		return result;
	}
	//102. Binary Tree Level Order Traversal
	void leveling(vector<vector<int>>& data, tree_node::TreeNode* current, int level = 0) {
		if (current == nullptr)return;

		leveling(data, current->left, level + 1);
		leveling(data, current->right, level + 1);

		if (data.size() <= level + 1)data.resize(level + 1);

		data[level].push_back(current->val);
	}

	vector<vector<int>> levelOrder(tree_node::TreeNode* root) {
		vector<vector<int>> result;
		leveling(result, root);
		return result;
	}
	//104. Maximum Depth of Binary Tree
	int maxDepth(tree_node::TreeNode* root, int level = 1) {
		return root == nullptr ? level - 1 : max(maxDepth(root->right, level + 1), maxDepth(root->left, level + 1));
	}

	//101. Symmetric Tree
	bool comparer(tree_node::TreeNode* left, tree_node::TreeNode* right) {
		if (bool(left) ^ bool(right))return false;
		if (left == nullptr)return true;
		if (left->val != right->val)return false;
		return comparer(left->left, right->left) && comparer(left->right, right->right);
	}

	bool isSymmetric(tree_node::TreeNode* root) {
		return comparer(root->left,root->right);
	}

	int maxUncrossedLines(vector<int>& nums1, vector<int>& nums2) {
		
		const size_t n1 = nums1.size();
		const size_t n2 = nums2.size();

		vector<int> *dp = new vector<int>(n2+1);
		vector<int> *dpPrev = new vector<int>(n2+1);

		for (int i = 0; i < n1; ++i) {
			for (int j = 0; j < n2; ++j) {
				if (nums1[i] == nums2[j]) {
					(*dp)[j + 1] = 1 + (*dpPrev)[j];
				}
				else {
					(*dp)[j + 1] = max((*dp)[j], (*dpPrev)[j + 1]);
				}
			}
			swap(dp, dpPrev);
		}

		return (*dp)[n2];
	}

	int nthUglyNumber(int n) {
		static vector<int> uglyNumbers = { 1 };
		static int i2 = 0, i3 = 0, i5 = 0;
		static int m2 = 2, m3 = 3, m5 = 5;
		if (n > uglyNumbers.size()) {

			int k = uglyNumbers.size() - 1;
			uglyNumbers.resize(n);
			while (++k < n) {
				uglyNumbers[k] = min(m2, min(m3, m5));
				if (uglyNumbers[k] == m2) {
					m2 = 2 * uglyNumbers[++i2];
				}
				if (uglyNumbers[k] == m3) {
					m3 = 3 * uglyNumbers[++i3];
				}
				if (uglyNumbers[k] == m5) {
					m5 = 5 * uglyNumbers[++i5];
				}
			}
		}

		return uglyNumbers[n - 1];
	}

	//2140. Solving Questions With Brainpower
	long long mostPoints(vector<vector<int>>& questions) {
		const size_t size = questions.size();
		vector<uint64_t> dp(size);

		dp[size - 1] = questions[size - 1][0];

		for (int k = questions.size() - 2; 0 <= k; --k) {
			int next = k + questions[k][1] + 1;
			dp[k] = max( uint64_t(next < size ? questions[next][0] : 0) + questions[k][0], dp[k + 1]);
		}

		return dp[0];
	}

	//226. Invert Binary Tree
	tree_node::TreeNode* invertTree(tree_node::TreeNode* root) {
		if (root != nullptr) {
			tree_node::TreeNode* l = root->left;
			root->left = invertTree(root->right);
			root->right = invertTree(l);
		}
		return root;
	}

	//112. Path Sum
	bool helper(tree_node::TreeNode* root, int target, int current) {
		if (root == nullptr)return false;
		current += root->val;
		if (target == current && root->left == root->right)return true;
		return helper(root->left, target, current) || helper(root->right, target, current);
	}

	bool hasPathSum(tree_node::TreeNode* root, int targetSum) {
		return helper(root, targetSum, 0);
	}

	//844. Backspace String Compare
	string format(string x) {
		string result = "";
		int to_skip = 0;
		int k = x.length();
		
		while (0 <= --k) {
			if (x[k] == '#') ++to_skip;
			else if (to_skip < 1)result += x[k];
			else --to_skip;
		}
		return result;
	}

	bool backspaceCompare(string s, string t) {
		return format(s) == format(t);
	}

	//986. Interval List Intersections
	vector<vector<int>> intervalIntersection(vector<vector<int>>& firstList, vector<vector<int>>& secondList) {
		vector<vector<int>> result;

		auto f = firstList.begin(), s = secondList.begin();
		while (f != firstList.end() && s != secondList.end()) {
			
		}

		return result;
	}

	//700. Search in a Binary Search Tree
	tree_node::TreeNode* searchBST(tree_node::TreeNode* root, int val) {
		while (root != nullptr) {
			if (root->val == val)break;
			root = root->val > val ? root->left : root->right;
		}
		return root;
	}

	//78. Subsets
	vector<vector<int>> subsets(vector<int> nums) {
		const size_t size = nums.size();
		int max = 1 << size;
		vector<vector<int>> result(max);
		int i = -1;
		while (++i < max) {
			vector<int> subset;

			int b = -1;
			while (++b < size) {
				if (i & (1 << b))subset.push_back(nums[b]);
			}

			result[i] = subset;
		}
		return result;
	}

	//75. Sort Colors
	void sortColors(vector<int> nums) {
		int l = -1, r = nums.size();
		while (++l < --r) {
			if (nums[l] > nums[r])
				swap(nums[l], nums[r]);
		}
	}

	//48. Rotate Image
	void rotate(vector<vector<int>>& matrix) {
		const size_t N = matrix.size();
		size_t n = N;
		size_t d = N / 2;
		for (int k = 0; k < d; ++k) {
			for (int i = 1; i < n; ++i) {
				swap(matrix[k][k + i],matrix[k + i][N - k - 1]);
				swap(matrix[k][k + i],matrix[N - k - 1 - i][k]);
				swap(matrix[N - k - 1 - i][k], matrix[N - k - 1][N - k - 1 - i]);
			}

			n -= 2;
		}
	}

	//43. Multiply Strings

	string multiply(string num1, string num2) {
		const size_t len1 = num1.length();
		const size_t len2 = num2.length();
		const size_t len3 = len1 + len2;
		vector<short> nums(len3);
		string result(len3, '0');

		int s2 = -1;
		while (++s2 < len2) {
			int s1 = -1;
			while (++s1 < len1) {
				nums[len3 - (s1 + s2 + 1)] += (num2[len2 - s2 - 1] - '0') * (num1[len1 - s1 - 1] - '0');
			}
		}

		int s3 = len3;
		while (0 <= --s3) {
			if (nums[s3] > 9) {
				nums[s3 - 1] += (nums[s3]) / 10;
			}
			result[s3] = (nums[s3]) % 10 + '0';
		}

		return result[0] == '0' ? result.substr(1, len3 - 1) : result;
	}

	//179. Largest Number
	string largestNumber(vector<int> nums) {
		const size_t size = nums.size();
		vector<string> as_string(size);

		for (int k = 0; k < size; ++k) {
			as_string[k] = to_string(nums[k]);
		}

		sort(as_string.begin(), as_string.end(), [](const string& l, const string& r) {return (l + r) > (r + l); });

		if (as_string[0][0] == '0')return "0";

		string result = "";
		for (auto& str : as_string) {
			result += str;
		}

		return result;
	}

	//37. Sudoku Solver
	void solveSudoku(vector<vector<char>>& board) {

	}

	//2466. Count Ways To Build Good Strings
	int countGoodStrings(int low, int high, int zero, int one) {
		static const size_t modulo = 1000000007;
		vector<size_t> variations(high);

		++variations[zero - 1];
		++variations[one - 1];

		size_t sum = 0;
		for (int k = 0; k < high; ++k) {
			if (0 <= k - zero)variations[k] += variations[k - zero];
			if (0 <= k - one)variations[k] += variations[k - one];
			variations[k] %= modulo;
			if (low <= k + 1)sum = (sum + variations[k]) % modulo;
		}

		return sum;
	}

	//
	vector<int> findAnagrams(string s, string p) {

	}

	//1799. Maximize Score After N Operations
	int maxScore(vector<int> nums) {
		const size_t N = nums.size();

		list<int> indexes;
		for (int k = 0; k < N; ++k)indexes.push_back(k);

		vector<int> relations(N, -1);
		while (indexes.size()) {
			if (relations[indexes.front()] != -1) {
				indexes.pop_front();
				continue;
			}
			int current = indexes.front();
			int best = relations[current];
			for (int k = 0; k < N; ++k) {
				if (current == k)continue;
				int nested = relations[k] == -1 ? 0 : gcd(nums[relations[k]], nums[relations[relations[k]]]);
				if (best == -1 && relations[k] == -1)best = k;
				if (nested < gcd(nums[k],nums[current])) {
					best = best == -1 ? k : ( gcd(nums[best],nums[current]) < gcd(nums[k],nums[current]) ? k : best);
				}
			}
			if (relations[best] != -1) {
				indexes.push_back(relations[relations[best]]);
				relations[relations[best]] = -1;
			}
			if (relations[current] != -1) {
				indexes.push_back(relations[relations[current]]);
				relations[relations[current]] = -1;
			}
			relations[current] = best;
			relations[best] = current;
		}

		vector<int> gcdPairs;
		vector<bool> used(N, false);
		for (int k = 0; k < N; ++k) {
			if (used[k])continue;
			int i1 = relations[k];
			int i2 = relations[i1];
			gcdPairs.push_back(gcd(nums[i1], nums[i2]));
			used[i1] = used[i2] = true;
		}

		sort(gcdPairs.begin(), gcdPairs.end());

		int amount = 0;
		int mult = 0;
		for (int v : gcdPairs)amount += v * ++mult;

		return amount;
	}

	//200. Number of Islands
	void dfs(vector<vector<char>>& grid, pair<size_t, size_t> pos) {
		static const vector<pair<int, int>> directions = { {1,0},{0,1},{-1,0},{0,-1} };

		grid[pos.first][pos.second] = '0';

		for (auto&p : directions) {
			size_t y = p.first + pos.first;
			size_t x = p.second + pos.second;
			if (y < grid.size() && x < grid[0].size() && grid[y][x] == '1') {
				dfs(grid, { y,x });
			}
		}
	}

	int numIslands(vector<vector<char>>& grid) {
		int count = 0;
		for (int r = 0; r < grid.size(); ++r) {
			for (int c = 0; c < grid[0].size(); ++c) {
				if (grid[r][c] == '1') {
					count++;
					dfs(grid, { r,c });
				}
			}
		}
	}

	//547. Number of Provinces
	int findCircleNum(vector<vector<int>> isConnected) {
		const size_t N = isConnected.size();
		Unions unions(N);

		for (int k = 0; k < N-1;++k) {
			for (int j = k+1; j < N; ++j) {
				if (isConnected[k][j]) {
					unions.insert(k-1, j);
				}
			}
		}
		auto [g, s] = unions.groups_amount();

		return g + s;
	}

	//
	ListNode* swapNodes(ListNode* head, int k) {
		
		ListNode* left = head;
		while(k-- > 0) left = left->next;
		
		ListNode* temp = left;
		ListNode* right = head;
		while (temp->next != nullptr) {
			right = right->next;
			temp = temp->next;
		}
		swap(right->val, left->val);
		return head;
	}

	//572. Subtree of Another Tree
	bool comparator(tree_node::TreeNode* root, tree_node::TreeNode* subRoot) {
		if (root == subRoot)return true;
		if (!root || !subRoot)return false;
		if (root->val != subRoot->val)return false;
		return comparator(root->left, subRoot->left) && comparator(root->right, subRoot->right);
	}
	bool isSubtree(tree_node::TreeNode* root, tree_node::TreeNode* subRoot) {
		if (!root)return false;
		return comparator(root, subRoot) || isSubtree(root->left, subRoot) || isSubtree(root->right, subRoot);
	}

	//117. Populating Next Right Pointers in Each Node II
	tree_node2::Node* connect(tree_node2::Node* root) {

	}

	//797. All Paths From Source to Target
	void tracer(const vector<vector<int>>& graph, vector<vector<int>>& result, vector<int>& visited, const int edge) {
		
		visited.push_back(edge);

		if (edge == graph.size() - 1) {
			result.push_back(visited);
		}
		else {
			for (int e : graph[edge]) tracer(graph, result, visited, e);
		}

		visited.pop_back();
	}

	vector<vector<int>> allPathsSourceTarget(vector<vector<int>> graph) {
		vector<vector<int>> result;
		vector<int> visited;
		visited.reserve(graph.size());
		tracer(graph, result, visited, 0);
		return result;
	}

	//49. Group Anagrams
	size_t count(const string& str) {
		const static vector<size_t> nums = { 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113 };
		size_t result = 1;
		for (char c : str) {
			result *= nums[c - 'a'];
		}
		return result;
	}

	vector<vector<string>> groupAnagrams(vector<string>& strs) {
		const size_t size = strs.size();
		vector<pair<size_t, int>> as_int(size);
		for (int k = 0; k < size; ++k) {
			as_int[k] = { count(strs[k]),k };
		}

		sort(as_int.begin(), as_int.end(),
			[](const pair<size_t, int>& l, const pair<size_t, int>& r)
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

	//2130. Maximum Twin Sum of a Linked List
	int pairSum(ListNode* head) {
		ListNode* iter = head, * iterF = head->next;
		
		while (iterF->next) {
			iter = iter->next;
			iterF = iterF->next->next;
		}
		
		iterF = reverseList(iter->next);
		iter = head;

		int result = 0;
		while (iter) {
			result = max(result, iter->val + iterF->val);
			iter = iter->next;
			iterF = iterF->next;
		}

		return result;
	}

	int pairSum2(ListNode* head) {
		vector<int> nums;
		ListNode* iter = head->next;

		while (true) {
			nums.push_back(head->val);
			if (iter->next == nullptr)break;
			head = head->next;
			iter = iter->next->next;
		}

		iter = head->next;
		int r = nums.size();
		int result = 0;
		while (--r) {
			int current = nums[r] + iter->val;
			if (current > result)result = current;
			iter = iter->next;
		}
		return result;
	}

	//90. Subsets II

	vector<vector<int>> subsetsWithDup(vector<int>& nums) {
		unordered_map<int, bool> used;
		const size_t size = nums.size();
		int max = 1 << size;
		vector<vector<int>> result = { {} };
		result.reserve(max);
		int i = 0;
		while (++i < max) {
			vector<int> subset;
			uint64_t finalSum = 0;

			int b = -1;
			while (++b < size) {
				if (i & (1 << b)) {
					subset.push_back(nums[b]);
					finalSum += pow(nums[b] + 22, 4);
				}
			}

			if (!used[finalSum]) {
				result.push_back(subset);
			}
			used[finalSum] = true;
		}
		return result;
	}

	//130. Surrounded Regions
	void mark(vector<vector<char>>& board, pair<size_t, size_t> pos) {
		static const vector<pair<int, int>> directions = { {1,0},{0,1},{-1,0},{0,-1} };
		if (board[pos.first][pos.second] == 'O')return;

		board[pos.first][pos.second] = 'F';

		for (auto& p : directions) {
			size_t y = pos.first + p.first;
			size_t x = pos.second + p.second;
			if (y < board.size() && x < board[0].size()) {
				mark(board, { y,x });
			}
		}
	}

	void solve(vector<vector<char>>& board) {
		const size_t R = board.size();
		const size_t C = board[0].size();

		for (int k = 0; k < C; ++k) {
			mark(board, { 0,k });
			mark(board, { R - 1,k });
		}
		for (int k = 0; k < R; ++k) {
			mark(board, { k,0 });
			mark(board, { k,C - 1 });
		}

		for (int r = 0; r < R; ++r) {
			for (int c = 0; c < C; ++c) {
				if (board[r][c] == 'O')board[r][c] = 'X';
				else if (board[r][c] == 'F')board[r][c] = 'O';
			}
		}
	}

	//1557. Minimum Number of Vertices to Reach All Nodes
	vector<int> findSmallestSetOfVertices(int n, vector<vector<int>>& edges) {
		vector<bool> common(n);
		size_t size = 0;
		for (auto& v : edges) {
			if (!common[v[1]]) {
				common[v[1]] = ++size;
			}
		}

		vector<int> result(n-size);
		for (int k = 0,i = 0; k < n; ++k) {
			if (!common[k])result[i++] = k;
		}
		return result;
	}

	//1318. Minimum Flips to Make a OR b Equal to c
	int minFlips(int a, int b, int c) {
		int pos = 1 << 30;
		int count = 0;
		while (0 < pos) {
			bool bitA = a & pos;
			bool bitB = b & pos;

			if (c & pos) count += !bitA && !bitB;
			else count += bitA + bitB;

			pos = pos >> 1;
		}
		return count;
	}

	//345. Reverse Vowels of a String
	bool is_vowel(char c) {
		static const vector<bool> isVowel = { 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0 };
		if ('a' <= c && c <= 'z')return isVowel[c - 'a'];
		if ('A' <= c && c <= 'Z')return isVowel[c - 'A'];
		return false;
	}
	string reverseVowels(string s) {
		int l = 0, r = s.length() - 1;
		while (l < r) {
			while (!is_vowel(s[l]))++l;
			while (!is_vowel(s[r]))--r;
			swap(s[l++], s[r--]);
		}
		return s;
	}

	//151. Reverse Words in a String
	string reverseWords(string s) {
		string result = "";

		return result;
	}

	//785. Is Graph Bipartite?
	char find(const vector<char>& groups, int pos) {
		if (groups[pos] == -1)return -1;

		int index = groups[pos];

		while (index != groups[index])
			index = groups[index];

		return groups[index];
	}

	bool isBipartite(vector<vector<int>>& graph) {
		const size_t N = graph.size();
		vector<char> groups(N, -1);
		
		for (char k = 0; k < N;k++) {

			if (graph[k].size() < 1)continue;
			
			char group1 = find(groups, k);
			char group2 = -1;
			for (int node : graph[k]) {
				if (groups[node] > group2) {
					group2 = groups[node];
					break;
				}
			}
			if (group1 == -1)group1 = groups[k] = k;
			if (group2 != -1)group2 = find(groups, group2);
			else group2 = graph[k][0];

			if (group1 == group2) return false;
			for (char node : graph[k])groups[node] = group2;
		}

		return true;
	}

	//399. Evaluate Division
	double find(const vector<vector<double>>& coofs, int from,int to) {
		
		static unordered_map<int, bool> visited;

		if (from == to)return 1;

		visited[from] = true;

		double result = 0;
		for (int k = 0; k < coofs.size(); ++k) {
			if (coofs[from][k] != -1 && !visited[k]) {
				result = max(result, find(coofs, k, to) * coofs[from][k]);
			}
		}

		visited[from] = false;

		return result;
	}

	vector<double> calcEquation(vector<vector<string>> equations, vector<double> values, vector<vector<string>> queries) {

		const size_t size = equations.size();
		unordered_map<string, short> vars;
		for (size_t k = 0; k < size; ++k) {
			if (!vars[equations[k][0]]) vars[equations[k][0]] = vars.size();
			if (!vars[equations[k][1]]) vars[equations[k][1]] = vars.size();
		}
		const size_t sizeV = vars.size();
		vector<vector<double>> coofs(sizeV, vector<double>(sizeV, -1));
		Unions unions(sizeV);

		for (size_t k = 0; k < size; k++) {
			size_t r = vars[equations[k][0]] - 1;
			size_t c = vars[equations[k][1]] - 1;
			coofs[r][c] = values[k];
			coofs[c][r] = 1 / values[k];
			unions.insert(r, c);
		}

		int k = 0;
		vector<double> result(queries.size());
		for (auto& query : queries) {
			int from = vars[query[0]] - 1;
			int to = vars[query[1]] - 1;
			if (from < 0 || to < 0)result[k] = -1;
			else if (unions.find(from) != unions.find(to))result[k] = -1;
			else result[k] = find(coofs, from, to);
			++k;
		}

		return result;
	}
};


class MyQueue {
	stack<int> reversed;
	stack<int> current;
public:
	MyQueue() {
		reversed = current;
	}

	void push(int x) {
		current.push(x);
	}

	int pop() {
		int result = this->peek();
		reversed.pop();
		return result;
	}

	int peek() {
		if (reversed.empty()) {
			while (!current.empty()) {
				reversed.push(current.top());
				current.pop();
			}
		}
		return reversed.top();
	}

	inline bool empty() {
		return reversed.empty() && current.empty();
	}
};