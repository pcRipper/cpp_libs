#include "../Headers/DataStructures.hpp"

#pragma region algorithms

pair<int, size_t> mappers::indexer1(const int& x, size_t index) {
	return make_pair(x, index);
}

pair<size_t, int> mappers::indexer2(const int& x, size_t index) {
	return make_pair(index, x);
}


vector<int> randomArray(const size_t SIZE, int from, int to) {
	srand(time(0));
	vector<int> result(SIZE, 0);

	int range = to - from + 1;

	for (size_t i = 0; i < SIZE; ++i) {
		result[i] = rand() % range + from;
	}

	return result;
}

#pragma endregion

std::string to_bit(int x) {
	std::string result = "";
	for (int k = 31; k >= 0; --k) {
		result += (x & (1 << k)) ? '1' : '0';
	}
	return result;
}

std::string to_bit(size_t x)
{
	std::string result = "";
	for (int k = 31; k >= 0; --k) {
		result += (x & (1 << k)) ? '1' : '0';
	}
	return result;
}

std::string to_bit(uint16_t x)
{
	std::string result = "";
	for (int k = 15; k >= 0; --k) {
		result += (x & (1 << k)) ? '1' : '0';
	}
	return result;
}

void ListNode::show()
{
	std::cout << val;
	if (next != nullptr) {
		std::cout << "->";
		next->show();
	}
	else {
		std::cout << "\n";
	}
}

vector<int> ListNode::as_vector(ListNode* head)
{
	vector<int> result = {};

	while (head) {
		result.push_back(head->val);
		head = head->next;
	}

	return result;
}

#pragma region to_string  

inline string to_string(string x) {
	return x;
}

inline string to_string(int* x) {
	return (x != nullptr) ? to_string(*x) : "?";
}

string to_string(char x)
{
	return string(1,x);
}

string to_string(tree_node::TreeNode* node) {
	return to_string(node->val);
}

#pragma endregion