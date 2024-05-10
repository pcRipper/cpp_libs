#pragma once
#include "../../GeneralLibs/Headers/includes.hpp"

#pragma region to_string  

string to_string(string x);
string to_string(int* x);
string to_string(char x);

template <class Key, class Value>
inline string to_string(pair<Key, Value> pair);

template <class T, size_t S>
string to_string(const array<T, S>& Array) {
	static const string EMPTY_ARRAY = "{}";
	if (S == 0)return EMPTY_ARRAY;

	string result = "{";
	for (size_t k = 1; k < S; ++k) {
		result += to_string(Array[k - 1]) + ", ";
	}
	result += to_string(Array.back()) + "}";

	return result;
}

template <class T>
inline string to_string(const vector<T>& vector) {
	static const string EMPTY_ARRAY = "{}";
	const size_t S = vector.size();
	if (S == 0)return EMPTY_ARRAY;

	string result = "{";
	for (size_t k = 1; k < S; ++k) {
		result += to_string(vector[k - 1]) + ", ";
	}
	result += to_string(vector.back()) + "}";

	return result;
}

template <class T>
string to_string(vector<T>* vector) {
	return to_string(*vector);
}

template <class Key, class Value>
inline string to_string(pair<Key, Value> pair) {
	return "(" + to_string(pair.first) + ", " + to_string(pair.second) + ")";
}

#pragma endregion

#pragma region algorithms

template <class T>
void mapIn(vector<T>& data, function<void(T&,size_t)> modifier) {
	for (size_t i = 0; i < data.size(); ++i) {
		modifier(data[i],i);
	}
}

template <class T,class F>
vector<F> mapOn(const vector<T>& data, function<F(const T&,size_t)> modifier) {
	vector<F> result = {};
	result.reserve(data.size());
	for (size_t i = 0; i < data.size(); ++i) {
		result.push_back(modifier(data[i],i));
	}
	return result;
}
namespace mappers {
	pair<int, size_t> indexer1(const int& x, size_t index);
	pair<size_t, int> indexer2(const int& x, size_t index);
}

vector<int> randomArray(const size_t SIZE, int from, int to);

#pragma endregion

string to_bit(int x);
string to_bit(size_t x);
string to_bit(uint16_t x);

template <class T>
void show_vector(const vector<T>& vector, int FLAGS = 0b111) {
	const size_t N = vector.size();
	size_t k = 0;
	if (FLAGS & 0b1)cout << N << " : ";
	cout << "{" << ( FLAGS & 0b100 ? "" : "\n  ");
	for (const T& element : vector) {
		cout << to_string(element) << (++k == N ? (FLAGS & 0b100 ? "}" : "\n}") : (FLAGS & 0b100 ? ", " : "\n  "));
	}
	if (FLAGS & 0b10)cout << "\n";
}

template <class T>
void show_array(const T* array, size_t size,int FLAGS = 0b111)
{
	vector<T> as_vector = vector<T>(&array[0], &array[size]);
	show_vector(as_vector);
}

template <class T>
void show_deque(const deque<T>& deque,int FLAGS = 0b111) {
	vector<T> vector(deque.begin(), deque.end());
	show_vector(vector, FLAGS);
}

template <class T>
void show_stack(const stack<T> stack, int FLAGS = 0b111) {
	show_deque(stack._Get_container());
}

template <class T,class Container,class Priority>
void show_priority_queue(priority_queue<T,Container, Priority> queue, int FLAGS = 0b111) {
	vector<T> data(queue.size());
	int i = 0;
	while (not queue.empty()) {
		data[i++] = queue.top();
		queue.pop();
	}
	show_vector<T>(data, FLAGS);
}


template <class Key, class Value, class Hasher>
void show_unordered_map(const unordered_map<Key, Value, Hasher>& map, int FLAGS = 0b111) {
	vector<pair<Key, Value>> data(map.begin(), map.end());
	show_vector(data, FLAGS);
}

template <class Key,class Value,class Predicate>
void show_map(const map<Key, Value, Predicate>& map,int FLAGS = 0b111) {
	vector<pair<Key, Value>> data(map.begin(), map.end());
	show_vector(data, FLAGS);
}

//0x1 - add size
//0x10 - add new line
template <class T,class Predicate>
void show_set(const set<T,Predicate> & set, int FLAGS = 0x11) {
	vector<T> data(set.begin(), set.end());
	show_vector(data, FLAGS);
}

template <class T>
void show_multiset(const multiset<T>& set, int FLAGS = 0x11) {
	vector<T> data(set.begin(), set.end());
	show_vector(data, FLAGS);
}

template <class T>
class vproxy {
private:
	std::vector<T>& v1, v2;
public:
	vproxy(std::vector<T>& ref1, std::vector<T>& ref2) : v1(ref1), v2(ref2) {}
	const T& operator[](const size_t& i) const;
	const size_t size() const;
};

template <class T>
const T& vproxy<T>::operator[](const size_t& i) const {
	return (i < v1.size()) ? v1[i] : v2[i - v1.size()];
};

template <class T>
const size_t vproxy<T>::size() const { return v1.size() + v2.size(); };

namespace node_1 {
	class Node {
	public:
		int val;
		vector<Node*> neighbors;
		Node() {
			val = 0;
			neighbors = vector<Node*>();
		}
		Node(int _val) {
			val = _val;
			neighbors = vector<Node*>();
		}
		Node(int _val, vector<Node*> _neighbors) {
			val = _val;
			neighbors = _neighbors;
		}
	};
}

namespace tree_node {
	struct TreeNode {
		int val;
		TreeNode* left;
		TreeNode* right;
		TreeNode() : val(0), left(nullptr), right(nullptr) {}
		TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
		TreeNode(int x, TreeNode* left, TreeNode* right) : val(x), left(left), right(right) {}
	};
}

struct ListNode
{
	int val;
	ListNode* next;
	ListNode() : val(0), next(nullptr) {}
	ListNode(int x) : val(x), next(nullptr) {}
	ListNode(int x, ListNode* next) : val(x), next(next) {}

	void show();
	template <class T> static ListNode* as_list(vector<T> data);
	static vector<int> as_vector(ListNode* head);
};

template <class T>
ListNode* ListNode::as_list(vector<T> data)
{
	if (data.size() == 0)return nullptr;

	ListNode* head = new ListNode(data[0]);
	ListNode* iter = head;
	const size_t size = data.size();

	for (int k = 1; k < size; k++) {
		iter = iter->next = new ListNode(data[k]);
	}

	iter->next = nullptr;

	return head;
}
namespace tree_node2 {
	class Node {
	public:
		int val;
		Node* left;
		Node* right;
		Node* next;

		Node() : val(0), left(NULL), right(NULL), next(NULL) {}

		Node(int _val) : val(_val), left(NULL), right(NULL), next(NULL) {}

		Node(int _val, Node* _left, Node* _right, Node* _next)
			: val(_val), left(_left), right(_right), next(_next) {}
	};
}

class Groups {
	vector<int> edges;
	size_t marked;
	vector<int> groups;
	size_t distinct_groups;
public:
	Groups(int n) {
		edges = vector<int>(n, 0);
		marked = 0;
		groups.reserve(n / 2);
		distinct_groups = 0;
	}
	bool insert(int l, int r) {
		if (bool(edges[l]) ^ bool(edges[r])) {
			edges[l] = edges[r] = max(edges[l], edges[r]);
			marked++;
		}
		else if (edges[r] == 0) {
			groups.push_back(groups.size() + 1);
			edges[l] = edges[r] = groups.back();
			marked += 2;
			distinct_groups++;
		}
		else if (edges[l] != edges[r]) {
			int to_find = edges[r];
			for (size_t k = 0; k < edges.size(); k++) {
				if (edges[k] == to_find)edges[k] = edges[l];
			}
			distinct_groups--;
		}
		else {
			return false;
		}
		return true;
	}
	//key - distinct (non zero) groups, value - out of group members
	inline pair<size_t, size_t> groups_amount() { return { distinct_groups, edges.size() - marked }; }

	inline int operator[](size_t index) {
		return (index < edges.size()) ? edges[index] : -1;
	}
};

class Unions {
	vector<int> nodes;
	size_t marked = 0;
	vector<int> groups;
	size_t distinct_groups = 0;
public:
	Unions(int n) {
		nodes = vector<int>(n, 0);
		groups.reserve((n / 2) + 1);
		groups.push_back(0);
	}

	int find(int x) {
		int index = nodes[x];

		while (index != groups[index])
			index = groups[index];

		return groups[index];
	}

	bool insert(int l, int r) {
		int g1 = find(l);
		int g2 = find(r);

		if (bool(g1) ^ bool(g2)) {
			nodes[l] = nodes[r] = max(nodes[l], nodes[r]);
			++marked;
		}
		else if (g1 == 0) {
			groups.push_back(groups.size());
			nodes[l] = nodes[r] = groups.back();
			marked += 2;
			++distinct_groups;
		}
		else if (g1 != g2) {
			groups[g2] = groups[g1];
			--distinct_groups;
		}
		else return false;
		return true;
	}

	inline pair<size_t, size_t> groups_amount() { return { distinct_groups, nodes.size() - marked }; }
};

struct CountingNode {
	int val;
	uint16_t times;
	uint16_t on_the_left;
	CountingNode* left;
	CountingNode* right;

	CountingNode(int val) :
		val(val),
		times(1),
		on_the_left(0),
		left(nullptr),
		right(nullptr)
	{};

	~CountingNode() {
		delete left;
		delete right;
	}
};

class FenwickTree {
public:
	FenwickTree(size_t n) : tree(n + 1, 0) {}

	void update(int idx) {
		while (idx < tree.size()) {
			++tree[idx];
			idx += (idx & -idx);
		}
	}

	int query(int idx) {
		int result = 0;
		while (idx > 0) {
			result += tree[idx];
			idx -= (idx & -idx);
		}
		return result;
	}

private:
	vector<int> tree;
};

//
namespace node_138 {
	class Node {
	public:
		int val;
		Node* next;
		Node* random;

		Node(int _val) {
			val = _val;
			next = NULL;
			random = NULL;
		}
	};
}

struct TrieNode {
public:
	TrieNode* letters[26];
	bool is_complete_word;

	TrieNode() {
		memset(&letters[0], 0, sizeof(TrieNode*) * 26);
		is_complete_word = false;
	}

	void insert(const string& word, char index) {
		if (index == word.length()) {
			is_complete_word = true;
			return;
		}

		if (letters[word[index] - 'a'] == nullptr) {
			letters[word[index] - 'a'] = new TrieNode();
		}

		letters[word[index] - 'a']->insert(word, index + 1);
	}

	bool find(const string& word, char index) {
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

namespace QuadTree{

}