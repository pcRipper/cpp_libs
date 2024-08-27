#include <general_libs/Functional.hpp>

size_t Functions::gcd(size_t a, size_t b)
{
	while (a * b != 0) {
		if (a > b)a %= b;
		else b %= a;
	}
	
	return a + b;
}

size_t Functions::lcm(size_t a, size_t b) {
	return a * b / gcd(a, b);
}

string Functions::split(string& text, size_t& pos, char symbol, bool shift) {
	string result = "";
	for (size_t length = text.length(); pos < length; pos++) {
		if (text[pos] == symbol)break;
		result += text[pos];
	}
	pos += shift;
	return(result);
}

pair<vector<int>*, vector<vector<int>>*> Functions::topological_sort(int n, vector<vector<int>> const& directedGraph){
	
	vector<vector<int>>* connections = new vector<vector<int>>(n);
	vector<int> roots(n, -1);

	for(const auto& edge : directedGraph) {
		if (edge[0] == edge[1]) return {nullptr, nullptr};
		roots[edge[1]] = max(0, roots[edge[1]]);
		roots[edge[1]] += 1;
		roots[edge[0]] = max(0, roots[edge[0]]);
		(*connections)[edge[0]].push_back(edge[1]);
	}

	deque<int> queue;
	int used = 0;
	for (int i = 0; i < n; ++i) {
		if (roots[i] == 0) {
			queue.push_back(i);
		}
		used += roots[i] != -1;
	}

	vector<int>* result = new vector<int>();
	result->reserve(used);

	while (!queue.empty()) {
		used -= 1;
		int top = queue.front(); queue.pop_front();
		result->push_back(top);

		for (int next : (*connections)[top]) {
			if (--roots[next] == 0) {
				queue.push_back(next);
			}
		}
	}

	if (used != 0) {
		delete result;
		delete connections;
		return {nullptr, nullptr};
	}

	return {result, connections};
}

size_t Functions::cantor_pairing(size_t x, size_t y){
    if(++x > ++y)swap(x, y);
    return (x * (x + 3 + 2 * y) + y * (y + 1)) / 2;
}
