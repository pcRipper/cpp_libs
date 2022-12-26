#pragma once
#include "Array.h"

using namespace std;

string to_string(string text) { return text; }

Range::Range(int from, int to, int step) :
	from(from),
	to(to),
	step(step) {};

Range::Range() {
	cout << "Input range :\n";
	do {
		cout << "From = ";
		cin >> from;
		cout << "To = ";
		cin >> to;
		cout << "With step = ";
		cin >> step;
	} while (((to - from) / step) < 0);
};

template <class T>
void SWAP(T& left, T& right) {
	T time = T(left);
	left = *new T(right);
	right = *new T(time);
}

string split(string& text, size_t& pos, char symbol, bool shift) {
	string result = "";
	for (size_t length = text.length(); pos < length; pos++) {
		if (text[pos] == symbol)break;
		result += text[pos];
	}
	pos += shift;
	return(result);
}

template <class T>
void showArray(T* array, size_t size, string separator, string end)
{
	if (array != nullptr) {
		cout << size << " : ";
		for (size_t k = 0; k < size; k++) {
			cout << array[k] << separator;
		}
		cout << end;
	}
}