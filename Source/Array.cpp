#pragma once
#include "../Headers/Array.h"

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