#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <list>

using namespace std;

template <class T>
void show_vector(const vector<T>& vector) {
	for (T element : vector) {
		cout << element << endl;
	}
}

vector<string> get_pins(string observed);
int twoLastDigits(int a, int b);
int last_digit(list<int> array);


size_t gcd(size_t a, size_t b);
size_t lcm(size_t a, size_t b);
vector<int> getDivisors(size_t num);
long properFractions(long n);

