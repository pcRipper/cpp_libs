#pragma once
#include "includes.h"

template <class T>
void show_vector(const vector<T>& vector,string start, string separator, string end) {
	if (vector.size() < 1){
		cout << "empty vector\n";
		return;
	}

	cout << start;
	for (T element : vector) {
		cout << element << separator;
	}
	cout << end;
}

template <class T>
void show_vector_commas(const vector<T>& vector) {
	show_vector(vector,to_string(vector.size()) + " : ",", ","\b\b;\n");
}

template <class T>
void show_vector_lines(const vector<T>& vector) {
	show_vector(vector, vector.size() + " :\n", "\n\t", "\r;\n");
}

template<class T>
void showArray(T* array,size_t size,string start, string separator, string end) {
	if(array != nullptr){

		if(size < 1){
			cout << "empty array\n";
			return;
		}

		cout << start;
		for (int k = 0; k < size; k++) {
			cout << array[k] << separator;
		}
		cout << end;

	}
}

size_t gcd(size_t a, size_t b);
size_t lcm(size_t a, size_t b);

string split(string& text, size_t& pos, char symbol, bool shift = 1);
