#pragma once
#include "includes.h"

template <class T>
void show_vector(const std::vector<T>& vector,std::string start, std::string separator, std::string end) {
	if (vector.size() < 1){
		std::cout << "empty std::vector\n";
		return;
	}

	std::cout << start;
	for (T element : vector) {
		std::cout << element << separator;
	}
	std::cout << end;
}

template <class T>
void show_vector_commas(const std::vector<T>&vector) {
	show_vector(vector,std::to_string(vector.size()) + " : ",", ","\b\b;\n");
}

template <class T>
void show_vector_lines(const std::vector<T>& vector) {
	show_vector(vector, vector.size() + " :\n", "\n\t", "\r;\n");
}

template<class T>
void showArray(T* array,size_t size,std::string start, std::string separator, std::string end) {
	if(array != nullptr){

		if(size < 1){
			std::cout << "empty array\n";
			return;
		}

		std::cout << start;
		for (int k = 0; k < size; k++) {
			std::cout << array[k] << separator;
		}
		std::cout << end;

	}
}

size_t gcd(size_t a, size_t b);
size_t lcm(size_t a, size_t b);

std::string split(std::string& text, size_t& pos, char symbol, bool shift = 1);
