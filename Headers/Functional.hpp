#pragma once
#include "includes.hpp"
#include <cstdint>
#include <chrono>

namespace Functions {

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

	template <class Type>
	uintptr_t pointerToInt(Type* ptr) {
		return reinterpret_cast<uintptr_t>(ptr);
	}

	template <class Type>
	Type* intToPointer(uintptr_t intPtr){
		return reinterpret_cast<Type*>(intPtr);
	}

	template <class Type, class Output>
	vector<Output> map(vector<Type> container, function<Output(Type)> functor){
		vector<Output> result = {};
		result.reserve(container.size());

		for(const auto& element : container){
			result.emplace_back(functor(element));
		}	

		return result;
	}

	template <class Outputable>
	std::string extract_from_operator(const Outputable & obj){
		static std::ostringstream capture_stream;
		capture_stream << obj;
		return capture_stream.str();
	}

	pair<vector<int>*, vector<vector<int>>*> topological_sort(int n, vector<vector<int>> const& directedGraph);

	size_t cantor_pairing(size_t x, size_t y);

	template<typename Func, typename... Args>
	uint64_t test(Func&& func, Args&&... args) {
		uint64_t start = std::chrono::duration_cast<std::chrono::milliseconds>(
			std::chrono::system_clock::now().time_since_epoch()
		).count();

		std::invoke(func, args...);

		uint64_t  finish = std::chrono::duration_cast<std::chrono::milliseconds>(
			std::chrono::system_clock::now().time_since_epoch()
		).count();
		return finish - start;
	}

	template<typename Func, typename... Args>
	double testN(uint32_t n, Func&& func, Args&&... args){
		double sum = 0;

		for(int i = 0; i < n; ++i){
			uint64_t start = std::chrono::duration_cast<std::chrono::milliseconds>(
				std::chrono::system_clock::now().time_since_epoch()
			).count();

			std::invoke(func, args...);

			uint64_t  finish = std::chrono::duration_cast<std::chrono::milliseconds>(
				std::chrono::system_clock::now().time_since_epoch()
			).count();

			sum += finish - start;
		}

		return sum / n;
	}
}