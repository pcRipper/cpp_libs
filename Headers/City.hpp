#pragma once
#include "Date.hpp"
#include "List.hpp"
#include "includes.hpp"

template <class T>
string by3(const  T& obj) {
	string text = to_string(obj);

	int end = (obj >= 0) ? 0 : 1;
	for (int n = text.length() - 3; n > end; n -= 3) {
		text.insert(n, ",");
	}

	return text;
}

class City {
	string   name;
	date     est;
	uint32_t population;
public:

	static int (*comparer)(City&, City&);
	static City example[10];

	City(string name, date est, uint32_t population);
	City(const City& obj);
	City(City* obj);

	static City* input_city();

	string   getName();
	uint32_t getPopulation();
	date     getDate();

	friend string to_string(const City& obj);
	friend ostream& operator<<(ostream& out, const City& obj);

	bool operator ==(City& obj);

	bool operator <(City& obj);

	bool operator >(City& obj);

	~City();
};

