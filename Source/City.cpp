#pragma once
#include "City.h"

using namespace std;

int CityPopulationGrow(City& l, City& r) {
	if (l.getPopulation() < r.getPopulation())return 1;
	if (l.getPopulation() > r.getPopulation())return -1;
	return 0;
}

int (*City::comparer)(City&, City&) = CityPopulationGrow;

City::City(string name, date est, uint32_t population) :
	name(name),
	est(est),
	population(population)
{};

City::City(const City& obj) {
	name = obj.name;
	est = obj.est;
	population = obj.population;
}

City::City(City* obj) {
	name = obj->name;
	est = obj->est;
	population = obj->population;
}

City* City::input_city() {
	string name;
	date est;
	uint32_t population;

	cin.clear();
	cin.ignore();

	cout << "Name : ";
	getline(cin, name);
	cout << "Date of foundations :\n";
	est = *date::input_date();
	cout << "Population : ";
	cin >> population;

	return new City(name, est, population);
}

string   City::getName() { return name; }
uint32_t City::getPopulation() { return population; }
date     City::getDate() { return est; }

string to_string(const City& obj) {
	return (
		obj.name + " : since " +
		to_string(obj.est) + ",pop. = " +
		by3(obj.population)
		);
}

ostream& operator<<(ostream& out, const City& obj) {
	cout << to_string(obj);
	return out;
}

bool City::operator ==(City& obj) {
	return (0 == comparer(*this, obj));
}

bool City::operator <(City& obj) {
	return (0 < comparer(*this, obj));
}

bool City::operator >(City& obj) {
	return (0 > comparer(*this, obj));
}

City::~City()
{

}

City City::example[10] =
{
	City("Kyiv", date(482, 1, 1), 2952301),
	City("Kharkiv", date(1654, 1, 1), 1421125),
	City("Lviv", date(1256, 1, 1), 717273),
	City("Odesa", date(1415, 4, 15), 1010537),
	City("Saint Petersburg", date(1703, 5, 27), 5351935),
	City("Moscow", date(1147, 1, 1), 13010112),
	City("Ottawa", date(1826, 1, 1), 1017449),
	City("Washington, D.C", date(1790, 1, 1), 689545),
	City("New Delhi", date(1911, 1, 1), 28514000),
	City("Rivne", date(1283, 1, 1), 243934)
};