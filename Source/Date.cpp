#pragma once
#include "../Headers/Date.h"
using namespace std;


date::date() {
	set_current();
}

date::date(uint64_t days) {
	*this = *new date(0, 1, days);
}

date::date(uint32_t _year, uint64_t _month, uint64_t _day) {

	_month--;
	bool isLeap = ((_year % 4 == 0) ? 1 : 0);
	if (_month > 11)_month = (_month % 12);

	for (; months_lengths[_month] + ((isLeap && _month == 1) ? 1 : 0) < _day; ++_month) {

		//cout << _day << "-" << _month << "-" << _year << endl;

		_day -= months_lengths[_month] + ((isLeap && _month == 1) ? 1 : 0);
		if (_month == 11) {
			_month = -1;
			_year++;
			isLeap = ((_year % 4 == 0) ? 1 : 0);
		}
	}
	if (_day == 0)_day = 1;

	day = _day;
	month = _month;
	year = _year;
};

date* date::input_date() {
	int year, month, day;

	cout << "Year = ";
	cin >> year;
	cout << "Month = ";
	cin >> month;
	cout << "Day = ";
	cin >> day;

	return new date(year, month, day);
}

void date::change_output_type() { output_type = !output_type; }

date& date::current() {
	const time_t current_time = time(0);
	tm* result = new tm();

	localtime_s(result, &current_time);

	return *new date(
		result->tm_year + 1900,
		result->tm_mon + 1,
		result->tm_mday
	);
}

void date::set_current() {
	date result = current();

	year = result.year;
	month = result.month;
	day = result.day;
}

string to_string(const date& obj) {
	if (obj.output_type) {
		return (
			to_string(obj.day) + "-" +
			to_string(obj.month + 1) + "-" +
			to_string(obj.year)
			);
	}
	return (
		"the " + to_string(obj.day) +
		" of " + obj.months_names[obj.month] +
		", " + to_string(obj.year)
		);
}

uint32_t date::days() {
	uint32_t result = (year * 365) + (year / 4) + day;

	for (int k = 0; k < month - 1; k++) {
		result += months_lengths[k];
	}

	return result;
}

ostream& operator<<(ostream& out, const date& obj) {
	out << to_string(obj);
	return out;
}

bool operator==(const date& left, const date& right) {
	return 0 == memcmp(&left, &right, sizeof(date));
}

bool operator<(const date& left, const date& right) {

	if (left.year < right.year)return true;
	else if (left.year == right.year) {
		if (left.month < right.month)return true;
		else if (left.month == right.month) {
			return left.day < right.day;
		}
	}
	return false;
}

bool operator>(const date& left, const date& right) {
	return right < left;
}

bool operator<=(const date& left, const date& right) {
	return !(left > right);
}

bool operator>=(const date& left, const date& right) {
	return !(left < right);
}

uint32_t operator-(date left, date right) {
	uint32_t l = left.days();
	uint32_t r = right.days();

	if (l > r)swap(l, r);

	return r - l;
}

date::~date() {

}

	  bool    date::output_type = false;
const uint8_t date::months_lengths[12] = { 31,28,31,30,31,30,31,31,30,31,30,31 };
const string  date::months_names[12] = { "January","February","March","April","May","June","July","August","September","October","November","December" };