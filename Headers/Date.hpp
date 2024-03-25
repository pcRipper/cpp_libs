#pragma once
#include "includes.hpp"

class date {
	static const uint8_t months_lengths[12];
	static const string  months_names[12];

	uint8_t day;
	uint8_t month;
	uint32_t year;

	static bool output_type;
public:

	date();
	date(uint64_t days);
	date(uint32_t _year, uint64_t _month, uint64_t _day);

	static date* input_date();
	static void change_output_type();
	static date& current();

	void set_current();
	uint32_t days();

	friend string to_string(const date& obj);

	friend ostream& operator<<(ostream& out, const date& obj);
	friend bool operator==(const date& left, const date& right);
	friend bool operator<(const date& left, const date& right);
	friend bool operator>(const date& left, const date& right);
	friend bool operator<=(const date& left, const date& right);
	friend bool operator>=(const date& left, const date& right);
	friend uint32_t operator-(date left, date right);

	~date();
};