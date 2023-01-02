#pragma once
#include "includes.h"

template<class T>
struct Point {
	T x;
	T y;
	Point();
	Point(T x, T y);

	///////////////////////////////

	void input();
	double get_distance(Point<T> const& end = Point<T>());
	Point<T>& operator+(Point<T> const& right);
	void operator +=(Point<T> const& right);
	bool operator==(Point<T> const& right);
	bool operator <(Point<T> const& right);
	bool operator <=(Point<T> const& right);
	bool operator >(Point<T> const& right);
	bool operator >=(Point<T> const& right);
	template<class T>friend ostream& operator <<(ostream& out, Point<T> const& p);
};