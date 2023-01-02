#pragma once
#include "../Headers/Point.h"

template<class T>
Point<T>::Point() {};

template<class T>
Point<T>::Point(T x, T y) :
	x(x),
	y(y)
{};

template<class T>
void Point<T>::input() {
	cout << "x = ";
	cin >> x;
	cout << "y = ";
	cin >> y;
}

template <class T>
double Point<T>::get_distance(Point<T> const & end) {
	return(sqrt(pow(x - end.x, 2) + pow(y - end.y, 2)));
}

double Point<int>::get_distance(Point<int> const & end) {
	return(sqrt(pow(x,2) + pow(y,2)));
}

template <class T>
Point<T>& Point<T>::operator+(Point<T> const& right) {
	return(*new Point<T>(x + right.x,y + right.y));
}

template <class T>
void Point<T>::operator+=(Point<T> const & right) {
	x += right.x;
	y += right.y;
}

template <class T>
bool Point<T>::operator==(Point<T> const& right) {
	return(x == right.x && y == right.y);
}

template <class T>
bool Point<T>::operator<(Point<T> const& right) {
	return (get_distance() < right.get_distance());
}

template <class T>
bool Point<T>::operator<=(Point<T> const& right) {
	return (get_distance() <= right.get_distance());
}

template <class T>
bool Point<T>::operator>(Point<T> const& right) {
	return (get_distance() > right.get_distance());
}

template <class T>
bool Point<T>::operator>=(Point<T> const& right) {
	return (get_distance() >= right.get_distance());
}

template <class T>
ostream& operator <<(ostream& out, Point<T> const& p) {
	cout << "(" << p.x << ";" << p.y << ")";
	return out;
}


