#pragma once

template <class T>
bool comparer_fall(T const& left, T const& right) {
	return(left > right);
}

template <class T>
bool comparer_fallN(T const& left, T const& right) {
	return(left >= right);
}

template <class T>
bool comparer_grow(T const& left, T const& right) {
	return(left < right);
}

template <class T>
bool comparer_growN(T const& left, T const& right) {
	return(left <= right);
}