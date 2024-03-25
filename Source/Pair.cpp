#pragma once
#include "../Headers/Pair.hpp"

template<class Key,class Value>
Pair<Key,Value>::Pair(Key key, Value) :key(key), value(value) {};

template<class Key,class Value>
Pair<Key, Value>::Pair(Pair<Key, Value> *obj) {
	if (obj != nullptr) {
		key = Key(obj->key);
		value = Value(obj->value);
	}
	else Pair<Key, Value>();
}

template<class Key, class Value>
Pair<Key, Value>::Pair(Pair<Key, Value> const & obj) {
	key = Key(obj.key);
	value = Value(obj.value);
}

template<class Key, class Value>
ostream& operator <<(ostream& out, Pair<Key, Value>* element) {
	if (element != nullptr) {
		out << "[" << element->key << "]" << element->value;
	}
	else out << "__";
	return out;
}

template<class Key, class Value>
ostream& operator <<(ostream& out, Pair<Key, Value> const & element) {
	out << "[" << element.key << "]" << element.value;
	//debug \/
	//out << &element << "[" << element.key << "]" << element.value;
	return out;
}

template<class Key, class Value>
void Pair<Key, Value>::operator += (Pair<Key, Value> const& obj) {
	key += obj.key;
	value += obj.value;
}

template<class Key,class Value>
bool Pair<Key, Value>::operator==(Pair<Key, Value>* element) {
	return key == element->key && value == element->value;
}

template<class Key, class Value>
bool Pair<Key, Value>::operator==(Pair<Key, Value> const & element) {
	return key == element.key && value == element.value;
}

bool pair_comparer_grow(Pair<char, int> const & left, Pair<char, int> const & right) {
	return left.value < right.value;
}

bool pair_comparer_fall(Pair<char, int> const & left, Pair<char, int> const & right) {
	return left.value > right.value;
}