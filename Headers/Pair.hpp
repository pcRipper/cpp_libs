#pragma once
#include "includes.hpp"

template <class Key, class Value>
struct Pair {
	Key key;
	Value value;
	////////////
	Pair(Key key = Key(), Value value = Value());
	Pair(Pair<Key, Value>* obj);
	Pair(Pair<Key, Value> const& obj);

	template <class Key, class Value> friend ostream& operator <<(ostream& out, Pair<Key, Value>* element);
	template <class Key, class Value> friend ostream& operator <<(ostream& out, Pair<Key, Value> const& element);
	void operator += (Pair<Key, Value> const& obj);
	bool operator ==(Pair<Key, Value>* element);
	bool operator ==(Pair<Key, Value> const& element);
};