#include "../Headers/Functional.hpp"

size_t gcd(size_t a, size_t b)
{
	while (a * b != 0) {
		if (a > b)a %= b;
		else b %= a;
	}
	
	return a + b;
}

size_t lcm(size_t a, size_t b) {
	return a * b / gcd(a, b);
}

string split(string& text, size_t& pos, char symbol, bool shift) {
	string result = "";
	for (size_t length = text.length(); pos < length; pos++) {
		if (text[pos] == symbol)break;
		result += text[pos];
	}
	pos += shift;
	return(result);
}