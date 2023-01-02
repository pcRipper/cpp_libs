#include "../Headers/Functional.h"

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