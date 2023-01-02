#pragma once
#include "../Headers/CodeWars.h"

map<char, string> shift = {
	{'1',"124"},
	{'2',"1235"},
	{'3',"236"},
	{'4',"1457"},
	{'5',"24568"},
	{'6',"3569"},
	{'7',"478"},
	{'8',"05879"},
	{'9',"689"},
	{'0',"08"},
};

vector<string> get_pins(string observed) {

	vector<string> result = vector<string>();

	if (observed.length() > 1) {

		vector<string> nested = get_pins(observed.substr(1, observed.length() - 1));

		for (char c : shift[observed[0]])
		{
			for (string s : nested)result.push_back(c + s);
		}
	}
	else {
		for (char c : shift[observed[0]]) {
			result.push_back(string(1, c));
		}
	}

	return result;
}

pair<bool, string> dep[] = {
	{1,"00136251249986374875"},
	{0,"00284286844426026860"},
	{0,"0165298347"},
	{1,"2"},
	{1,"31975"},
	{0,"0440"},
	{0,"06196451229380354877"},
	{0,"08266446280"},
};

int twoLastDigits(int a, int b) {

	a %= 10;
	b %= 100;

	if (b == 0 || a == 1)return 1;
	if (a == 0)return 0;

	int t = b - dep[a - 2].first;
	int l = dep[a - 2].second.length();
	int secondDigit = dep[a - 2].second[((t % l == 0) ? l : t % l) - 1] - 48;

	if (a == 5 || a == 6)b = 1;
	return secondDigit * 10 + int(pow(a, (b % 4 == 0) ? 4 : b % 4)) % 10;
}


int last_digit(list<int> array) {
	if (array.size() == 0)return 1;

	vector<int> vector(array.begin(), array.end());
	int result = vector[vector.size() - 1];

	for (int k = vector.size() - 2; 0 <= k; k--) {
		int mediator = twoLastDigits(vector[k], result);

		result = twoLastDigits(vector[k], result);
	}

	return result % 10;

}

vector<int> getDivisors(size_t num) {
	vector<int> divisors = vector<int>();

	for (int k = 2; num != 1;k++) {
		if (num % k == 0) {
			divisors.push_back(k);
			while (num % k == 0)num /= k;
		}
	}

	return divisors;
}

long properFractions(long n)
{
	vector<int> divisors = getDivisors(n);
	
	int result = n;

	for (int k = 0; k < divisors.size(); k++) {

		for (int j = k - 1; 0 <= j; j--) {
			result += n / (divisors[k]*divisors[j]);
		}

		result -= n / divisors[k];
	}

	return result;
}

long properFractionsR(long n)
{
	vector<int> divisors = getDivisors(n);
	long count = 1;

	for (int k = 2; k < n; k++) {
		if (divisors.end() == find_if(divisors.begin(), divisors.end(), [k](const int& e) {return k % e == 0; }))count++;
	}

	return count;
}
