#pragma once
#include "includes.h"

struct numPair{
public:
	
	int digit;
	size_t power;
	
	numPair(int digit,size_t power);
	
	friend bool operator == (const numPair & left,const numPair & right);
	friend ostream& operator<<(ostream& out,const numPair& obj);
	~numPair();
};

class BigNum{
	
	bool sign;
	vector<numPair> num;

public:
	
	BigNum();
	BigNum(long long number);
	BigNum(string number,bool sign = true);
//	BigNum(vector<numPair> dNumber);

	void add(numPair pair);
	void show();
	BigNum operator*(const BigNum& obj);
	
	friend ostream& operator<<(ostream& out, const BigNum& obj);

	~BigNum();
	
};
