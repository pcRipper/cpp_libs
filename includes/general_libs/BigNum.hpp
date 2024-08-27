#pragma once
#include "includes.hpp"

struct numPair{
public:
	
	int digit;
	size_t power;
	
	numPair(int digit,size_t power);
	
	friend bool operator == (const numPair & left,const numPair & right);
	friend std::ostream& operator<<(std::ostream& out,const numPair& obj);
	~numPair();
};

class BigNum{
	
	bool sign;
	std::vector<numPair> num;

public:
	
	BigNum();
	BigNum(long long number);
	BigNum(std::string number,bool sign = true);
//	BigNum(vector<numPair> dNumber);

	void add(numPair pair);
	void show();
	BigNum operator*(const BigNum& obj);
	
	friend std::ostream& operator<<(std::ostream& out, const BigNum& obj);

	~BigNum();
	
};
