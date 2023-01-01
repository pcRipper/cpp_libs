#include "BigNum.h"
#include <map>
#include <functional>

numPair::numPair(int digit,size_t power):
	digit(digit),
	power(power)
{};

bool operator == (const numPair & left,const numPair & right){
	return left.power == right.power;
}

ostream& operator<<(ostream& out,const numPair& obj){
	out << obj.digit << " * 10^" << obj.power;
	return out;
}

ostream& operator<<(ostream& out, const BigNum& obj)
{
	auto it = max_element(obj.num.begin(), obj.num.end(), [](const numPair& left, const numPair& right) {return left.power < right.power; });

	if (it != obj.num.end()) {
		for (int k = it->power; k > -1; k--) {
			
			it = find_if(obj.num.begin(), obj.num.end(), [k](const numPair& element) {return element.power == k; });
			
			out << ((it != obj.num.end()) ? it->digit : 0);

		}
	}

	return out;
}

numPair::~numPair(){}

BigNum::BigNum(){
	num = vector<numPair>();
	sign = true;
}

BigNum::BigNum(long long number){
	num = vector<numPair>();
	sign = number < 0;
	
	for(int k = 0; number > 0; k++){
		num.push_back(numPair(number%10,k));
		number /= 10;
	}
}

BigNum::BigNum(string number,bool sign):sign(sign)
{
	num = vector<numPair>();
	
	if (number.length() > 0) {
		for (int k = 0; k < number.length(); k++) {
			num.push_back(numPair(
				number[k]-48,
				number.length() - (k+1)
			));
		}
	}
}

void BigNum::add(numPair pair){
	vector<numPair>::iterator findE = find_if(num.begin(),num.end(), [pair]( const numPair & element) {return element.power == pair.power; });
	
	if(findE != num.end()){
		pair.digit = findE->digit + pair.digit;
		findE->digit = pair.digit%10;
	}
	else{
		if (pair.digit % 10 != 0) {
			num.push_back(numPair(pair.digit % 10, pair.power));
		}
	}
	
	
	if(pair.digit > 9){
		add(numPair(pair.digit/10,pair.power+1));
	}
	
}

void BigNum::show(){
	cout << "Num :\n";
	for(long k = 0;k < num.size();k++){
		cout << num[k] << endl;
	}
}

BigNum BigNum::operator*(const BigNum& obj){
	
	BigNum result = BigNum(0);
	vector<numPair>::iterator findE; 
	
	
	for(long k = 0; k < this->num.size();k++){
		for(long j = 0;j < obj.num.size();j++){
			
			int newDigit = this->num[k].digit * obj.num[j].digit;
			long long newPower = this->num[k].power + obj.num[j].power; 
			
			if(newDigit != 0){
				
				result.add(numPair(newDigit,newPower));	
			}
		}
	}
	
	result.sign = (sign + obj.sign) % 2;
	
	return result;
}


class Date{};


class Report {};
class DataTable{};
class Measurement{};

class ORM {
public:
	void connect(map<string, string> settings);
	DataTable getData(function<bool(const Measurement& obj)> predicate);
};

class Statistic {
public:
	void mean(DataTable* data);
	void sd(DataTable* data);
	void correlation(DataTable* data, int c1, int c2);
};

class ReportComposer {
public:
	Report composeReportPDF(DataTable* data);
	Report composeReportDOCX(DataTable* data);
};

class ReportCreator {
public:
	static Report createReport(int ID_Station, Date From, Date To);
};



BigNum::~BigNum(){}

