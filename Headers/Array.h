#pragma once
#include <iostream>
#include <string>
#include <fstream>

using namespace std;

string to_string(string text);

class Range {
public:
	int from;
	int to;
	int step;

	Range(int from, int to, int step);
	Range();
};

template <class T>
void SWAP(T& left, T& right);

string split(string& text, size_t& pos, char symbol, bool shift = 1);

template <class T>
void showArray(T* array, size_t size, string separator = ", ", string end = "\b\b;\n");

#pragma region Array<T>

template <class T>
class Array {
	T* array;
	size_t size;
public:
	Array() :size(0), array(nullptr) {};
	Array(size_t init_size) : array(new T[init_size]), size(init_size) {};
	Array(size_t size, T* array, bool del_array = true);
	Array(Array<T> const& obj);
	//////////////////////////////////////////////////////////////////////////
	Array<T>* getPart_object(size_t from, size_t to);
	T* getPart_array(size_t from, size_t to);
	size_t getSize() { return(size); }
	void fill_function(T(*functor)());
	void fill_element(T element);
	void fill_index(T(*functor)(size_t));
	void set_size(size_t size);
	void add(T& element);
	void remove(size_t index);
	void show(string start, string separator, string end);
	void showComma();
	void showLine();
	int indexOf(T& obj);
	size_t count(bool (*comparer)(const T&));
	void mapIn(T* (*functor)(const T&));
	template <class F>
	Array<F>* mapOn(F* (*functor)(const T&));
	void read_file(string file_path, string key_word, T* (parse)(string));
	void qSort(bool (*comparer)(const T&, const T&), void (*swap)(T&, T&) = std::swap);
	void bubbleSort(bool (*comparer)(const T&, const T&), void (*swap)(T&, T&) = std::swap);
	void pickSort(bool (*comparer)(const T&, const T&), void (*swap)(T&, T&) = std::swap);
	void insertSort(bool (*comparer)(const T&, const T&), void (*swap)(T&, T&) = std::swap);
	void mergeSort(bool (*comparer)(const T&, const T&));
	void clean() { delete[] array; array = nullptr; size = 0; }
private:
	void mergeSortHelper(bool (*comparer)(const T&, const T&), size_t l, size_t r);
	void hoareSort(bool (*comparer)(const T&, const T&), void (*swap)(T&, T&), int l, int r);
	size_t quick_select_sort(bool (*comparer)(const T&, const T&), void (*swap)(T&, T&), size_t l, size_t r);
	//T* quick_select(bool (*comparer)(T, T), size_t l, size_t r, size_t k);
public:
	T* most(bool (*comparer)(const T&, const T&));
	T* mostN(bool (*comparer)(const T&, const T&), size_t index = 0);
	T& operator[](size_t index);
	void operator += (Array<T>* obj);
	template <class T> friend ostream& operator<<(ostream& out, Array<T> obj);
	~Array() { delete[] array; };
};

template <class T>
Array<T>::Array(size_t size, T* array, bool del_array) {
	if (array != nullptr) {
		this->size = size;
		this->array = new T[size];

		for (size_t k = 0; k < size; k++) {
			this->array[k] = T(array[k]);
		}

		if (del_array)delete[] array;
	}
}

template <class T>
Array<T>::Array(Array<T> const & obj) {

	this->size = obj.size;
	
	array = new T[size];

	for (size_t k = 0; k < size;k++) {
		array[k] = *new T(obj.array[k]);
	}

}

template <class T>
Array<T>* Array<T>::getPart_object(size_t from, size_t to) {
	
	Array<T>* result = Array<T>(1 + to - from);
	result->array = getPart_array(from, to);
	
	return result;
}

template <class T>
T* Array<T>::getPart_array(size_t from, size_t to) {

	from = (from > size) ? 0 : from;
	to = min(to, size);

	T* newArray = new T[1 + to - from];

	for (size_t k = from, j = 0; k <= to; k++, j++)newArray[j] = *new T(array[k]);

	return newArray;
}

template <class T>
void Array<T>::fill_function(T(*functor)()) {
	for (size_t k = 0; k < size; k++) {
		array[k] = *new T(functor());
	}
}

template <class T>
void Array<T>::fill_element(T element) {
	for (size_t k = 0; k < size; k++) {
		array[k] = *new T(element);
	}
}

template <class T>
void Array<T>::fill_index(T(*functor)(size_t)) {
	for (size_t k = 0; k < size; k++) {
		array[k] = *new T(functor(k));
	}
}

template<class T>
void Array<T>::set_size(size_t size) {
	T* previous = array;
	array = new T[size];

	size_t copying = min(size, this->size);

	for (size_t k = 0; k < copying; k++) {
		array[k] = *new T(previous[k]);
	}

	this->size = size;
	delete[] previous;
}

template<class T>
void Array<T>::add(T & element) {

	set_size(size + 1);

	array[size - 1] = *new T(element);
}

template <class T>
void Array<T>::remove(size_t index) {
	if (index < size) {
		T* previous = array;
		array = new T[--size];

		for (size_t k = 0; k < index; k++)array[k] = *new T(previous[k]);
		for (size_t k = index; k < size; k++)array[k] = *new T(previous[k + 1]);

		delete[] previous;
	}
}

template<class T>
void Array<T>::show(string start, string separator, string end) {
	cout << start;
	for (int k = 0; k < size; k++) {
		cout << array[k] << separator;
	}
	cout << end;
}

template<class T>
void Array<T>::showComma() {
	show(to_string(getSize()) + " : ", ", ", "\b\b;\n");
}

template<class T>
inline void Array<T>::showLine()
{
	show(to_string(getSize()) + " :\n\t", "\n\t", "\r;\n");
}

template <class T>
int Array<T>::indexOf(T& obj) {
	for (size_t k = 0; k < size; k++) {
		if (array[k] == obj)return k;
	}
	return -1;
}

template <class T>
size_t Array<T>::count(bool (*comparer)(const T&)) {
	size_t count = 0;

	for (int k = 0; k < size; k++) {
		if (comparer(array[k]))count++;
	}

	return count;
}

template <class T>
void Array<T>::mapIn(T* (*functor)(const T&)) {
	for (int k = 0; k < size; k++) {
		array[k] = *functor(array[k]);
	}
}

template <class T>
template <class F>
Array<F>* Array<T>::mapOn(F* (*functor)(const T&)) {
	Array<F>* result = new Array<F>(size);

	for (size_t k = 0; k < size; k++) {
		(*result)[k] = *functor(array[k]);
	}

	return result;
}

template<class T>
void Array<T>::read_file(string file_path, string key_word, T* (parser)(string)) {
	ifstream file(file_path);

	if (file.is_open()) {

		string line;

		while (!file.eof()) {

			getline(file, line);

			size_t pos = ((key_word == "") ? 0 : line.find(key_word));

			if (pos != string::npos && line.length() > key_word.length()) {
				pos += key_word.length();
				T* res = parser(line.substr(pos, line.length() - pos));

				if (res != nullptr)add(*res);

			}
		}

		file.close();
	}
}

template <class T>
void Array<T>::bubbleSort(bool (*comparer)(const T&, const T&), void (*swap)(T&, T&)) {
	for (int k = 1, j; k < size - 1; ++k) {
		for (j = 1; j < size; ++j) {
			if (comparer(array[j], array[j - 1]))swap(array[j], array[j - 1]);
		}
	}
}

template <class T>
void Array<T>::pickSort(bool (*comparer)(const T&, const T&),void (*swap)(T&,T&)) {
	for (int k = 0; k < size; ++k) {
		int b = k;
		for (int j = k; j < size; ++j) {
			if (comparer(array[j], array[b]))b = j;
		}
		if (k != b)swap(array[k], array[b]);
	}
}

template <class T>
void Array<T>::insertSort(bool (*comparer)(const T&, const T&), void (*swap)(T&, T&)) {
	for (int j = 1; j < size; j++) {
		for (int i = j; i != 0 && comparer(array[i], array[i - 1]); i--) {
			swap(array[i], array[i - 1]);
		}
	}
}

template <class T>
void Array<T>::mergeSort(bool (*comparer)(const T&, const T&)) {
	mergeSortHelper(comparer, 0, size - 1);
}

template <class T>
void Array<T>::mergeSortHelper(bool (*comparer)(const T&, const T&), size_t l, size_t r) {

	size_t middle = (l + r) / 2;

	if (r - l > 1) {
		mergeSortHelper(comparer, l, middle);
		mergeSortHelper(comparer, middle + 1, r);
	}

	size_t csize = 1 + r - l;
	T* newArray = new T[csize];
	size_t i = 0, j = l, k = middle + 1;

	while (j <= middle && k <= r) newArray[i++] = (comparer(array[j], array[k])) ? array[j++] : array[k++];
	while (j <= middle)		      newArray[i++] = array[j++];
	while (k <= r)			      newArray[i++] = array[k++];

	memcpy(&array[l], &newArray[0], sizeof(T) * csize);
	delete newArray;
}

template <class T>
void Array<T>::qSort(bool (*comparer)(const T&, const T&), void (*swap)(T&, T&)) {
	hoareSort(comparer, swap ,0, size - 1);
}

template <class T>
void Array<T>::hoareSort(bool (*comparer)(const T&, const T&), void (*swap)(T&, T&), int l, int r) {
	if (0 < size && l < r) {
		T x = array[(l+r)/2];
		int i = l, j = r;

		while (i < j)
		{
			while (comparer(array[i], x)) i++;
			while (comparer(x, array[j])) j--;

			if (i <= j)swap(array[i++], array[j--]);
		}

		hoareSort(comparer, swap, l, j);
		hoareSort(comparer, swap, i, r);
	}
}

template <class T>
size_t Array<T>::quick_select_sort(bool (*comparer)(const T&, const T&), void (*swap)(T&, T&), size_t l, size_t r) {
	size_t x = array[r], i = l;

	for (int j = l; j <= r - 1; j++) {
		if (comparer(array[j], array[x])) {
			swap(array[i], array[j]);
			i++;
		}
	}
	swap(array[i], array[r]);
	return i;
}

template <class T>
T* Array<T>::most(bool (*comparer)(const T&, const T&)) {
	size_t index = 0;

	for (size_t k = 1; k < size; k++) {

		if (comparer(array[index], array[k])) {
			index = k;
		}
	}

	return &array[index];
}

template <class T>
T* Array<T>::mostN(bool (*comparer)(const T&, const T&), size_t index) {//a lot of memory(rewrite)
	Array<T> sorted = Array<T>(*this);

	sorted.qSort(comparer);

	if (index > size)index = size - 1;
	return new T(sorted[index]);
}

template <class T>
T& Array<T>::operator[](size_t index) {
	if (index < size) {
		return(array[index]);
	}
	return(*new T());
}

template <class T>
void Array<T>::operator+=(Array<T>* obj) {
	if (obj != nullptr && obj->size > 0) {
		set_size(size + obj->size);

		for (int k = 0, j = size - obj->size; k < obj->size;) {
			array[j++] = T(obj->array[k++]);
		}

	}
}

template <class T>
ostream& operator<<(ostream& out, Array<T> obj) {
	out << to_string(obj);
	return out;
}

template <class T>
string to_string(Array<T>& obj) {
	string result = "";

	if (obj.getSize() == 0)result = "empty array";
	else {
		result += to_string(obj.getSize()) + " : ";
		for (int k = 0; k < obj.getSize(); k++) {
			result += to_string(obj[k]) + ", ";
		}
		result += "\b\b;";
	}

	return result;

}

#pragma endregion

#pragma region Matrix<T>

template <class T>
class Matrix {
	Array<T>* matrix;
	size_t rows;
public:
	Matrix(size_t rows = 0, size_t columns = 0, bool fill = false, T filler = T());
	Matrix(Matrix* obj, bool del_obj = false);
	size_t getRows() { return(rows); }
	size_t getColumns() { return(matrix[0].getSize()); }
	void fill_function(T(*functor)());
	void fill_element(T element);
	void setSize(size_t rows, size_t columns);
	void show(string separator);
	void sort(bool (*comparer)(const T&, const T&));
	void sortRows(bool (*comparer)(const T&, const T&));
	void read_file(string file_path, char separator, T(*parser)(string));
	Array<T>& operator[](size_t index);
private:
	void set_columns(size_t size);
	void set_rows(size_t size);
public:
	~Matrix() { delete[] matrix; }
};

template <class T>
Matrix<T>::Matrix(size_t rows, size_t columns, bool fill, T filler) :rows(rows) {
	matrix = new Array<T>[rows];

	for (size_t k = 0; k < rows; k++) {
		matrix[k].set_size(columns);

		if (fill)matrix[k].fill_element(filler);
	}

}

template <class T>
Matrix<T>::Matrix(Matrix<T>* obj, bool del_obj) {
	if (obj != nullptr) {

		rows = obj->rows;
		matrix = new Array<T>[rows];

		for (size_t r = 0; r < obj->rows; r++) {
			matrix[r] = *new Array<T>(obj->matrix[r]);
		}

		if (del_obj)delete obj;
	}
}

template <class T>
void Matrix<T>::fill_function(T(*functor)()) {
	for (size_t r = 0; r < rows; r++) {
		matrix[r].fill_function(functor);
	}
}

template <class T>
void Matrix<T>::fill_element(T element) {
	for (size_t r = 0; r < rows; r++) {
		matrix[r].fill_element(element);
	}
}

template <class T>
void Matrix<T>::setSize(size_t rows, size_t columns) {
	set_rows(rows);
	set_columns(columns);
}

template <class T>
void Matrix<T>::set_columns(size_t size) {
	for (int k = 0; k < rows; k++) {
		matrix[k].set_size(size);
	}
}

template <class T>
void Matrix<T>::set_rows(size_t size) {
	Array<T>* n_matrix = new Array<T>[size];

	size_t max = ((rows < size) ? rows : size);

	for (size_t k = 0; k < max; k++) {
		n_matrix[k].set_size(getColumns());

		memcpy(&n_matrix[k][0], &matrix[k][0], sizeof(T) * matrix[k].getSize());
	}

	for (max--; max < size; max++) {
		n_matrix[max].set_size(getColumns());
	}

	delete[] matrix;

	matrix = n_matrix;
	rows = size;
}

template <class T>
void Matrix<T>::show(string separator) {
	cout << rows << "x" << matrix[0].getSize() << " :\n";
	for (size_t r = 0; r < rows; r++) {
		matrix[r].show("", separator, "\n");
	}
	cout << "\n\n";
}

template <class T>
void Matrix<T>::sortRows(bool (*comparer)(const T&, const T&)) {

	for (int rows = 0; rows < matrix->getSize(); rows++) {
		matrix[rows].mergeSort(comparer);
	}

}

template <class T>
void Matrix<T>::sort(bool (*comparer)(const T&, const T&)) {

	//undefined

}

template <class T>
void Matrix<T>::read_file(string file_path, char separator, T(*parser)(string)) {
	ifstream file(file_path);

	if (file.is_open()) {

		string line;

		for (size_t r = 0; !file.eof(); r++) {

			if (r + 1 > rows)set_rows(r + 1);

			getline(file, line);

			size_t left = 0;
			size_t right = 0;

			for (size_t c = 0; right < line.length(); c++) {

				if (c + 1 > getColumns())set_columns(c + 1);

				matrix[r][c] = parser(split(line, right, separator));

				left = right;
			}
		}

		file.close();
	}
}

template <class T>
Array<T>& Matrix<T>::operator[](size_t index) {
	if (index < rows) {
		return(matrix[index]);
	}
	return(*new Array<T>());
}

#pragma endregion