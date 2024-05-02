#pragma once
#include "Array.hpp"

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
	void show(std::string separator);
	void sort(bool (*comparer)(const T&, const T&));
	void sortRows(bool (*comparer)(const T&, const T&));
	void read_file(std::string file_path, char separator, T(*parser)(std::string));
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
		matrix[r].fill_std::function(functor);
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
void Matrix<T>::show(std::string separator) {
	std::cout << rows << "x" << matrix[0].getSize() << " :\n";
	for (size_t r = 0; r < rows; r++) {
		matrix[r].show("", separator, "\n");
	}
	std::cout << "\n\n";
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
void Matrix<T>::read_file(std::string file_path, char separator, T(*parser)(std::string)) {
	std::ifstream file(file_path);

	if (file.is_open()) {

		std::string line;

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