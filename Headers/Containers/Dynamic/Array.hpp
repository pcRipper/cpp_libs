#pragma once
#include "../../includes.hpp"
#include "../../Functional.hpp"

template <class T>
void SWAP(T& left, T& right) {
	T time = T(left);
	left = *new T(right);
	right = *new T(time);
}

#pragma region Array<T>

template <class T>
class Array {
	T* array;
	size_t size;
	size_t maxSize;
public:
	///constructors

	Array() :size(0), maxSize(0), array(nullptr) {};
	Array(size_t init_size) : array(new T[init_size]), size(init_size), maxSize(init_size) {};
	Array(size_t size, T* array, bool del_array = true);
	Array(Array<T> const& obj);

	/// fillers
	void fill_element(T element);
	//void fill_index(T(*functor)(size_t)); <- old version
	void fill_index(std::function<T(const size_t)> functor = [](const size_t x) { return x; });
	void read_file(std::string file_path, std::string key_word, std::function<T* (std::string)> parser);

	/// sets && gets
	void set_size(size_t size);
	Array<T>* getPart_object(size_t from, size_t to);
	T* getPart_array(size_t from, size_t to);
	size_t getSize() { return(size); }

	/// add && remove
	void add(T & element);
	void add(T && element);
	void remove(size_t index);
	void clean() { delete[] array; array = nullptr; size = 0; }

	/// show
	void show(std::string start, std::string separator, std::string end);
	void showComma();
	void showLine();

	/// filter && find
	int indexOf(T& obj);
	/*size_t count(bool (*comparer)(const T&));
	T* most(bool (*comparer)(const T&, const T&));
	T* mostN(bool (*comparer)(const T&, const T&), size_t index = 0);*/

	size_t count(std::function <bool (const T&)> comparer);
	T* most(std::function <bool (const T&, const T&)> comparer);
	T* mostN(std::function <bool(const T&, const T&)> comparer, size_t index = 0);
	Array<T>* filter(std::function <bool(const T&)> comparer);

	///transformers
	//template <class F> Array<F>* mapOn(F* (*functor)(const T&)); <- old version
	//void mapIn(T* (*functor)(const T&)); <- old version

	template <class F> Array<F>* mapOn(std::function <F* (const T&)> functor);
	void mapIn(std::function<T* (const T&)> functor);

	///sort
	//void qSort(bool (*comparer)(const T&, const T&), void (*swap)(T&, T&) = std::swap); <- old version
	//void bubbleSort(bool (*comparer)(const T&, const T&), void (*swap)(T&, T&) = std::swap); <- old version
	//void pickSort(bool (*comparer)(const T&, const T&), void (*swap)(T&, T&) = std::swap); <- old version
	//void insertSort(bool (*comparer)(const T&, const T&), void (*swap)(T&, T&) = std::swap); <- old version
	//void mergeSort(bool (*comparer)(const T&, const T&)); <- old version

	void qSort(std::function<bool(const T&, const T&)> comparer, std::function<void(T&, T&)> swap = std::swap);
	void bubbleSort(std::function<bool(const T&, const T&)> comparer, std::function<void(T&, T&)> swap = std::swap);
	void pickSort(std::function<bool(const T&, const T&)> comparer, std::function<void(T&, T&)> swap = std::swap);
	void insertSort(std::function<bool(const T&, const T&)> comparer, std::function<void(T&, T&)> swap = std::swap);
	void mergeSort(std::function<bool(const T&, const T&)> comparer);

private:

	//void mergeSortHelper(bool (*comparer)(const T&, const T&), size_t l, size_t r); <- old version
	//void hoareSort(bool (*comparer)(const T&, const T&), void (*swap)(T&, T&), int l, int r); <- old version

	void mergeSortHelper(std::function<bool(const T&, const T&)> comparer, size_t l, size_t r);
	void hoareSort(std::function<bool(const T&, const T&)> comparer, std::function<void(T&, T&)> swap, int l, int r);

	//size_t quick_select_sort(bool (*comparer)(const T&, const T&), void (*swap)(T&, T&), size_t l, size_t r);
	//T* quick_select(bool (*comparer)(T, T), size_t l, size_t r, size_t k);
public:
	///operators
	T& operator[](size_t index);
	void operator += (Array<T>* obj);
	friend std::ostream& operator<<(std::ostream& out, Array<T> obj);
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
	to = std::min(to, size);

	T* newArray = new T[1 + to - from];

	for (size_t k = from, j = 0; k <= to; k++, j++)newArray[j] = *new T(array[k]);

	return newArray;
}

template <class T>
void Array<T>::fill_element(T element) {
	for (size_t k = 0; k < size; k++) {
		array[k] = *new T(element);
	}
}

template <class T>
void Array<T>::fill_index(std::function<T (const size_t)> functor) {
	for (size_t k = 0; k < size; k++) {
		array[k] = *new T(functor(k));
	}
}

template<class T>
void Array<T>::set_size(size_t size) {

	if(size < maxSize){
		this->size = size;
		return;
	}

	if(std::is_compound<T>::value){
		T* previous = this->array;
		this->array = new T[size];

		for (size_t k = 0; k < this->size; k++) {
			array[k] = *new T(previous[k]);
		}

		delete[] previous;
	}
	else {
		this->array = (T*)realloc(this->array, size);
	}

	this->maxSize = size;
	this->size = size;	
}

template<class T>
void Array<T>::add(T & element) {

	set_size(size + 1);

	array[size - 1] = element;

}

template <class T>
void Array<T>::add(T && element) {
	set_size(size + 1);
	array[size - 1] = std::move(element);
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
void Array<T>::show(std::string start, std::string separator, std::string end) {
	std::cout << start;
	for (size_t k = 0; k < size; k++) {
		std::cout << array[k] << separator;
	}
	std::cout << end;
}

template<class T>
void Array<T>::showComma() {
	show(std::to_string(getSize()) + " : ", ", ", "\b\b;\n");
}

template<class T>
inline void Array<T>::showLine()
{
	show(std::to_string(getSize()) + " :\n\t", "\n\t", "\r;\n");
}

template <class T>
int Array<T>::indexOf(T& obj) {
	for (size_t k = 0; k < size; k++) {
		if (array[k] == obj)return k;
	}
	return -1;
}

template <class T>
size_t Array<T>::count(std::function <bool(const T&)> comparer) {
	size_t count = 0;

	for (int k = 0; k < size; k++) {
		if (comparer(array[k]))count++;
	}

	return count;
}

template <class T>
void Array<T>::mapIn(std::function<T* (const T&)> functor) {
	for (int k = 0; k < size; k++) {
		array[k] = *functor(array[k]);
	}
}

template <class T>
template <class F>
Array<F>* Array<T>::mapOn(std::function<F* (const T&)> functor) {
	Array<F>* result = new Array<F>(size);

	for (size_t k = 0; k < size; k++) {
		(*result)[k] = *functor(array[k]);
	}

	return result;
}

template<class T>
void Array<T>::read_file(std::string file_path, std::string key_word, std::function<T* (std::string)> parser) {
	std::ifstream file(file_path);

	if (file.is_open()) {

		std::string line;

		while (!file.eof()) {

			getline(file, line);

			size_t pos = ((key_word == "") ? 0 : line.find(key_word));

			if (pos != std::string::npos && line.length() > key_word.length()) {

				pos += key_word.length();
				T* res = parser(line.substr(pos, line.length() - pos));

				if (res != nullptr)add(*res);

			}
		}

		file.close();
	}
}

template <class T>
void Array<T>::bubbleSort(std::function<bool(const T&, const T&)> comparer, std::function<void(T&, T&)> swap) {
	for (int k = 1, j; k < size - 1; ++k) {
		for (j = 1; j < size; ++j) {
			if (comparer(array[j], array[j - 1]))swap(array[j], array[j - 1]);
		}
	}
}

template <class T>
void Array<T>::pickSort(std::function<bool(const T&, const T&)> comparer, std::function<void(T&, T&)> swap) {
	for (int k = 0; k < size; ++k) {
		int b = k;
		for (int j = k; j < size; ++j) {
			if (comparer(array[j], array[b]))b = j;
		}
		if (k != b)swap(array[k], array[b]);
	}
}

template <class T>
void Array<T>::insertSort(std::function<bool(const T&, const T&)> comparer, std::function<void(T&, T&)> swap) {
	for (int j = 1; j < size; j++) {
		for (int i = j; i != 0 && comparer(array[i], array[i - 1]); i--) {
			swap(array[i], array[i - 1]);
		}
	}
}

template <class T>
void Array<T>::mergeSort(std::function<bool(const T&, const T&)> comparer) {
	mergeSortHelper(comparer, 0, size - 1);
}

template <class T>
void Array<T>::mergeSortHelper(std::function<bool(const T&, const T&)> comparer, size_t l, size_t r) {

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
void Array<T>::qSort(std::function<bool(const T&, const T&)> comparer, std::function<void(T&, T&)> swap) {
	hoareSort(comparer, swap ,0, size - 1);
}

template <class T>
void Array<T>::hoareSort(std::function<bool(const T&, const T&)> comparer, std::function<void(T&, T&)> swap, int l, int r) {
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

//template <class T>
//size_t Array<T>::quick_select_sort(bool (*comparer)(const T&, const T&), void (*swap)(T&, T&), size_t l, size_t r) {
//	size_t x = array[r], i = l;
//
//	for (int j = l; j <= r - 1; j++) {
//		if (comparer(array[j], array[x])) {
//			swap(array[i], array[j]);
//			i++;
//		}
//	}
//	swap(array[i], array[r]);
//	return i;
//}

template <class T>
T* Array<T>::most(std::function <bool(const T&, const T&)> comparer) {
	size_t index = 0;

	for (size_t k = 1; k < size; k++) {

		if (comparer(array[index], array[k])) {
			index = k;
		}
	}

	return &array[index];
}

template <class T>
T* Array<T>::mostN(std::function <bool(const T&, const T&)> comparer, size_t index) {//a lot of memory(rewrite)
	Array<T> sorted = Array<T>(*this);

	sorted.qSort(comparer);

	if (index > size)index = size - 1;
	return new T(sorted[index]);
}

template <class T>
Array<T>* Array<T>::filter(std::function <bool(const T&)> comparer) {
	Array<T>* result = new Array<T>();

	for (size_t k = 0; k < this->size; k++) {
		if (comparer(this->array[k])) {
			result->add(*new T(this->array[k]));
		}
	}

	return result;
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
std::ostream& operator<<(std::ostream& out, Array<T> obj) {
	out << to_string(obj);
	return out;
}

template <class T>
std::string to_string(Array<T>& obj) {
	std::string result = "";

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