#pragma once
#include "List.h"

template <class T>
ListNode<T>::ListNode(T* element, ListNode* next) :
	element(element),
	next(next)
{}

template<class T>
ListNode<T>::ListNode(T & element, ListNode * next) :
	element(new T(element)),
	next(next)
{};

template <class T>
ListNode<T>::ListNode(const ListNode<T>& obj) {
	element = new T(*obj.element);
	next = nullptr;
}

template <class T>
ostream& operator<<(ostream& out,const ListNode<T>* element) {
	out << *element->element;
	//cout << "[" << &*element->element << "]" << *element->element;
	return out;
}

template <class T>
ostream& operator<<(ostream& out, const ListNode<T>& element)
{
	out << element->element;
	return out;
}

template <class T>
ListNode<T>* ListNode<T>::operator++(int) {
	return next;
}

template <class T>
ListNode<T>::~ListNode() {
	delete element;
	next = nullptr;
}

template <class T>
List<T>::List() {
	size = 0;
	head = nullptr;
	tail = nullptr;
	current = nullptr;
}


template <class T>
List<T>::List(const List<T>& obj) {
	if (obj.head != nullptr) {
		ListNode<T>* current_obj;

		head = new ListNode<T>(*obj.head);
		current = head;
		current_obj = obj.head->next;

		while (current_obj != nullptr) {

			current->next = new ListNode<T>(*current_obj);

			current = (*current)++;
			current_obj = current_obj->next;
		}

		size = obj.size;
		tail = current;
		current = nullptr;
	}
	else {
		List();
	}
}

template <class T>
List<T>::List(T *arguments,size_t size) {
	for (int k = 0; k < size; k++) {
		add(arguments[k]);
	}
}

template <class T>
void List<T>::add(T* element) {
	if (head == nullptr) {
		tail = head = new ListNode<T>(element);
	}
	else {
		tail = tail->next = new ListNode<T>(element);
	}
	size++;
}

template <class T>
void List<T>::add(T& element) {
	if (head == nullptr) {
		tail = head = new ListNode<T>(element);
	}
	else {
		tail = tail->next = new ListNode<T>(element);
	}
	size++;
}

template <class T>
void List<T>::addOnSorted(T* element, bool (*comparer)(T&, T&)) {

	if (head != nullptr) {
		if (comparer(*element, *head->element)) {
			head = new ListNode<T>(element, head);
		}
		else {
			for (current = head->next, previous = head; comparer(*current->element, *element) && current != nullptr;) {
				previous = current;
				current = (*current)++;
			}
			previous->next = new ListNode<T>(element, current);

			if (current == nullptr)tail = previous->next;
		}
	}
	else {
		add(element);
	}

	size++;

}

template <class T>
void List<T>::concat(List<T>* list) {
	for (current = list->head; current != nullptr; current = (*current)++)add(new T(current->element));
}

template<class T>
void List<T>::remove(size_t index) {
	if (head != nullptr) {
		size_t element = 0;

		for (current = head; element < index && current != nullptr; element++) {
			previous = current;
			current = (*current)++;
		}

		if (element == index && current != nullptr) {

			if (current == head) {
				head = current->next;
			}
			else if (current->next == nullptr) {
				tail = previous;
				previous->next = nullptr;
			}
			else {
				previous->next = current->next;
			}

			current->next = nullptr;
			delete current;
			size--;

		}
	}
}

template <class T>
void List<T>::reverse() {
	if (head != nullptr) {
		ListNode<T>* prev = nullptr;
		current = head->next;
		tail = head;

		while (true) {
			head->next = prev;
			prev = head;
			if (current == nullptr)break;
			head = current;
			current = (*current)++;
		}
	}
}

template <class T>
ListNode<T>* List<T>::find(bool(*comparer)(T&, T&), T* element) {
	for (current = head; current != nullptr && !comparer(*current->element, *element); current = (*current)++);

	return current;
}

template <class T>
void List<T>::setTail() {
	for (current = head; current->next != nullptr; current = (*current)++);
	tail = current;
}

template <class T>
List<T>* List<T>::filter(bool (*func)(T&)) {
	List<T>* result = new List<T>();

	for (current = head; current != nullptr; current = (*current)++) {
		if (func(*current->element))result->add(current->element);
	}

	return result;
}

template <class T>
void List<T>::mapIn(T* (*func)(T&)) {
	for (current = head; current != nullptr; current = (*current)++) {
		T* prev_element = current->element;

		current->element = func(*current->element);

		delete prev_element;
	}
}

template <class T>
template <class F>
List<F>* List<T>::mapOn(F* (*func)(T&)) {
	List<F>* result = new List<F>();

	for (current = head; current != nullptr; current = (*current)++) {
		result->add(func(*current->element));
	}

	return result;
}

template<class T>
template<class F>
F* List<T>::fold(void (*fold_function)(F*, T*), F* start_value)
{
	for (current = head; current != nullptr; current = (*current)++) {
		fold_function(start_value, current->element);
	}

	return start_value;
}

template<class T>
void List<T>::show(string separator, string begin, string end) {
	if (head != nullptr) {
		cout << begin;
		for (current = head; current != nullptr; current = (*current)++) {
			cout << current << separator;
		}
		cout << end;
	}
	else cout << "\nList is empty!\n";
}

template<class T>
void List<T>::show_t1()
{
	show(", ", to_string(size) + " : ", "\b\b;\n");
}

template<class T>
void List<T>::show_t2()
{
	show("\n\t", to_string(size) + " :\n\t", "\r;\n");
}

template<class T>
void List<T>::pickSort(bool (*comparer)(T&, T&)) {
	List<T>* new_list = new List<T>();

	while (head != nullptr) {

		size_t k = 1;
		size_t max = 0;
		ListNode<T>* maxP = head;

		for (current = head->next; current != nullptr; current = (*current)++) {
			if (comparer(*current->element, *maxP->element)) {
				max = k;
				maxP = current;
			}
			k++;
		}

		new_list->add(new T(*maxP->element));
		remove(max);
	}

	head = new_list->head;
	tail = new_list->tail;
	size = new_list->size;

}

template<class T>
void List<T>::insertSort(bool (*comparer)(T&, T&)) {
	if (head != nullptr) {

		ListNode<T>* prev = head, * picked, * current;

		for (current = head->next; current != nullptr; current = (*current)++) {

			if (comparer(*current->element, *prev->element)) {

				picked = current;
				prev->next = current->next;

				if (comparer(*picked->element, *head->element)) {
					picked->next = head;
					head = picked;
				}
				else {
					for (current = head; current->next != nullptr && !comparer(*picked->element, *current->next->element); current = (*current)++);

					picked->next = current->next;
					current->next = picked;
				}

				current = prev;
			}
			prev = current;

		}

		setTail();

	}
}

template <class T>
void List<T>::clean() {

	if (head != nullptr) {
		for (current = head->next; true;) {
			delete head;
			if (current == nullptr)return;
			head = current;
			current = head->next;
		}
	}

	current = head = tail = nullptr;
}

template<class T>
T* List<T>::operator[](size_t index) {
	for (current = head; index != 0 && current != nullptr; current = (*current)++) {
		index--;
	}
	return((current == nullptr) ? nullptr : current->element);
}

template<class T>
List<T>::~List() {
	clean();
}