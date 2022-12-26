#pragma once
#include <iostream>
#include <string>
using namespace std;

template<class T>
class ListNode {
public:
	T* element;
	ListNode* next;

	ListNode(T* element, ListNode* next = nullptr);
	ListNode(T& element, ListNode* next = nullptr);
	ListNode(const ListNode<T>& obj);
	
	template <class T> friend ostream& operator<<(ostream& out, const ListNode<T>* element);
	template <class T> friend ostream& operator<<(ostream& out, const ListNode<T>& element);
	ListNode<T>* operator++(int);

	~ListNode();
};

template<class T>
class List {
protected:
	uint64_t size;

	ListNode<T>* head;
	ListNode<T>* tail;
	ListNode<T>* current;  //for iterations
	ListNode<T>* previous; //for iterations
public:
	List();

	List(const List<T>& obj);
	List(T *arguments,size_t size);

	uint64_t Size() { return(size); }
	void add(T* element);
	void add(T& element);
	void addOnSorted(T* element, bool (*comparer)(T&, T&));
	void concat(List<T>* list);
	void remove(size_t index);
	void reverse();
	ListNode<T>* find(bool(*comparer)(T&, T&), T* element);
	void setTail();
	List<T>* filter(bool (*func)(T&));
	void mapIn(T* (*func)(T&));
	template <class F> List<F>* mapOn(F* (*func)(T&));
	template <class F> F* fold(void (*fold_function)(F*, T*), F* start_value);
	void show(string separator, string begin = "", string end = "");
	void show_t1();
	void show_t2();
	void pickSort(bool (*comparer)(T&, T&));
	void insertSort(bool (*comparer)(T&, T&));
	void clean();
	T* operator[](size_t index);
	~List();
};