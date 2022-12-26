#pragma once

#include <iostream>
#include "List.cpp"
using namespace std;

template <class T>
class Node {
public:
    Node<T>* left, * right, * prev;
    T* element;
    ///////
    Node(
        T* element = nullptr,
        Node<T>* prev = nullptr,
        Node<T>* left = nullptr,
        Node<T>* right = nullptr) :
        element(element),
        prev(prev),
        left(left),
        right(right)
    {};
    Node(Node<T>* obj);
    Node(Node<T> const& obj);
    template <class T> friend std::ostream& operator << (std::ostream& out, Node<T>* element);
    template <class T> friend std::ostream& operator << (std::ostream& out, Node<T>& element);
    bool operator == (Node<T> const& element);
    ~Node();
};

template <class T>
struct NWP {
    Node<T>* element;
    std::string path;
    NWP(std::string path = "", Node<T>* element = nullptr) :path(path), element(element) {};
    ~NWP() {
        element = nullptr;
    }
};

template <class T>
class Tree {
protected:
    Node<T>* root, * current, * previous;
public:
    Tree();
    Node<T>* getRoot();
    void setRoot(Node<T>* root);
    virtual void add(T* element) = 0;
    virtual void remove(bool (*predicate)(T*), Node<T>* (*normalization)(Node<T>*), bool recursively = false) = 0;
    int countNodes(Node<T>* current = root);
    int countLeafes(Node<T>* current = root);
    virtual List<T> findAll(bool (*predicate)(T*));
    //in passed comparer MUST be defined cases for `NULL` values
    virtual T* getMost(bool (*comparer)(T*, T*));
    template <class F> F* fold(F* (*fold_function)(F*, T*), F* start_value);
    void showTree();
protected:
    virtual void findAllHelper(bool (*predicate)(T*), Node<T>* current, List<T>* result);
    virtual void show(Node<T>* current, int level = 0);
    virtual void getMostHelper(bool (*comparer)(T*, T*), Node<T>* current, T* most);
    template <class F> F* foldHelper(F* (*fold_function)(F*, T*), F* value, Node<T>* current);
public:
    ~Tree();
};

template <class T>
class SortedTree : public Tree<T> {
public:
    void add(T* element) override;
    NWP<T>* find(T* compare_object);
    void remove(bool (*predicate)(T*), Node<T>* (*normalization)(Node<T>*), bool recursively = false) override;
private:
    void removeImplementation(Node<T>* iterator, bool (*predicate)(T*), Node<T>* (*normalization)(Node<T>*), bool recursively);
public:
    ~SortedTree();
};