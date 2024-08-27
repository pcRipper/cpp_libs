#pragma once
#include "../includes.hpp"
#include "List.hpp"
#include "TreeNode.hpp"

#pragma region NWP<T>

template <class T>
struct NWP {
    TreeNode<T>* element;
    std::string path;
    NWP(std::string path = "", TreeNode<T>* element = nullptr) :path(path), element(element) {};
    ~NWP() {
        element = nullptr;
    }
};

#pragma endregion

#pragma region Tree<T>

template <class T>
class Tree {
protected:
    TreeNode<T>* root, * current, * previous;
public:
    Tree();
    TreeNode<T>* getRoot();
    void setRoot(TreeNode<T>* root);
    virtual string add(T* element) = 0;
    virtual string add(T& element) = 0;
    virtual void remove(function<bool(T*)> predicate, function<TreeNode<T>* (TreeNode<T>*)> normalization, bool recursively = false) = 0;
    int countTreeNodes(TreeNode<T>* current);
    int countLeafes(TreeNode<T>* current);
    int countHeight(TreeNode<T>* current, int height = 0);
    List<T>* findAll(function<bool(T*)> predicate);
    virtual List<T>* toList();
    T* getMost(bool (*comparer)(T*, T*));
    template <class F> F* fold(void (*fold_function)(F*, T*), F* start_value);
    virtual void show();
    virtual void show_pointers();
protected:
    void findAllHelper(function<bool(T*)> predicate, TreeNode<T>* current, List<T>* result);
    virtual void showTree(TreeNode<T>* current, int level = 0);
    virtual void showTreePointers(TreeNode<T>* current, List<int> levels, int level = 0);
    virtual void getMostHelper(bool (*comparer)(T*, T*), TreeNode<T>* current, T*& most);
    virtual void getAllHelper(TreeNode<T>* current, List<T>*& list);
    template <class F> void foldHelper(void (*fold_function)(F*, T*), F* value, TreeNode<T>* current);
public:
    ~Tree();
};

template <class T>
Tree<T>::Tree() :root(nullptr), current(nullptr), previous(nullptr) {};

template <class T>
Tree<T>::~Tree() {}

template <class T>
TreeNode<T>* Tree<T>::getRoot() {
    return root;
}

template <class T>
void Tree<T>::setRoot(TreeNode<T>* root) {
    if (root != nullptr) {
        this->root = new TreeNode<T>(*root);
    }
}

template<class T>
int Tree<T>::countTreeNodes(TreeNode<T>* current)
{
    if (current != nullptr) {
        return 1 + countTreeNodes(current->left) + countTreeNodes(current->right);
    }
    return 0;
}

template<class T>
int Tree<T>::countLeafes(TreeNode<T>* current)
{
    if (current != nullptr) {
        if (current->left == nullptr && current->right == nullptr)return 1;
        return countLeafes(current->left) + countLeafes(current->right);
    }
    return 0;
}

template<class T>
int Tree<T>::countHeight(TreeNode<T>* current, int height)
{
    if (current != nullptr) {
        return max(height + 1, max(countHeight(current->left, height + 1), countHeight(current->right, height + 1)));
    }
    return height;
}

template<class T>
List<T>* Tree<T>::findAll(function<bool(T*)> predicate)
{
    List<T>* result = new List<T>();

    findAllHelper(predicate, root, result);

    return result;
}

template<class T>
List<T>* Tree<T>::toList()
{
    List<T>* result = new List<T>();
    getAllHelper(root, result);
    return result;
}

template<class T>
T* Tree<T>::getMost(bool(*comparer)(T*, T*))
{
    T* result = nullptr;
    getMostHelper(comparer, root, result);
    //if (result != nullptr) {
    //    result = new T(result);
    //}
    return result;
}

template<class T>
template<class F>
inline F* Tree<T>::fold(void (*fold_function)(F*, T*), F* start_value)
{
    foldHelper(fold_function, start_value, root);
    return start_value;
}

template<class T>
template<class F>
void Tree<T>::foldHelper(void (*fold_function)(F*, T*), F* value, TreeNode<T>* current)
{
    if (current != nullptr) {
        fold_function(value, current->element);
        foldHelper(fold_function, value, current->left);
        foldHelper(fold_function, value, current->right);
    }
}

template<class T>
void Tree<T>::getMostHelper(bool (*comparer)(T*, T*), TreeNode<T>* current, T*& most)
{
    if (current != nullptr) {
        if (most == nullptr)most = current->element;
        else if (current->element != nullptr) {
            if (comparer(most, current->element))most = current->element;
        }
        getMostHelper(comparer, current->left, most);
        getMostHelper(comparer, current->right, most);
    }
}

template<class T>
void Tree<T>::getAllHelper(TreeNode<T>* current, List<T>*& list)
{
    if (current != nullptr) {
        list->add(*current->element);
        getAllHelper(current->left, list);
        getAllHelper(current->right, list);
    }
}

template<class T>
void Tree<T>::findAllHelper(function<bool(T*)> predicate, TreeNode<T>* current, List<T>* result)
{
    if (current != nullptr) {
        if (predicate(current->element)) {
            result->add(current->element);
        }

        findAllHelper(predicate, current->left, result);
        findAllHelper(predicate, current->right, result);
    }
}

template <class T>
void Tree<T>::show() {
    this->showTree(root, 0);
}

template<class T>
void Tree<T>::show_pointers()
{
    this->showTreePointers(root, *new List<int>(), 0);
}

template <class T>
void Tree<T>::showTree(TreeNode<T>* current, int level) {
    if (current != nullptr) {
        showTree(current->right, level + 1);
        //std::cout << std::string(level * 3, ' ') << level + 1 << "|" << current << "\n\n";
        cout << std::string(level * 3, ' ');
        if (current != root)cout << "\b\b" << ((current->prev->right == current) ? char(218) : char(192)) << ">";
        std::cout << level + 1 << "|" << current << "\n";
        showTree(current->left, level + 1);
    }
}

template<class T>
void Tree<T>::showTreePointers(TreeNode<T>* current, List<int> levels, int level)
{
    if (current != nullptr) {


        showTreePointers(current->right, List<int>(), level + 1);

        cout << std::string(level * 3, ' ');
        if (current != root)cout << "\b\b" << ((current->prev->right == current) ? char(218) : char(192)) << ">";
        std::cout << level + 1 << "|" << current << "\n";

        showTreePointers(current->left, List<int>(), level + 1);

    }
}

#pragma endregion