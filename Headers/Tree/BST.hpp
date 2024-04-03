#pragma once
#include "Tree.hpp"

template <class T>
class SortedTree :
    public Tree<T> 
{
public:
    SortedTree();
    string add(T* element) override;
    string add(T& element) override;
    NWP<T>* find(T* compare_object);
    void remove(function<bool(T*)> predicate, function<TreeNode<T>* (TreeNode<T>*)> normalization, bool recursively = false) override;
    virtual List<T>* toList() override;
    virtual void balanceHeight();
protected:
    void removeImplementation(TreeNode<T>* iterator, function<bool(T*)> predicate, function<TreeNode<T>* (TreeNode<T>*)> normalization, bool recursively);
    virtual void getAllHelper(TreeNode<T>* current, List<T>*& list) override;
    virtual void balanceHeightHelper(List<T>*& asList, int left, int right);
public:
    ~SortedTree();
};

template<class T>
SortedTree<T>::SortedTree() :Tree<T>() {

}

template <class T>
string SortedTree<T>::add(T* element) {
    NWP<T>* data = new NWP<T>();
    std::string path("root:");

    if (Tree<T>::root != nullptr) {
        data = find(element);

        if (*data->element->element < *element) {
            data->element->right = new TreeNode<T>(element, data->element);
        }
        else {
            data->element->left = new TreeNode<T>(element, data->element);
        }

        path = data->path;
    }
    else {
        Tree<T>::root = new TreeNode<T>(element);
    }

    delete data;
    return(path);
}

template <class T>
string SortedTree<T>::add(T& element) {
    return add(new T(element));
}

template <class T>
void SortedTree<T>::remove(function<bool(T*)> predicate, function<TreeNode<T>* (TreeNode<T>*)> normalization, bool recursively) {
    removeImplementation(Tree<T>::root, predicate, normalization, recursively);
}

template<class T>
List<T>* SortedTree<T>::toList()
{
    List<T>* result = new List<T>();
    SortedTree<T>::getAllHelper(Tree<T>::root, result);
    return result;
}

template<class T>
void SortedTree<T>::balanceHeight()
{
    List<T>* asList = this->toList();
    removeImplementation(Tree<T>::root, [](T* e) {return true; }, NULL, true);
    Tree<T>::root = nullptr;
    balanceHeightHelper(asList, 0, asList->Size() - 1);
    delete asList;
}

template <class T>
NWP<T>* SortedTree<T>::find(T* compare_object) {
    NWP<T>* data = new NWP<T>();

    if (Tree<T>::root != nullptr) {

        data->path = "root:";

        for (Tree<T>::current = Tree<T>::root; Tree<T>::current != nullptr;) {
            Tree<T>::previous = Tree<T>::current;

            if (*Tree<T>::previous->element == *compare_object) {
                break;
            }
            else if (*Tree<T>::current->element < *compare_object) {
                Tree<T>::current = Tree<T>::current->right;
                data->path += "r,";
            }
            else {
                Tree<T>::current = Tree<T>::current->left;
                data->path += "l,";
            }
        }
        data->element = Tree<T>::previous;

        data->path.resize(data->path.length() - 1);

    }
    return(data);
}

template <class T>
void SortedTree<T>::removeImplementation(TreeNode<T>* iterator, function<bool(T*)> predicate, function<TreeNode<T>* (TreeNode<T>*)> normalization, bool recursively) {
    if (iterator != nullptr) {
        if (predicate(iterator->element)) {

            if (recursively) {
                removeImplementation(iterator->left, [](T* e) {return true; }, NULL, true);
                removeImplementation(iterator->right, [](T* e) {return true; }, NULL, true);
                delete iterator;
                return;
            }
            else {
                Tree<T>::previous = iterator;
                iterator = normalization(iterator);
                delete Tree<T>::previous;
            }

        }

        removeImplementation(iterator->left, predicate, normalization, recursively);
        removeImplementation(iterator->right, predicate, normalization, recursively);

    }
}

template<class T>
void SortedTree<T>::getAllHelper(TreeNode<T>* current, List<T>*& list)
{
    if (current != nullptr) {
        getAllHelper(current->left, list);
        list->add(*current->element);
        getAllHelper(current->right, list);
    }
}

template<class T>
void SortedTree<T>::balanceHeightHelper(List<T>*& asList, int left, int right)
{
    if (left <= right) {
        int middle = (right + left) / 2;

        this->add(*(*asList)[middle]);
        balanceHeightHelper(asList, left, middle - 1);
        balanceHeightHelper(asList, middle + 1, right);
    }
}

template <class T>
SortedTree<T>::~SortedTree() {
    removeImplementation(Tree<T>::root, [](T* e) {return true; }, NULL, true);
    Tree<T>::root = nullptr;
}
