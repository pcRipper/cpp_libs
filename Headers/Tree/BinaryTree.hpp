#pragma once
#include "Tree.hpp"

template <class T>
class BinaryTree : public Tree<T> {
public:
    BinaryTree():Tree<T>(){}
    virtual string add(T* element) override;
    virtual string add(T& element) override;
    virtual void remove(function<bool(T*)> predicate, function<TreeNode<T>* (TreeNode<T>*)> normalization, bool recursively = false) override;
};

template <class T>
string BinaryTree<T>::add(T* element) {
    return string();
}

template <class T>
inline string BinaryTree<T>::add(T &element)
{
    return string();
}

template <class T>
inline void BinaryTree<T>::remove(function<bool(T *)> predicate, function<TreeNode<T> *(TreeNode<T> *)> normalization, bool recursively)
{

}