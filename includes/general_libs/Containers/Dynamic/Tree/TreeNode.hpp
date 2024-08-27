#pragma once
#include "../includes.hpp"

template <class T>
class TreeNode {
public:
    TreeNode<T>* left, * right, * prev;
    T* element;
    ///////
    TreeNode(
        T* element = nullptr,
        TreeNode<T>* prev = nullptr,
        TreeNode<T>* left = nullptr,
        TreeNode<T>* right = nullptr) :
        element(element),
        prev(prev),
        left(left),
        right(right)
    {};
    TreeNode(
        T& element,
        TreeNode<T>* prev = nullptr,
        TreeNode<T>* left = nullptr,
        TreeNode<T>* right = nullptr) :
        element(new T(element)),
        prev(prev),
        left(left),
        right(right)
    {};
    TreeNode(TreeNode<T>* obj);
    TreeNode(TreeNode<T> const& obj);
    template <class T> friend std::ostream& operator << (std::ostream& out, TreeNode<T>* element);
    template <class T> friend std::ostream& operator << (std::ostream& out, TreeNode<T>& element);
    bool operator == (TreeNode<T> const& element);

    static TreeNode<T>* fromLevelOrder(std::vector<string> data, function<T*(const string&)> transformer);

    ~TreeNode();
};

template<class T>
TreeNode<T>::TreeNode(TreeNode<T>* obj) {
    if (obj != nullptr) {
        element = new T(*obj->element);
        left = obj->left;
        right = obj->right;
        prev = obj->prev;
        /*       left = prev = right = nullptr;*/
    }
}

template<class T>
TreeNode<T>::TreeNode(TreeNode<T> const& obj) {
    element = new T(*obj.element);
    left = obj.left;
    right = obj.right;
    prev = obj.prev;
    //left = right = prev = nullptr;
}

template <class T>
std::ostream& operator << (std::ostream& out, TreeNode<T>* element) {
    if (element == nullptr || element->element == nullptr)out << "__";
    else out << *(element->element);
    return(out);
}

template <class T>
std::ostream& operator << (std::ostream& out, TreeNode<T>& element) {
    if (element.element == nullptr)out << "__";
    else out << *element.element;
    return(out);
}

template <class T>
bool TreeNode<T>::operator==(TreeNode<T> const& element) {
    return *this->element == *element.element;
}

template<class T>
TreeNode<T>* TreeNode<T>::fromLevelOrder(
        std::vector<string> data,
        function<T*(const string&)> transform
    ){

    if(data.size() == 0)return nullptr;
    
    auto rootValue = transform(data[0]);
    if(rootValue == nullptr)return nullptr;
    
    TreeNode<T>* root = new TreeNode<T>(*rootValue);

    deque<TreeNode<T>*> queue;
    queue.push_back(root);
    
    size_t itr = 1;
    while(queue.size() != 0){
        int size = queue.size();
        
        while(size-- > 0){
            auto top = queue.front(); queue.pop_front();
            if(itr < data.size()){
                auto value = transform(data[itr++]);
                if(value != nullptr){
                    top->left = new TreeNode<T>(*value);
                    top->left->prev = top;
                    queue.push_back(top->left);
                }
            }
            if(itr < data.size()){
                auto value = transform(data[itr++]);
                if(value != nullptr){
                    top->right = new TreeNode<T>(*value);
                    top->right->prev = top;
                    queue.push_back(top->right);
                }
            }
        }
    }

    return root;
}

template <class T>
TreeNode<T>::~TreeNode() {
    prev = left = right = nullptr;
    delete element;
}