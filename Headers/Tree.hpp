#pragma once
#include "List.h"
#include "includes.h"

#pragma region Node<T>

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
    Node(
        T& element,
        Node<T>* prev = nullptr,
        Node<T>* left = nullptr,
        Node<T>* right = nullptr) :
        element(new T(element)),
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

template<class T>
Node<T>::Node(Node<T>* obj) {
    if (obj != nullptr) {
        element = new T(*obj->element);
        left = obj->left;
        right = obj->right;
        prev = obj->prev;
        /*       left = prev = right = nullptr;*/
    }
}

template<class T>
Node<T>::Node(Node<T> const& obj) {
    element = new T(obj.element);
    left = obj.left;
    right = obj.right;
    prev = obj.prev;
    //left = right = prev = nullptr;
}

template <class T>
std::ostream& operator << (std::ostream& out, Node<T>* element) {
    if (element == nullptr || element->element == nullptr)out << "__";
    else out << *(element->element);
    return(out);
}

template <class T>
std::ostream& operator << (std::ostream& out, Node<T>& element) {
    if (element.element == nullptr)out << "__";
    else out << *element.element;
    return(out);
}

template <class T>
bool Node<T>::operator==(Node<T> const& element) {
    return *this->element == *element.element;
}

template <class T>
Node<T>::~Node() {
    prev = left = right = nullptr;
    delete element;
}

#pragma endregion

#pragma region NWP<T>

template <class T>
struct NWP {
    Node<T>* element;
    std::string path;
    NWP(std::string path = "", Node<T>* element = nullptr) :path(path), element(element) {};
    ~NWP() {
        element = nullptr;
    }
};

#pragma endregion

#pragma region Tree<T>

template <class T>
class Tree {
protected:
    Node<T>* root, * current, * previous;
public:
    Tree();
    Node<T>* getRoot();
    void setRoot(Node<T>* root);
    virtual string add(T* element) = 0;
    virtual string add(T& element) = 0;
    virtual void remove(function<bool(T*)> predicate, function<Node<T>* (Node<T>*)> normalization, bool recursively = false) = 0;
    int countNodes(Node<T>* current);
    int countLeafes(Node<T>* current);
    int countHeight(Node<T>* current, int height = 0);
    List<T>* findAll(function<bool(T*)> predicate);
    virtual List<T>* toList();
    T* getMost(bool (*comparer)(T*, T*));
    template <class F> F* fold(void (*fold_function)(F*, T*), F* start_value);
    virtual void show();
    virtual void show_pointers();
protected:
    void findAllHelper(function<bool(T*)> predicate, Node<T>* current, List<T>* result);
    virtual void showTree(Node<T>* current, int level = 0);
    virtual void showTreePointers(Node<T>* current, List<int> levels, int level = 0);
    virtual void getMostHelper(bool (*comparer)(T*, T*), Node<T>* current, T*& most);
    virtual void getAllHelper(Node<T>* current, List<T>*& list);
    template <class F> void foldHelper(void (*fold_function)(F*, T*), F* value, Node<T>* current);
public:
    ~Tree();
};

template <class T>
Tree<T>::Tree() :root(nullptr), current(nullptr), previous(nullptr) {};

template <class T>
Tree<T>::~Tree() {}

template <class T>
Node<T>* Tree<T>::getRoot() {
    return root;
}

template <class T>
void Tree<T>::setRoot(Node<T>* root) {
    if (root != nullptr) {
        this->root = new Node<T>(*root);
    }
}

template<class T>
int Tree<T>::countNodes(Node<T>* current)
{
    if (current != nullptr) {
        return 1 + countNodes(current->left) + countNodes(current->right);
    }
    return 0;
}

template<class T>
int Tree<T>::countLeafes(Node<T>* current)
{
    if (current != nullptr) {
        if (current->left == nullptr && current->right == nullptr)return 1;
        return countLeafes(current->left) + countLeafes(current->right);
    }
    return 0;
}

template<class T>
int Tree<T>::countHeight(Node<T>* current, int height)
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
void Tree<T>::foldHelper(void (*fold_function)(F*, T*), F* value, Node<T>* current)
{
    if (current != nullptr) {
        fold_function(value, current->element);
        foldHelper(fold_function, value, current->left);
        foldHelper(fold_function, value, current->right);
    }
}

template<class T>
void Tree<T>::getMostHelper(bool (*comparer)(T*, T*), Node<T>* current, T*& most)
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
void Tree<T>::getAllHelper(Node<T>* current, List<T>*& list)
{
    if (current != nullptr) {
        list->add(*current->element);
        getAllHelper(current->left, list);
        getAllHelper(current->right, list);
    }
}

template<class T>
void Tree<T>::findAllHelper(function<bool(T*)> predicate, Node<T>* current, List<T>* result)
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
void Tree<T>::showTree(Node<T>* current, int level) {
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
void Tree<T>::showTreePointers(Node<T>* current, List<int> levels, int level)
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

#pragma region SortedTree<T>

template <class T>
class SortedTree :
    public Tree<T> {
public:
    SortedTree();
    string add(T* element) override;
    string add(T& element) override;
    NWP<T>* find(T* compare_object);
    void remove(function<bool(T*)> predicate, function<Node<T>* (Node<T>*)> normalization, bool recursively = false) override;
    virtual List<T>* toList() override;
    virtual void balanceHeight();
protected:
    void removeImplementation(Node<T>* iterator, function<bool(T*)> predicate, function<Node<T>* (Node<T>*)> normalization, bool recursively);
    virtual void getAllHelper(Node<T>* current, List<T>*& list) override;
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
            data->element->right = new Node<T>(element, data->element);
        }
        else {
            data->element->left = new Node<T>(element, data->element);
        }

        path = data->path;
    }
    else {
        Tree<T>::root = new Node<T>(element);
    }

    delete data;
    return(path);
}

template <class T>
string SortedTree<T>::add(T& element) {
    return add(new T(element));
}

template <class T>
void SortedTree<T>::remove(function<bool(T*)> predicate, function<Node<T>* (Node<T>*)> normalization, bool recursively) {
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
void SortedTree<T>::removeImplementation(Node<T>* iterator, function<bool(T*)> predicate, function<Node<T>* (Node<T>*)> normalization, bool recursively) {
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
void SortedTree<T>::getAllHelper(Node<T>* current, List<T>*& list)
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

#pragma endregion