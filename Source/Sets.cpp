#pragma once

template <typename T>
void showArray(int size, T* arr) {
    for (int k = 0; k < size; k++)std::cout << arr[k] << ((k + 1 < size) ? "," : "");
    std::cout << std::endl;
}

template <typename TArr>
class Set {
    class SubSet {
    public:
        int size;
        TArr* arr;
        SubSet* next;

        SubSet(int size = 0, TArr* arr = nullptr, SubSet* next = nullptr) {
            if (size > 0 && arr != nullptr) {
                this->size = size;
                this->arr = arr;
                this->next = next;
            }
        }
    };
    SubSet* head = nullptr, * current = nullptr, * previousAdded = nullptr;
public:
    SubSet* operator[](int position) {
        if (head != nullptr) {
            int k = 0;
            for (current = head; current != nullptr && k < position; k++, current = current->next);
            if (current != nullptr && k == position)return(current);
            else return(nullptr);
        }
    }
    unsigned __int64 count = 0;
    int getRamSize(int size);
    void showObjectSize();
    bool includes(int size, TArr* arr, SubSet* object);
    unsigned __int64 add(int size, TArr* arr, bool check);
    void show(SubSet* obj);
    void drop(SubSet* startObject);
    ~Set() {
        drop(head);
    }
};

template <typename TArr>
int Set<TArr>::getRamSize(int size) {
    return(sizeof(TArr) * size + sizeof(int));
}

template <typename TArr>
void Set<TArr>::showObjectSize() {
    unsigned __int64 RASetize = 0;
    if (head != nullptr) {
        for (current = head; current != nullptr; current = current->next)RASetize += this->getRamSize(current->size);
    }
    unsigned __int64 mB = RASetize / 1048576;
    unsigned __int64 kB = RASetize % 1048576 / 1024;
    unsigned __int64 B = RASetize % 1024;
    std::cout << RASetize << " = " << mB << "," << kB << "," << B << "\n\n";
}

template <typename TArr>
bool Set<TArr>::includes(int size, TArr* arr, SubSet* object) {
    if (object != nullptr && object->size > 0 && object->arr != nullptr && size > 0 && arr != nullptr && size == object->size) {

        TArr* used = new TArr[object->size];
        bool included = false;
        int counter = 0;

        for (int k = 0, c, o; k < size; k++) {
            for (c = 0; c < object->size; c++, included = false) {

                for (o = 0; o < counter; o++) {
                    if (used[o] == c) {
                        included = true;
                        break;
                    }
                }

                if (arr[k] == object->arr[c] && !included) {
                    used[counter] = c;
                    counter++;
                    included = false;
                    break;
                }
            }
        }

        delete[] used;
        return((counter == object->size) ? 1 : 0);
    }
    return 0;
}

template <typename TArr>
unsigned __int64 Set<TArr>::add(int size, TArr* arr, bool check) {
    if (head == nullptr) {
        head = new SubSet(size, arr);
    }
    else {
        if (previousAdded != nullptr && !check)current = previousAdded;
        else {
            current = head;
            for (; current->next != nullptr; current = current->next) {
                if (check && includes(size, arr, current))return 0;
            }
            if (current == head && check && includes(size, arr, head))return 0;
        }
        previousAdded = current->next = new SubSet(size,arr);
    }
    count++;
    return(count);
}

template <typename TArr>
void Set<TArr>::show(SubSet* obj) {
    if (head != nullptr) {
        if (obj == nullptr) {
            for (current = head; current != nullptr; current = current->next) {
                std::cout << current << "->" << current->size << "(" << this->getRamSize(current->size) << "): ";
                for (int k = 0; k < current->size; k++)std::cout << current->arr[k] << ((k + 1 == current->size) ? "" : ", ");
                std::cout << std::endl;
            }
        }
        else {
            std::cout << obj << "->" << obj->size << ":";
            for (int k = 0; k < obj->size; k++)std::cout << obj->arr[k] << ", ";
        }
        std::cout << std::endl;
    }
}

template <typename TArr>
void Set<TArr>::drop(SubSet* startObject) {
    if (startObject != nullptr){
        for (current = startObject->next;current != nullptr;startObject = current,current = current->next)delete startObject;
        delete startObject;
    }
}

template <typename TArr>
void setSubs(Set<TArr>& object, int size, TArr* arr,int end,int start = 1) {
    if (size > 0 && arr != nullptr && 0 < start && start <= size && 0 < end && end <= size && start <= end) {
        unsigned __int64 RASetize = 0;
        int* numArray;
        TArr* newArray;
        for (int currentSize = start; currentSize <= end; currentSize++) {

            numArray = new int[currentSize];

            for (int m = 0; m < currentSize; m++)numArray[m] = m;

            for (int last = currentSize - 1; true; numArray[last]++) {
                newArray = new TArr[currentSize];
                for (int m = 0; m < currentSize; m++)newArray[m] = arr[numArray[m]];

                object.add(currentSize, newArray, 0);

                if (numArray[0] == size - currentSize) {
                    break;
                }

                if (size - 1 <= numArray[last] && currentSize > 1) {
                    for (int k = last, rank = size - 1; 0 < k; k--, rank--) {
                        if (rank <= numArray[k]) {
                            numArray[k - 1]++;
                            for (int k2 = k, startRank = numArray[k - 1] + 1; k2 < currentSize; k2++, startRank++)numArray[k2] = startRank;
                            if (numArray[k - 1] - 1 < rank - 1)break;
                        }
                    }
                    numArray[last]--;
                }

            }
            delete numArray;
        }
    }
}

size_t setSubs2(int* arr, size_t size, size_t end, size_t start = 1) {
    size_t count = 0;

    for (int currentSize = start; currentSize <= end; currentSize++) {

        int* numArray = new int[currentSize];

        for (int m = 0; m < currentSize; m++)numArray[m] = m;

        for (int last = currentSize - 1; true; numArray[last]++) {

            int* newArray = new int[currentSize];
            for (int m = 0; m < currentSize; m++)newArray[m] = arr[numArray[m]];

            count++;

            showArray<int>(newArray, currentSize);

            if (numArray[0] == size - currentSize) {
                break;
            }


            if (currentSize > 1 && size - 1 <= numArray[last]) {
                int index = currentSize;
                int rank = size - 1;

                for (int k = currentSize - 1; 0 < k; k--) {

                    if (rank <= numArray[k]) {
                        numArray[k - 1]++;
                        index = k;
                        if (numArray[k - 1] != rank)break;
                    }

                    rank--;
                }

                for (int k = index; k < currentSize; k++) {
                    numArray[k] = numArray[k - 1] + 1;
                }

                if (index != currentSize)numArray[last]--;

            }

        }

        delete numArray;
    }

    return count;
}