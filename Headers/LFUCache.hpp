#pragma once

//460. LFU Cache

#include <vector>
#include <unordered_map>

template <class Type>
struct ListNode {
    ListNode* next;
    ListNode* prev;
    Type data;

    ListNode(
        Type data, 
        ListNode* next = 0, 
        ListNode* prev = 0
    ):
        value(data), 
        next(next), 
        prev(prev)
    {};

    ~ListNode() {
        next = prev = 0;
    };
};

template <class Type>
class List{
public:
    List() {
        head = tail = nullptr;
        size = 0;
    }

    void delete_at(ListNode<Type>* element){
        if(element->prev == element->next){
            head = tail = 0;
        }
        if(element->prev == nullptr){
            head = element->next;
            element->next->prev = nullptr;
        }
        else if(element->next == nullptr){
            tail = element->prev;
            tail->next = nullptr;
        }
        else {
            element->prev->next = element->next;
            element->next->prev = element->prev;
        }
        size -= 1;
    }

    void push_front(Type element){
        if(head == 0){
            head = tail = new ListNode<Type>(element);
        }
        else {
            head = new ListNode<Type>(element, head);
            head->next->prev = head;
        }
        size += 1;
    }

    void push_back(Type element){
        if(head == 0){
            head = tail = new ListNode<Type>(element);
        }
        else {
            tail = new ListNode<Type>(element, 0, tail);
            tail->prev->next = tail;
        }
        size += 1;
    }

    void push_after(ListNode<Type>* current, Type element){
        if(current == nullptr)return;
        if(current->next == 0){
            return push_back(element);
        }

        auto next = current->next;
        current->next = new ListNode<Type>(element, next, current);
        next->prev = current->next;

        size += 1;
    }

    void push_before(ListNode<Type>* current, Type element){
        if(current == 0)return;
        if(current->prev == 0){
            return push_front(element);
        }

        auto prev = current->prev;
        current->prev = new ListNode<Type>(element, current, prev);
        prev->next = current->prev;

        size += 1;
    }

    ~List() {
        ListNode<Type>* itr = head;
        while(itr != 0){
            ListNode<Type> next = itr->next;
            delete itr;
            itr = next;
        }
        head = tail = 0;
    }

public:   
    ListNode<Type>* head;
    ListNode<Type>* tail;
    int size;
};


class LFUCache {
public:
    LFUCache(int capacity) {
        capacity = capacity;
    }
    
    int get(int key) {
        auto itr = values.find(key);

        if(itr == values.end())return -1;

        auto&[freq_list, key_list] = positions[key];
        int current_frequency      = freq_list->data.first;

        freq_list->data.second.delete_at(key_list);
        
        if(freq_list->next->data.first != current_frequency + 1 || freq_list->next == nullptr){
            list.push_after(freq_list, {current_frequency + 1, {}});
        }

        freq_list->next->data.second.push_back(key);
        positions[key] = {freq_list->next, freq_list->next->data.second.tail};

        if(freq_list->data.second.size == 0){
            list.delete_at(freq_list);
        }

        return itr->second;
    }
    
    void put(int key, int value) {
        auto itr = values.find(key);

        if(itr == values.end()){
            if(list.tail == nullptr || list.tail->data.first != 1){
                list.push_back({1, {}});
            }
            list.tail->data.second.push_front(key);
        }
    }

private:
    std::unordered_map<int, int> values;
    std::unordered_map<int, std::pair<ListNode<std::pair<int, List<int>>>*, ListNode<int>*>> positions;
    int capacity;
    List<std::pair<int, List<int>>> list;
};

/**
 * Your LFUCache object will be instantiated and called as such:
 * LFUCache* obj = new LFUCache(capacity);
 * int param_1 = obj->get(key);
 * obj->put(key,value);
 */