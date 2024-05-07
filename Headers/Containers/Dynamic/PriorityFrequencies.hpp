#pragma once
#include "../../includes.hpp"


template <
    class Type,
    class Predicate = less<pair<int, Type>>
>
class PriorityFrequencies {
public:
    using PriorityType = pair<int, Type>;
private:
    priority_queue<PriorityType, vector<PriorityType>, Predicate> queue;
    unordered_map<Type, int> frequenices; 
public:
    PriorityFrequencies() = default;

    pair<int, Type> top(){
        if(queue.size() == 0)return {-1, Type()};
        
        while(true){
            auto[freq, key] = queue.top();
            
            if(freq == frequenices[key])break;
            while(queue.size() != 0 && queue.top().second == key){
                queue.pop();
            } 
            
            queue.push({frequenices[key] ,key});
        }
        
        return queue.top();
    }

    void pop(){
        if(queue.size() == 0)return;
        frequenices.erase(queue.top().second);
        queue.pop();
    }

    void push(Type key, int frequency){
        frequenices[key] += frequency;
        queue.push({frequenices[key], key});
    }

    ~PriorityFrequencies() = default;
};