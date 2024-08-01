#pragma once

#include <functional>

template <class Container, class Iterator>
class FunctionalContainer {
public:
    using Type = Container::ValueType;
public:

    FunctionalContainer() = delete;

    FunctionalContainer(Container* c): c(c) {};

    ~FunctionalContainer() {}

    virtual Container& mapIn(std::function<Type(Type const&)> transformer) const {
        Container result;

        auto itr_result = result.begin();
        for(Iterator itr = c->begin(), end = c->end(); itr != end; ++itr){
            *itr_result = transformer(*itr);
            ++itr_result;
        }

        return std::ref(result);
    }

    virtual void mapOn(std::function<void(Type&)> transfromer) {
        for(Iterator itr = c->begin(), end = c->end(); itr != end; ++itr){
            transfromer(*itr);
        }
    }

private:
    Container* c;
};