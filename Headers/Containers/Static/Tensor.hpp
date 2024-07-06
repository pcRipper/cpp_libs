#pragma once

template <class Type, size_t dimensions>
class Tensor{
public:
    using ClassType = Tensor<Type, dimensions>;
public:
    Tensor() = delete;

    Tensor(array<size_t, dimensions> const dim_sizes):
        isNested(false)
    {
        sizes = new size_t[dimensions];
        data_size = 1;
        for(int i = 0; i < dimensions; ++i){
            sizes[i] = dim_sizes[i];
            data_size *= dim_sizes[i];
        }

        data = new Type[data_size];
    }

protected:
    static ClassType* prepareNested(array<size_t, dimensions> const sizes){
        auto res = (ClassType*)malloc(sizeof(ClassType));
        res->isNested = true;
        return res;
    }
public:

    ClassType& operator [](size_t index) const {
        if(dimensions == 0)return *this;
        if(index >= sizes[0]){
            throw exception("Indexing error: out of bounds");
        }

        auto nested = Tensor<Type, dimensions - 1>::prepareNested();
        nested->sizes = this->sizes + 1;
        nested->data  = &this->data[index * sizes[0]];
        nested->data_size = this->data_size / sizes[0];

        return *nested;
    }

    bool isElement() const {
        return ClassType::isSingleElement;
    }

    Type& getElement(){
        if(isElement()){
            return *data;
        }

        throw exception("Access error : misleaded type");
    }

    ~Tensor(){
        if(isNested)return;
        delete[] sizes;
        delete[] data;
    }
protected:
    static constexpr bool isSingleElement = dimensions == 0;
protected:
    bool isNested;
protected:
    Type* data;
    size_t data_size;
    size_t* sizes;
};