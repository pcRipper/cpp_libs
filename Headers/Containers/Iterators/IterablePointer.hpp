#include "ContinuousIterator.hpp"

/// @brief Wrapper, that allows to easily iterate the simple pointer arrays
/// @tparam Type is the type of pointer array
/// @tparam size is the size of the pointer array
template <
    class Type,
    int size
>
class IterablePointer {
public:
    using ValueType = Type;
    using ContainerType = IterablePointer<Type, size>;
    using Iterator = ContinuousIterator<ContainerType>;
private:
    Type* array;
public:
    IterablePointer() = delete;

    IterablePointer(Type* ptr):
        array(ptr)
    {};

    Iterator begin(){
        return Iterator(array);
    }

    Iterator end(){
        return Iterator(array + size);
    }
};