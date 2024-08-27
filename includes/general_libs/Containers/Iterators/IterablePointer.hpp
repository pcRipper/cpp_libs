#include "ContinuousIterator.hpp"
#include "ReversedContinuousIterator.hpp"

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
    //Forward Iterator
    using ForwardIterator = ContinuousIterator<ContainerType>;
    //Reversed Iterator
    using ReversedIterator = ReversedContinuousIterator<ContainerType>;
private:
    Type* array;
public:
    IterablePointer() = delete;

    IterablePointer(Type* ptr):
        array(ptr)
    {};

    ForwardIterator begin(){
        return ForwardIterator(array);
    }

    ForwardIterator end(){
        return ForwardIterator(array + size);
    }

    ReversedIterator rbegin(){
        return ReversedIterator(array + size - 1);
    }

    ReversedIterator rend(){
        return ReversedIterator(array - 1);
    }
};