#pragma once
#include <cstdint>

template <int size>
class BitSet {
    static constexpr int ARRAY_SIZE = (size + 63) / 64;
    uint64_t bits[ARRAY_SIZE];
    int max_offset;
public:
    BitSet() {
        memset(bits, 0, ARRAY_SIZE * 8);
        max_offset = 0;
    }

    bool getBit(int position) {
        return position < size
            ? static_cast<bool>(bits[position / 64] & (1ull << (position % 64)))
            : false;
    }

    inline void set(int position) {
        if (position >= size) return;
        bits[position / 64] |= (1ull << (position % 64));
    }

    BitSet<size>& operator &&(BitSet<size>& right) {
        BitSet<size> result;

        for (int i = 0; i < ARRAY_SIZE; ++i) {
            result[i] = bits[i] & right.bits[i];
        }

        return result;
    }

    bool operator ==(const BitSet<size>& right) {

        for (int i = 0; i < ARRAY_SIZE; ++i) {
            if (bits[i] != right[i])return false;
        }

        return true;
    }
};