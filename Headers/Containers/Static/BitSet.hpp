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
        max_offset = max(max_offset, position);
        bits[position / 64] |= (1ull << (position % 64));
    }

    inline void unset(int position) {
        static const uint64_t MASK = 0xFFFFFFFFFFFFFFFF;
        if(position >= size)return;
        bits[position / 64] &= (MASK ^ (1ull << (position % 64)));
    }

    inline void reset(){
        memset(bits, 0, max_offset/8 + 1);
    }

    BitSet<size>& operator &&(BitSet<size>& right) {
        BitSet<size> result;

        for (int i = 0; i < ARRAY_SIZE; ++i) {
            result[i] = bits[i] & right.bits[i];
        }

        return result;
    }

    BitSet<size>& operator ||(BitSet<size>& right) {
        BitSet<size> result;

        for (int i = 0; i < ARRAY_SIZE; ++i) {
            result[i] = bits[i] | right.bits[i];
        }

        return result;
    }

    void operator |=(BitSet<size>& right) {
        for (int i = 0; i < ARRAY_SIZE; ++i) {
            bits[i] |= right.bits[i];
        }
    }

    bool operator ==(const BitSet<size>& right) {

        for (int i = 0; i < ARRAY_SIZE; ++i) {
            if (bits[i] != right[i])return false;
        }

        return true;
    }
};