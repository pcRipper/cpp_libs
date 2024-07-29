#include "../../../Headers/Containers/Dynamic/Sets.hpp"

size_t setSubs2(int* arr, size_t size, size_t end, size_t start) {
    size_t count = 0;

    for (int currentSize = start; currentSize <= end; currentSize++) {

        int* numArray = new int[currentSize];

        for (int m = 0; m < currentSize; m++)numArray[m] = m;

        for (int last = currentSize - 1; true; numArray[last]++) {

            int* newArray = new int[currentSize];
            for (int m = 0; m < currentSize; m++)newArray[m] = arr[numArray[m]];

            count++;

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