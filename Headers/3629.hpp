#include <string>
#include <vector>
#include <bitset>
#include <unordered_map>


using namespace std;

template <uint64_t SIZE>
constexpr bitset<SIZE> sieve()
{
    bitset<SIZE> result;

    for (int i = 2; i < SIZE; ++i) {
        if (result[i]) {
            continue;
        }
        for (int j = i * 2; j < SIZE; j += i) {
            result[j] = true;
        }
    }

    return result;
}

class Solution {
public:
    int minJumps(vector<int> const& nums)
    {
        static auto notPrimes = sieve<MAX_VALUE>(); 

        static bitset<MAX_VALUE> primeVisited;
        primeVisited.reset();

        unordered_map<int, deque<int>> memo;

        uint64_t maxValue = nums[0];
        const int SIZE = nums.size();
        for(int i = 0; i < SIZE; ++i) {
            memo[nums[i]].push_back(i);
            maxValue = max(maxValue, uint64_t(nums[i]));
        }

        vector<int> steps(SIZE, INT_MAX);
        deque<pair<int, int>> queue;
        queue.push_back({0, 0});
        
        while(!queue.empty()) {
            const auto[pos, cost] = queue.front(); queue.pop_front();
            
            if (steps[pos] <= cost) {
                continue;
            }
            steps[pos] = cost;
            if (pos == SIZE - 1)
            {
                break;
            }
            
            if (pos != 0) {
                queue.emplace_back(std::make_pair(pos - 1, cost + 1));
            }
            queue.emplace_back(std::make_pair(pos + 1, cost + 1));
            if (notPrimes[nums[pos]] || primeVisited[nums[pos]]) {
                continue;
            }

            primeVisited[nums[pos]] = true;

            uint64_t currentPrime = nums[pos];
            while(currentPrime <= maxValue) {
                for(int index : memo[currentPrime]) {
                    queue.emplace_back(std::make_pair(index, cost + 1));
                }

                currentPrime += nums[pos];
            }
        }


        return steps[SIZE - 1];
    }
private:
    static constexpr size_t MAX_VALUE = 1000001;
};
