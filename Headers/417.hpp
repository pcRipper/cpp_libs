#pragma once

#include <general_libs/includes.hpp>

constexpr int MAX_ROWS = 200;
constexpr int MAX_COLUMNS = 200;
constexpr int MAX_POINTS_COUNT = MAX_ROWS * MAX_COLUMNS;

class Solution {
public:
    vector<vector<int>> pacificAtlantic(vector<vector<int>>& heights)
    {
        const int ROWS = heights.size();
        const int COLUMNS = heights[0].size();

        deque<pair<int, int>> pacificStart, atlanticStart;
        for(int r = 0; r < ROWS; ++r)
        {
            pacificStart.push_back({r, 0});
            atlanticStart.push_back({r, COLUMNS - 1});
        }
        for(int c = 1; c + 1 < COLUMNS; ++c)
        {
            pacificStart.push_back({0, c});
            atlanticStart.push_back({ROWS - 1, c});
        }

        auto pacific = Solution::getReachablePoints(pacificStart, heights); 
        auto atlantic = Solution::getReachablePoints(atlanticStart, heights);
        
        auto same = pacific & atlantic;

        vector<vector<int>> result;
        result.reserve(same.count());

        for(int r = 0; r < ROWS; ++r)
        {
            for(int c = 0; c < COLUMNS; ++c)
            {
                if(same[r * COLUMNS + c])
                {
                    result.push_back({r, c});
                }
            }
        }

        return result;
    }

    static bitset<MAX_POINTS_COUNT> getReachablePoints(deque<pair<int,int>> queue, const vector<vector<int>>& grid)
    {   
        bitset<MAX_POINTS_COUNT> result;

        const int ROWS = grid.size();
        const int COLUMNS = grid[0].size();
        const int directions[2][4] = {
            {0, 1, 0, -1},
            {1, 0, -1, 0},
        };

        while(queue.size())
        {
            const auto[row, col] = queue.front(); queue.pop_front();
            result[row * COLUMNS + col] = true;

            for(int i = 0; i < 4; ++i)
            {
                int nRow = row + directions[0][i];
                int nCol = col + directions[1][i];

                if(nRow < 0 || nRow >= ROWS || nCol < 0 || nCol >= COLUMNS)
                {
                    continue;
                }
                if(grid[nRow][nCol] < grid[row][col])
                {
                    continue;
                }
                const int NFLAT = nRow * COLUMNS + nCol;
                if(result[NFLAT])
                {
                    continue;
                }

                queue.push_back({nRow, nCol});
                result[NFLAT] = true;
            }
        }

        return result;
    }
};
