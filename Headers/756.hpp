#include <string>
#include <vector>
#include <unordered_map>
#include <iostream>
#include <format>

using namespace std;

class Solution {
public:
    bool pyramidTransition(string line, vector<string> const& allowed)
    {
        unordered_map<char, char> allowed_map;

        for(const string& triplet : allowed)
        {
            allowed_map[toBit(triplet[0]) | toBit(triplet[1])] |= toBit(triplet[2]);
        }


        for(char &c : line)
        {
            c = toBit(c);
        }

        while(line.length() > 1)
        {
            for(int i = 1; i < line.length(); ++i)
            {
                char byte = line[i - 1];
                line[i - 1] = 0;

                if(line[i] == 0 || byte == 0) 
                {
                    return false;
                }

                for(int t = 0; t < 7; ++t)
                {
                    if((byte & (1 << t)) == 0)
                    {
                        continue;
                    }

                    for(int b = 0; b < 7; ++b)
                    {
                        if((line[i] & (1 << b)) == 0)
                        {
                            continue;
                        }

                        int key = (1 << b) | (1 << t);
                        if(allowed_map.count(key) == 0)
                        {
                            continue;
                        }

                        line[i - 1] |= allowed_map[key];
                    }
                }
            }
            line.pop_back();
            
            //debug
            for(char c : line)
            {
                string can_output;
                for(int i = 0; i < 8; ++i)
                {
                    if((1 << i) & c)
                    {
                        can_output += 'A' + i;
                    }
                }
                cout << format("[{}] ", can_output);
            }
            cout << "\n";
        }

        return line[0] != 0;
    }

    static char toBit(char c)
    {
        return 1 << (c - 'A');
    }
};
