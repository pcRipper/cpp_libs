#pragma once

#include <fstream>
#include <string>

class Writer{
    std::fstream stream;
public:
    Writer(const std::string file_path, const std::ios_base::openmode flags){
        stream = std::fstream(file_path, flags);
    }

    void write(const std::string& text){
        if(! stream.is_open())return;
        
        stream.write(&text[0], text.size());
    }

    ~Writer(){
        if(stream.is_open()){
            stream.close();
        }
    }
};