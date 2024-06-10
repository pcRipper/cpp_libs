#pragma once

#include <Containers/Dynamic/ThreadSafeDeque.hpp>

#include <thread>
#include <atomic>
#include <functional>
#include <list>
#include <array>

template <size_t threads>
class ThreadPool :
    public std::enable_shared_from_this<ThreadPool<threads>>
{
public:
    ThreadPool(){
        for(size_t i = 0; i < threads; ++i){
            pool[i] = std::make_pair(
                
            );
        }
    }

    static void TP_thread(std::shared_ptr<ThreadPool> ptr){
        while(ptr->isRunnig()){

        }
    }

    bool isRunning(){
        std::lock_guard<std::mutex> lock(mute);
        return run;
    }

    void stop(){
        std::lock_guard<std::mutex> lock(mute);
        run = false;
    }
private:
    using ArrayType = std::pair<
        std::thread,
        ThreadSafeDeque<std::function<void()>>
    >;

    std::list<size_t> queue;
    std::array<ArrayType, threads> pool;
    
    std::atomic<bool> run;
    std::mutex mute;
};