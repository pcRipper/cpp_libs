#pragma once

#include <thread>
#include <atomic>
#include <functional>
#include <array>
#include <vector>
#include <condition_variable>
#include <memory>
#include <deque>

template <size_t threads_count>
class ThreadPool : public std::enable_shared_from_this<ThreadPool<threads_count>>
{
public:
    using ClassType = ThreadPool<threads_count>;

    // Static factory method to create instances
    static std::shared_ptr<ClassType> create(){
        std::shared_ptr<ClassType> instance(new ClassType());
        instance->start();
        return instance;
    }

    ~ThreadPool(){
        stop();
    }

    void push(std::function<void()> task){
        {
            std::unique_lock<std::mutex> lock(task_mutex);
            task_pool.push_back(task);
        }
        task_condition.notify_one();
    }

    bool isRunning() const {
        return run;
    }

    bool isIdle(){
        std::unique_lock<std::mutex> lock(task_mutex);
        return idle_count == threads_count && task_pool.empty();
    }

    void stop(){
        {
            std::unique_lock<std::mutex> lock(task_mutex);
            run = false;
        }
        task_condition.notify_all();
        for(auto& th : thread_pool){
            if(th.joinable()){
                th.join();
            }
        }
    }

protected:
    ThreadPool() : run(true), idle_count(0) {}

    void start(){
        for(size_t i = 0; i < threads_count; ++i){
            thread_pool[i] = std::thread(&ClassType::pool_worker, this);
        }
    }

    void pool_worker(){
        while(run){
            idle_count += 1;
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(task_mutex);
                task_condition.wait(lock, [this]{
                    return !task_pool.empty() || !run;
                });

                if(!run && task_pool.empty()){
                    return;
                }

                task = task_pool.front();
                task_pool.pop_front();
            }
            idle_count -= 1;
            task();
        }
    }

protected:
    std::array<std::thread, threads_count> thread_pool;
    std::deque<std::function<void()>> task_pool;
    std::atomic<bool> run;
    std::atomic<size_t> idle_count;

    std::mutex task_mutex;
    std::condition_variable task_condition;
};

#include <utility>

namespace ThPool{
    /// @brief wraps a function with any context to a void-returning 0 parameters
    /// @tparam Func 
    /// @tparam ...Args 
    /// @param func 
    /// @param ...args 
    /// @return 
    template<typename Func, typename... Args>
    std::function<void()> wrap(Func&& func, Args&&... args) {
        return [func = std::forward<Func>(func), ... args = std::forward<Args>(args)]() {
            std::invoke(func, args...);
        };
    }
}
