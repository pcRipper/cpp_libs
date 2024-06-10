#pragma once

#include <deque>
#include <mutex>
#include <condition_variable>

template<typename T>
class ThreadSafeDeque {
public:
    ThreadSafeDeque() = default;

    void push_back(const T& value) {
        std::lock_guard<std::mutex> lock(mutex_);
        deque_.push_back(value);
        condition_.notify_one();
    }

    void emplace_back(const T&& value) {
        std::lock_guard<std::mutex> lock(mutex_);
        deque_.emplace_back(value);
        condition_.notify_one();
    }

    void push_front(const T& value) {
        std::lock_guard<std::mutex> lock(mutex_);
        deque_.push_front(value);
        condition_.notify_one();
    }

    T pop_back() {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [this] { return !deque_.empty(); });
        T value = deque_.back();
        deque_.pop_back();
        return value;
    }

    T front() {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [this] {return !deque_.empty(); });
        return deque_.front();
    }

    T back() {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [this] {return !deque_.empty(); });
        return deque_.back();
    }

    T pop_front() {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [this] { return !deque_.empty(); });
        T value = deque_.front();
        deque_.pop_front();
        return value;
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return deque_.empty();
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return deque_.size();
    }

private:
    std::deque<T> deque_;
    mutable std::mutex mutex_;
    std::condition_variable condition_;
};