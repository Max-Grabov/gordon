#pragma once

#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>

namespace Gordon
{
template <typename T> class ThreadSafeQueue
{
private:
  mutable std::mutex mutex_;
  std::queue<T> position_data_queue_;
  std::condition_variable condition_;

public:
  ThreadSafeQueue(){};

  ThreadSafeQueue(const ThreadSafeQueue &other)
  {
    std::lock_guard<std::mutex> lk(other.mutex_);
    position_data_queue_ = other.position_data_queue_;
  }

  // For simplicity sake, = is not defined.
  ThreadSafeQueue &operator=(const ThreadSafeQueue &) = delete;

  void push(T val)
  {
    std::lock_guard<std::mutex> lk(mutex_);
    position_data_queue_.push(val);
    condition_.notify_one();
  }

  std::shared_ptr<T> wait_and_pop()
  {
    std::unique_lock<std::mutex> lk(mutex_);
    condition_.wait(lk, [this] { return !position_data_queue_.empty(); });
    std::shared_ptr<T> res(std::make_shared<T>(position_data_queue_.front()));
    position_data_queue_.pop();
    return res;
  }

  bool empty() const
  {
    std::lock_guard<std::mutex> lk(mutex_);
    return position_data_queue_.empty();
  }
};
}
