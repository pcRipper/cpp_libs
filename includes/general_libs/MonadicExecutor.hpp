#pragma once

#include <functional>

template<typename Context>
class MonadicExecutor {
public:
    using Function = std::function<void(Context&)>;
    using ErrorHandler = std::function<void(Context&, const std::exception&)>;
public:

    MonadicExecutor() = default;
    MonadicExecutor(MonadicExecutor const& r) = delete;
    ~MonadicExecutor() = default;

    MonadicExecutor& add(Function func, ErrorHandler onError) {
        steps.emplace_back(func, onError);
        return *this;
    }

    Context& execute(Context& context) {
        for (const auto& step : steps) {
            try {
                step.first(context);
            } catch (const std::exception& e) {
                step.second(context, e);
                break;
            }
        }

        steps.clear();

        return context;
    }

private:
    std::vector<std::pair<Function, ErrorHandler>> steps;
};