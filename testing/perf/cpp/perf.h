#ifndef PERF_PERF_H
#define PERF_PERF_H


template<typename Func, typename... Args>
void benchmark(Func func, int max_runs = 1000, int time_limit = 5,
               const std::string &test_name = "func", Args &&... args) {
    using namespace std::chrono;

    auto start_time = high_resolution_clock::now();
    int run_count = 0;

    for (int i = 0; i < max_runs; ++i) {
        auto current_time = high_resolution_clock::now();
        auto elapsed = duration_cast<seconds>(current_time - start_time).count();

        if (elapsed >= time_limit) {
            break;
        }

        func(std::forward<Args>(args)...);
        ++run_count;
    }

    auto end_time = high_resolution_clock::now();
    auto total_runtime = duration_cast<milliseconds>(end_time - start_time).count();
    double average_duration = static_cast<double>(total_runtime) / run_count;

    std::cout << "[" << test_name << "]\tTotal runtime: " << total_runtime
              << "ms   nRuns: " << run_count
              << "    " << average_duration << "ms/call" << std::endl;
}

#endif //PERF_PERF_H
