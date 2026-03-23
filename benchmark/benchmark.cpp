#include <chrono>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <utility>
#include <vector>

#if defined(__APPLE__)
#include <dispatch/dispatch.h>
#include <mach/mach_host.h>
#include <mach/mach_time.h>
#include <mach/thread_act.h>
#include <mach/thread_policy.h>
#include <os/proc.h>
#include <pthread.h>
#include <sys/sysctl.h>
#elif defined(__linux__)
#include <pthread.h>
#include <sched.h>
#include <sys/sysinfo.h>
#endif

extern "C" {
int peak_sme_fmopa_1_fp32_fp32_fp32(long num_repeats);
int peak_sme_fmopa_2_fp32_fp32_fp32(long num_repeats);
int peak_sme_fmopa_3_fp32_fp32_fp32(long num_repeats);
int peak_sme_fmopa_4_fp32_fp32_fp32(long num_repeats);
}

// get total memory in GB
std::pair<size_t, size_t> get_avail_memory() {
#if defined(__APPLE__)
    int read_kernel_info[2] = {CTL_HW, HW_MEMSIZE};
    size_t len = sizeof(size_t);
    size_t memory_B = 0;
    if (sysctl(read_kernel_info, 2, &memory_B, &len, nullptr, 0) == -1) {
        perror("sysctl");
        return std::make_pair(0, 0);
    }
    vm_size_t page_size;
    mach_port_t mach_port = mach_host_self();
    mach_msg_type_number_t count = HOST_VM_INFO_COUNT;
    vm_statistics_data_t vm_stat;
    if (host_page_size(mach_port, &page_size) != KERN_SUCCESS) {
        perror("host_page_size");
        return std::make_pair(0, 0);
    }

    if (host_statistics(mach_port, HOST_VM_INFO, (host_info_t)&vm_stat, &count) != KERN_SUCCESS) {
        perror("host_statistics");
        return std::make_pair(0, 0);
    }

    uint64_t free_memory = vm_stat.free_count * page_size;
    uint64_t active_memory = vm_stat.active_count * page_size;
    uint64_t inactive_memory = vm_stat.inactive_count * page_size;
    uint64_t wired_memory = vm_stat.wire_count * page_size;

    uint64_t available_memory = free_memory + inactive_memory;

    return std::make_pair(available_memory / 1024 / 1024 / 1024,
                          memory_B / 1024 / 1024 / 1024); // GB
#elif defined(__linux__)
    struct sysinfo info;
    if (sysinfo(&info) == -1) {
        perror("sysinfo");
        return std::make_pair(0, 0);
    }
    // info.mem_unit 是实际字节倍率（通常为 1），total/freeram 均须乘以它
    uint64_t total_B = (uint64_t)info.totalram * info.mem_unit;
    // freeram + bufferram + cached（cached 在 sysinfo 中无直接字段，
    // 使用 freeram + bufferram 作为保守的可用内存估算，
    // 与 /proc/meminfo 的 MemAvailable 相比偏小但无需解析文件）
    uint64_t avail_B = (uint64_t)(info.freeram + info.bufferram) * info.mem_unit;

    return std::make_pair(avail_B / 1024 / 1024 / 1024,
                          total_B / 1024 / 1024 / 1024); // GB
#else
    return std::make_pair(0, 0);
#endif
}

// get the list of CPU cores the current process is allowed to run on
std::vector<int> get_allowed_cpus() {
#if defined(__linux__)
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    if (sched_getaffinity(0, sizeof(cpuset), &cpuset) == -1) {
        perror("sched_getaffinity");
        return {};
    }
    std::vector<int> cpus;
    for (int i = 0; i < CPU_SETSIZE; ++i) {
        if (CPU_ISSET(i, &cpuset))
            cpus.push_back(i);
    }
    return cpus;
#else
    return {};
#endif
}

// benchmark the compute capability of the current machine
void benchmark_compute(int num_threads, long num_repeats, int (*kernel)(long), int p_core = 0, int e_core = 0,
                       const std::vector<int> &cpus = {}) {
    // when p_core/e_core are explicitly set, their sum must equal num_threads
    if ((p_core > 0 || e_core > 0) && p_core + e_core != num_threads) {
        std::cout << "p_core + e_core != num_threads. Exp sets wrong!" << std::endl;
        return;
    }
    std::cout << "Running benchmark compute. Using " << num_threads << " threads." << std::endl;
#if defined(__APPLE__)
    // macOS has no public API to hard-bind threads to physical cores.
    // pthread_attr_set_qos_class_np is the closest available mechanism:
    // QOS_CLASS_USER_INTERACTIVE  ->  P-cores (performance)
    // QOS_CLASS_BACKGROUND        ->  E-cores (efficiency)
    // when p_core/e_core are both 0, fall back to all-P-core mode
    // macOS does not implement pthread_barrier_t (optional POSIX extension).
    // Roll our own barrier with mutex + condvar:
    //   - each arriving thread increments arrived and sleeps
    //   - the last thread broadcasts to wake everyone simultaneously
    // struct Barrier {
    //     pthread_mutex_t mutex;
    //     pthread_cond_t cond;
    //     int arrived;
    //     int total;
    //     void wait() {
    //         pthread_mutex_lock(&mutex);
    //         if (++arrived == total)
    //             pthread_cond_broadcast(&cond);
    //         else
    //             while (arrived < total)
    //                 pthread_cond_wait(&cond, &mutex);
    //         pthread_mutex_unlock(&mutex);
    //     }
    // };
    // Barrier barrier = {PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER, 0, num_threads + 1};

    // struct ThreadArg {
    //     int (*kernel)(long);
    //     long num_repeats;
    //     Barrier *barrier;
    // };
    // ThreadArg arg = {kernel, num_repeats, &barrier};

    // auto thread_func = [](void *p) -> void * {
    //     ThreadArg *a = static_cast<ThreadArg *>(p);
    //     a->barrier->wait();

    //     // Request non-preemptible execution via Mach time-constraint policy.
    //     // This is the closest user-space can get to "no time-slice interruption"
    //     // on macOS; kernel interrupts (IRQ/timer) are still unavoidable.
    //     mach_timebase_info_data_t tb;
    //     mach_timebase_info(&tb);
    //     // Express a generous 60-second budget in Mach absolute time units.
    //     // period=0 means one-shot (non-periodic), matching a long compute burst.
    //     uint64_t budget = (uint64_t)60e9 * tb.denom / tb.numer;
    //     thread_time_constraint_policy_data_t ttcpolicy;
    //     ttcpolicy.period = 0;
    //     ttcpolicy.computation = budget;
    //     ttcpolicy.constraint = budget;
    //     ttcpolicy.preemptible = FALSE;
    //     thread_policy_set(pthread_mach_thread_np(pthread_self()), THREAD_TIME_CONSTRAINT_POLICY,
    //                       reinterpret_cast<thread_policy_t>(&ttcpolicy), THREAD_TIME_CONSTRAINT_POLICY_COUNT);

    //     a->kernel(a->num_repeats);
    //     return nullptr;
    // };

    // std::vector<pthread_t> threads(num_threads);
    // pthread_attr_t p_attr, e_attr;
    // pthread_attr_init(&p_attr);
    // pthread_attr_init(&e_attr);
    // pthread_attr_set_qos_class_np(&p_attr, QOS_CLASS_USER_INTERACTIVE, 0);
    // pthread_attr_set_qos_class_np(&e_attr, QOS_CLASS_BACKGROUND, 0);

    // for (int i = 0; i < p_core; i++)
    //     pthread_create(&threads[i], &p_attr, thread_func, &arg);
    // for (int i = p_core; i < num_threads; i++)
    //     pthread_create(&threads[i], &e_attr, thread_func, &arg);

    // pthread_attr_destroy(&p_attr);
    // pthread_attr_destroy(&e_attr);

    // // main thread reaches barrier last; all threads are released simultaneously
    // barrier.wait();
    // auto time_start = std::chrono::steady_clock::now();

    // for (int i = 0; i < num_threads; i++)
    //     pthread_join(threads[i], nullptr);
    // auto time_end = std::chrono::steady_clock::now();

    // pthread_mutex_destroy(&barrier.mutex);
    // pthread_cond_destroy(&barrier.cond);

    // double time_duration = std::chrono::duration_cast<std::chrono::duration<double>>(time_end - time_start).count();

    // Threaded + QoS path above is disabled; run equivalent total work on the main thread for timing.
    dispatch_qos_class_t qos_class = QOS_CLASS_USER_INTERACTIVE; // QOS_CLASS_BACKGROUND

    dispatch_queue_attr_t dispatch_queue_attr = dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_CONCURRENT, qos_class, 0);
    dispatch_queue_t dispatch_queue = dispatch_queue_create("bench_queue", dispatch_queue_attr);
    dispatch_group_t dispatch_group = dispatch_group_create();

    // benchmarking vars
    double l_gops = 0;
    std::chrono::steady_clock::time_point time_start;
    std::chrono::steady_clock::time_point time_end;
    double time_duration = 0;

    // run benchmark
    time_start = std::chrono::steady_clock::now();
    for (int l_td = 0; l_td < num_threads; l_td++) {
        dispatch_group_async(dispatch_group, dispatch_queue, ^{
          kernel(num_repeats);
        });
    }
    dispatch_group_wait(dispatch_group, DISPATCH_TIME_FOREVER);
    time_end = std::chrono::steady_clock::now();
    time_duration = std::chrono::duration_cast<std::chrono::duration<double>>(time_end - time_start).count();

    double gops = kernel(1);
    gops *= num_repeats * num_threads;
    gops *= 1.0E-9;
    gops /= time_duration;
    std::cout << "using time:" << time_duration << "s. GOPS: " << gops << " GOPS/GFLOPS" << std::endl;
#elif defined(__linux__)
    // resolve the CPU list: use the provided list or fall back to all allowed
    // CPUs
    const std::vector<int> &cpu_list = cpus.empty() ? get_allowed_cpus() : cpus;
    if ((int)cpu_list.size() < num_threads) {
        std::cout << "Not enough allowed CPUs (" << cpu_list.size() << ") for " << num_threads << " threads." << std::endl;
        return;
    }

    // Linux has native pthread_barrier_t (POSIX Barriers extension is
    // implemented)
    pthread_barrier_t barrier;
    pthread_barrier_init(&barrier, nullptr, num_threads + 1);

    struct ThreadArg {
        int (*kernel)(long);
        long num_repeats;
        pthread_barrier_t *barrier;
    };
    ThreadArg arg = {kernel, num_repeats, &barrier};

    auto thread_func = [](void *p) -> void * {
        ThreadArg *a = static_cast<ThreadArg *>(p);
        pthread_barrier_wait(a->barrier);
        a->kernel(a->num_repeats);
        return nullptr;
    };

    std::vector<pthread_t> threads(num_threads);
    static bool sched_fifo_warned = false;
    for (int i = 0; i < num_threads; i++) {
        pthread_attr_t attr;
        pthread_attr_init(&attr);

        // pin thread exclusively to one CPU core from the allowed list
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(cpu_list[i], &cpuset);
        pthread_attr_setaffinity_np(&attr, sizeof(cpuset), &cpuset);

        // SCHED_FIFO: non-preemptive real-time policy — thread runs until it
        // voluntarily yields or blocks; no time-slice interruption from the OS.
        // requires CAP_SYS_NICE or running as root.
        pthread_attr_setschedpolicy(&attr, SCHED_FIFO);
        struct sched_param sp;
        sp.sched_priority = sched_get_priority_max(SCHED_FIFO);
        pthread_attr_setschedparam(&attr, &sp);
        // must be EXPLICIT so the attr policy/priority override the parent's
        pthread_attr_setinheritsched(&attr, PTHREAD_EXPLICIT_SCHED);

        int ret = pthread_create(&threads[i], &attr, thread_func, &arg);
        if (ret == EPERM) {
            // no CAP_SYS_NICE: fall back to default scheduling policy
            if (!sched_fifo_warned) {
                std::cout << "[warn] SCHED_FIFO unavailable (EPERM), falling back to "
                             "SCHED_OTHER\n";
                sched_fifo_warned = true;
            }
            pthread_attr_setschedpolicy(&attr, SCHED_OTHER);
            pthread_attr_setinheritsched(&attr, PTHREAD_INHERIT_SCHED);
            ret = pthread_create(&threads[i], &attr, thread_func, &arg);
        }
        if (ret != 0) {
            std::cerr << "[error] pthread_create failed for thread " << i << ": " << strerror(ret) << "\n";
            pthread_barrier_destroy(&barrier);
            pthread_attr_destroy(&attr);
            return;
        }
        pthread_attr_destroy(&attr);
    }

    // main thread reaches barrier last; all threads are released simultaneously
    pthread_barrier_wait(&barrier);
    auto time_start = std::chrono::steady_clock::now();

    for (int i = 0; i < num_threads; i++)
        pthread_join(threads[i], nullptr);
    auto time_end = std::chrono::steady_clock::now();

    pthread_barrier_destroy(&barrier);

    double time_duration = std::chrono::duration_cast<std::chrono::duration<double>>(time_end - time_start).count();
    double gops = kernel(1);
    gops *= num_repeats * num_threads;
    gops *= 1.0E-9;
    gops /= time_duration;
    std::cout << "using time:" << time_duration << "s. GOPS: " << gops << " GOPS/GFLOPS" << std::endl;
#endif
}

int main() {
    std::pair<size_t, size_t> memory_GB = get_avail_memory();
    std::cout << "Memory: " << memory_GB.first << " / " << memory_GB.second << " GB" << std::endl;
#if defined(__linux__)
    std::vector<int> cpus = get_allowed_cpus();
    std::cout << "Allowed CPUs (" << cpus.size() << "):";
    for (int id : cpus)
        std::cout << " " << id;
    std::cout << std::endl;
#endif

    benchmark_compute(5, 1000000000, peak_sme_fmopa_4_fp32_fp32_fp32, 5, 0);
    return 0;
}