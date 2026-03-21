#include <chrono>
#include <cstddef>
#include <iostream>
#include <utility>
#include <vector>

#if defined(MAC)
#include <dispatch/dispatch.h>
#include <mach/mach_host.h>
#include <os/proc.h>
#include <sys/sysctl.h>
#elif defined(HW)
#include <pthread.h>
#include <sched.h>
#include <sys/sysinfo.h>
#endif

extern "C" {
int peak_sme_fmopa_1_fp32_fp32_fp32(int num_repeats);
}

// get total memory in GB
std::pair<size_t, size_t> get_avail_memory() {
#if defined(MAC)
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
#elif defined(HW)
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
#if defined(HW)
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
void benchmark_compute(int num_threads, int num_repeats, int (*kernel)(int)) {
#if defined(MAC)
    dispatch_qos_class_t qos = QOS_CLASS_USER_INTERACTIVE; // the highest priority
    // set up dispatch queue
    dispatch_queue_attr_t l_attr = dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_CONCURRENT, qos, 0);
    dispatch_queue_t l_queue = dispatch_queue_create("bench_queue", l_attr);
    dispatch_group_t l_group = dispatch_group_create();

    double gops = 0;
    std::chrono::steady_clock::time_point time_start;
    std::chrono::steady_clock::time_point time_end;
    double time_duration = 0;
    time_start = std::chrono::steady_clock::now();
    for (int l_td = 0; l_td < num_threads; l_td++) {
        dispatch_group_async(l_group, l_queue, ^{
          kernel(num_repeats);
        });
    }
    dispatch_group_wait(l_group, DISPATCH_TIME_FOREVER);
    time_end = std::chrono::steady_clock::now();
    time_duration = std::chrono::duration_cast<std::chrono::duration<double>>(time_end - time_start).count();

    // determine GOPS
    gops = kernel(1);
    gops *= num_repeats * num_threads;
    gops *= 1.0E-9;
    gops /= time_duration;
    std::cout << "GOPS: " << gops << " GOPS/GFLOPS" << std::endl;
#elif defined(HW)
    struct ThreadArg { int (*kernel)(int); int num_repeats; };
    ThreadArg arg = { kernel, num_repeats };

    // non-capturing lambda can be cast to a plain C function pointer for pthread
    auto thread_func = [](void *p) -> void * {
        ThreadArg *a = static_cast<ThreadArg *>(p);
        a->kernel(a->num_repeats);
        return nullptr;
    };

    std::vector<pthread_t> threads(num_threads);
    double gops = 0;
    auto time_start = std::chrono::steady_clock::now();
    for (int l_td = 0; l_td < num_threads; l_td++)
        pthread_create(&threads[l_td], nullptr, thread_func, &arg);
    for (int l_td = 0; l_td < num_threads; l_td++)
        pthread_join(threads[l_td], nullptr);
    auto time_end = std::chrono::steady_clock::now();
    double time_duration = std::chrono::duration_cast<std::chrono::duration<double>>(time_end - time_start).count();

    gops = kernel(1);
    gops *= num_repeats * num_threads;
    gops *= 1.0E-9;
    gops /= time_duration;
    std::cout << "GOPS: " << gops << " GOPS/GFLOPS" << std::endl;
#endif
}

void run_benchmark_compute(int num_threads, int num_repeats) {
    std::cout << "Running benchmark compute with " << num_threads << " threads" << std::endl;

    benchmark_compute(num_threads, num_repeats, peak_sme_fmopa_1_fp32_fp32_fp32);
}
int main() {
    std::pair<size_t, size_t> memory_GB = get_avail_memory();
    std::cout << "Memory: " << memory_GB.first << " / " << memory_GB.second << " GB" << std::endl;
#if defined(HW)
    std::vector<int> cpus = get_allowed_cpus();
    std::cout << "Allowed CPUs (" << cpus.size() << "):";
    for (int id : cpus)
        std::cout << " " << id;
    std::cout << std::endl;
#endif
    run_benchmark_compute(1, 1000000000);
    return 0;
}