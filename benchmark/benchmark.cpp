#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <thread>
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

#define ITERATIONS (100 * 1000 * 1000)

void L1_miss_Latency();
void test_sector_fill_granularity();

extern "C" {
int peak_sme_fmopa_1_fp32_fp32_fp32(long num_repeats);
int peak_sme_fmopa_1_fp32_fp32_fp32_za0(long num_repeats);
int peak_sme_fmopa_1_fp32_fp32_fp32_za1(long num_repeats);
int peak_sme_fmopa_1_fp32_fp32_fp32_za2(long num_repeats);
int peak_sme_fmopa_1_fp32_fp32_fp32_za3(long num_repeats);
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

    /*
    struct Barrier {
        pthread_mutex_t mutex;
        pthread_cond_t cond;
        int arrived;
        int total;
        void wait() {
            pthread_mutex_lock(&mutex);
            if (++arrived == total)
                pthread_cond_broadcast(&cond);
            else
                while (arrived < total)
                    pthread_cond_wait(&cond, &mutex);
            pthread_mutex_unlock(&mutex);
        }
    };
    Barrier barrier = {PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER, 0, num_threads + 1};

    struct ThreadArg {
        int (*kernel)(long);
        long num_repeats;
        Barrier *barrier;
    };
    ThreadArg arg = {kernel, num_repeats, &barrier};

    auto thread_func = [](void *p) -> void * {
        ThreadArg *a = static_cast<ThreadArg *>(p);
        a->barrier->wait();

        // Request non-preemptible execution via Mach time-constraint policy.
        // This is the closest user-space can get to "no time-slice interruption"
        // on macOS; kernel interrupts (IRQ/timer) are still unavoidable.
        mach_timebase_info_data_t tb;
        mach_timebase_info(&tb);
        // Express a generous 60-second budget in Mach absolute time units.
        // period=0 means one-shot (non-periodic), matching a long compute burst.
        uint64_t budget = (uint64_t)60e9 * tb.denom / tb.numer;
        thread_time_constraint_policy_data_t ttcpolicy;
        ttcpolicy.period = 0;
        ttcpolicy.computation = budget;
        ttcpolicy.constraint = budget;
        ttcpolicy.preemptible = FALSE;
        thread_policy_set(pthread_mach_thread_np(pthread_self()), THREAD_TIME_CONSTRAINT_POLICY,
                          reinterpret_cast<thread_policy_t>(&ttcpolicy), THREAD_TIME_CONSTRAINT_POLICY_COUNT);

        a->kernel(a->num_repeats);
        return nullptr;
    };

    std::vector<pthread_t> threads(num_threads);
    pthread_attr_t p_attr, e_attr;
    pthread_attr_init(&p_attr);
    pthread_attr_init(&e_attr);
    pthread_attr_set_qos_class_np(&p_attr, QOS_CLASS_USER_INTERACTIVE, 0);
    pthread_attr_set_qos_class_np(&e_attr, QOS_CLASS_BACKGROUND, 0);

    for (int i = 0; i < p_core; i++)
        pthread_create(&threads[i], &p_attr, thread_func, &arg);
    for (int i = p_core; i < num_threads; i++)
        pthread_create(&threads[i], &e_attr, thread_func, &arg);

    pthread_attr_destroy(&p_attr);
    pthread_attr_destroy(&e_attr);

    // main thread reaches barrier last; all threads are released simultaneously
    barrier.wait();
    auto time_start = std::chrono::steady_clock::now();

    for (int i = 0; i < num_threads; i++)
        pthread_join(threads[i], nullptr);
    auto time_end = std::chrono::steady_clock::now();

    pthread_mutex_destroy(&barrier.mutex);
    pthread_cond_destroy(&barrier.cond);

    double time_duration = std::chrono::duration_cast<std::chrono::duration<double>>(time_end - time_start).count();
    */

    // Threaded + QoS path above is disabled; run equivalent total work on the main thread for timing.
    dispatch_qos_class_t qos_class =
        QOS_CLASS_BACKGROUND; // QOS_CLASS_USER_INTERACTIVE, QOS_CLASS_USER_INITIATED, QOS_CLASS_BACKGROUND

    dispatch_queue_attr_t dispatch_queue_attr = dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_CONCURRENT, qos_class, 0);
    dispatch_queue_t dispatch_queue = dispatch_queue_create("bench_queue", dispatch_queue_attr);
    dispatch_group_t dispatch_group = dispatch_group_create();

    // benchmarking vars
    std::chrono::steady_clock::time_point time_start;
    std::chrono::steady_clock::time_point time_end;

    // run benchmark
    time_start = std::chrono::steady_clock::now();
    for (int l_td = 0; l_td < num_threads; l_td++) {
        dispatch_group_async(dispatch_group, dispatch_queue, ^{
          kernel(num_repeats);
        });
        // dispatch_group_async(dispatch_group, dispatch_queue, ^{
        //   peak_sme_fmopa_1_fp32_fp32_fp32_za0(num_repeats);
        // });
    }
    // dispatch_group_async(dispatch_group, dispatch_queue, ^{
    //   peak_sme_fmopa_1_fp32_fp32_fp32_za1(num_repeats);
    // });
    // }
    dispatch_group_wait(dispatch_group, DISPATCH_TIME_FOREVER);
    time_end = std::chrono::steady_clock::now();
    double time_duration = std::chrono::duration_cast<std::chrono::duration<double>>(time_end - time_start).count();

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

void print_system_info(void) {
    printf("\n=== System Hardware Info ===\n");
#ifdef __APPLE__
    // helper: read an int64 sysctl, return -1 on failure
    auto rd_i64 = [](const char *name) -> int64_t {
        int64_t v = 0; // zero-init: 32-bit sysctl values only fill low bytes
        size_t len = sizeof(v);
        if (sysctlbyname(name, &v, &len, nullptr, 0) != 0)
            return -1;
        return v;
    };
    // helper: read a uint32 sysctl
    auto rd_u32 = [](const char *name) -> int64_t {
        uint32_t v = 0;
        size_t len = sizeof(v);
        if (sysctlbyname(name, &v, &len, nullptr, 0) != 0)
            return -1;
        return (int64_t)v;
    };
    // helper: read a string sysctl
    auto rd_str = [](const char *name, char *buf, size_t bufsz) -> bool {
        buf[0] = '\0';
        return sysctlbyname(name, buf, &bufsz, nullptr, 0) == 0 && buf[0] != '\0';
    };

    char brand[256] = {0};
    if (rd_str("machdep.cpu.brand_string", brand, sizeof(brand)))
        printf("  CPU:                    %s\n", brand);

    {
        int64_t p = rd_i64("hw.physicalcpu");
        int64_t l = rd_i64("hw.logicalcpu");
        if (p > 0)
            printf("  Physical CPUs:          %lld\n", p);
        if (l > 0)
            printf("  Logical CPUs:           %lld  (%s)\n", l, l == p ? "no hyperthreading" : "hyperthreading");
    }

    {
        int64_t nlevels = rd_i64("hw.nperflevels");
        if (nlevels > 0)
            printf("  Perf levels:            %lld\n", nlevels);
    }

    printf("\n");

    // iterate performance levels: level 0 = P-cluster, level 1 = E-cluster (ARM convention)
    int64_t nlevels = rd_i64("hw.nperflevels");
    if (nlevels < 1)
        nlevels = 2; // fallback

    for (int lvl = 0; lvl < (int)nlevels; ++lvl) {
        char key[128];
        char lvl_name[64] = {0};

        snprintf(key, sizeof(key), "hw.perflevel%d.name", lvl);
        rd_str(key, lvl_name, sizeof(lvl_name));
        if (lvl_name[0] == '\0')
            snprintf(lvl_name, sizeof(lvl_name), "Level %d", lvl);

        printf("  --- %s cluster (perflevel%d) ---\n", lvl_name, lvl);

        // core counts
        {
            snprintf(key, sizeof(key), "hw.perflevel%d.physicalcpu", lvl);
            int64_t v = rd_u32(key);
            if (v > 0)
                printf("    %-28s %lld\n", "physicalcpu:", v);
        }
        {
            snprintf(key, sizeof(key), "hw.perflevel%d.logicalcpu", lvl);
            int64_t v = rd_u32(key);
            if (v > 0)
                printf("    %-28s %lld\n", "logicalcpu:", v);
        }

        // L1 caches (bytes → KB)
        {
            snprintf(key, sizeof(key), "hw.perflevel%d.l1icachesize", lvl);
            int64_t v = rd_i64(key);
            if (v > 0)
                printf("    %-28s %lld KB\n", "l1icachesize:", v / 1024);
        }
        {
            snprintf(key, sizeof(key), "hw.perflevel%d.l1dcachesize", lvl);
            int64_t v = rd_i64(key);
            if (v > 0)
                printf("    %-28s %lld KB  (per core)\n", "l1dcachesize:", v / 1024);
        }

        // L2 (bytes → MB), shared across the cluster
        {
            snprintf(key, sizeof(key), "hw.perflevel%d.l2cachesize", lvl);
            int64_t v = rd_i64(key);
            if (v > 0)
                printf("    %-28s %lld MB  (shared by cluster)\n", "l2cachesize:", v / (1024 * 1024));
        }

        // CPU frequency (cpufrequency may be absent on Apple Silicon; try anyway)
        {
            snprintf(key, sizeof(key), "hw.perflevel%d.cpufrequency", lvl);
            int64_t v = rd_i64(key);
            if (v > 0)
                printf("    %-28s %.2f GHz\n", "cpufrequency:", v / 1e9);
        }

        printf("\n");
    }

    // global values (not per-level)
    printf("  --- Global ---\n");
    {
        int64_t v = rd_i64("hw.cachelinesize");
        if (v > 0)
            printf("    %-28s %lld B  (hw.cachelinesize)\n", "cache line size:", v);
    }
    {
        int64_t v = rd_i64("hw.memsize");
        if (v > 0)
            printf("    %-28s %lld GB\n", "total memory:", v / (1024LL * 1024 * 1024));
    }
    {
        int64_t v = rd_i64("hw.pagesize");
        if (v > 0)
            printf("    %-28s %lld KB\n", "page size:", v / 1024);
    }
    {
        // TLB sizes are not exposed by sysctl on Apple Silicon; skip
        // bus frequency (may be absent on M-series)
        int64_t v = rd_i64("hw.busfrequency");
        if (v > 0)
            printf("    %-28s %.0f MHz\n", "bus frequency:", v / 1e6);
    }

#elif defined(__linux__)
    auto rd_sc = [](int name) -> long { return sysconf(name); };
    printf("  Cache line size:        %ld B\n", rd_sc(_SC_LEVEL1_DCACHE_LINESIZE));
    printf("  L1D cache (per core):   %ld KB\n", rd_sc(_SC_LEVEL1_DCACHE_SIZE) / 1024);
    printf("  L2 cache:               %ld KB\n", rd_sc(_SC_LEVEL2_CACHE_SIZE) / 1024);
#else
    printf("  (system query not available)\n");
#endif
}

/*
 * Method 1: False Sharing
 *
 * Two threads write to addresses separated by `stride` bytes.
 * If both addresses fall in the same cache line, the line bounces
 * between cores (false sharing) and performance tanks.
 * Once stride >= cache_line_size, each thread owns its own line
 * and contention disappears — a dramatic speedup.
 *
 * On Apple Silicon (heterogeneous P+E core topology), we run three
 * scenarios to isolate which core pairing is responsible for the
 * observed boundary:
 *   P+P  QOS_CLASS_USER_INTERACTIVE × 2  → same P-core cluster
 *   E+E  QOS_CLASS_BACKGROUND       × 2  → same E-core cluster
 *   P+E  one of each                     → cross-cluster coherency
 *
 * A spin barrier ensures both threads are actually running
 * concurrently before they start incrementing.
 */

#ifdef __APPLE__
struct FalseSharingArg {
    volatile int *ptr;
    std::atomic<int> *ready;
};

static void *false_sharing_worker(void *p) {
    auto *arg = static_cast<FalseSharingArg *>(p);
    // spin until both threads have been scheduled
    arg->ready->fetch_add(1, std::memory_order_acq_rel);
    while (arg->ready->load(std::memory_order_acquire) < 2)
        ;
    for (int i = 0; i < ITERATIONS; ++i)
        ++(*(arg->ptr));
    return nullptr;
}

static void run_false_sharing_qos(const char *label, qos_class_t qos1, qos_class_t qos2) {
    char *buffer = nullptr;
    if (posix_memalign(reinterpret_cast<void **>(&buffer), 4096, 4096) != 0) {
        fprintf(stderr, "posix_memalign failed\n");
        return;
    }
    memset(buffer, 0, 4096);

    printf("\n  [%s]\n", label);
    printf("  %10s  %10s  %8s  %s\n", "Stride(B)", "Time(ms)", "Ratio", "Note");
    printf("  ----------  ----------  --------  ----\n");

    double prev_ms = 0;

    for (int stride = 4; stride <= 512; stride *= 2) {
        memset(buffer, 0, 4096);

        volatile int *a = reinterpret_cast<volatile int *>(buffer);
        volatile int *b = reinterpret_cast<volatile int *>(buffer + stride);
        std::atomic<int> ready{0};

        FalseSharingArg arg1 = {a, &ready};
        FalseSharingArg arg2 = {b, &ready};

        pthread_attr_t attr1, attr2;
        pthread_attr_init(&attr1);
        pthread_attr_init(&attr2);
        pthread_attr_set_qos_class_np(&attr1, qos1, 0);
        pthread_attr_set_qos_class_np(&attr2, qos2, 0);

        pthread_t t1, t2;
        auto t0 = std::chrono::steady_clock::now();
        pthread_create(&t1, &attr1, false_sharing_worker, &arg1);
        pthread_create(&t2, &attr2, false_sharing_worker, &arg2);
        pthread_join(t1, nullptr);
        pthread_join(t2, nullptr);
        auto t_end = std::chrono::steady_clock::now();

        pthread_attr_destroy(&attr1);
        pthread_attr_destroy(&attr2);

        double ms = std::chrono::duration<double, std::milli>(t_end - t0).count();
        double ratio = (prev_ms > 0) ? ms / prev_ms : 0.0;
        const char *note = "";
        if (prev_ms > 0 && ratio < 0.5)
            note = "<-- SPEEDUP (crossed cache line boundary)";

        printf("  %10d  %10.1f  %8.2f  %s\n", stride, ms, ratio, note);
        prev_ms = ms;
    }

    free(buffer);
}
#endif // __APPLE__

void test_false_sharing(void) {
    printf("\n=== False Sharing (2 threads) ===\n");

#ifdef __APPLE__
    // P+P: both threads hint at P-cores (same performance cluster)
    // → cache line fill size is 128 B on M-series; expect speedup at 128 B
    run_false_sharing_qos("P+P  (both QOS_CLASS_USER_INTERACTIVE)", QOS_CLASS_USER_INTERACTIVE, QOS_CLASS_USER_INTERACTIVE);

    // E+E: both threads hint at E-cores
    // → E-core L1 cache line may differ; observe where speedup appears
    run_false_sharing_qos("E+E  (both QOS_CLASS_BACKGROUND)", QOS_CLASS_BACKGROUND, QOS_CLASS_BACKGROUND);

    // P+E: threads on different cluster types
    // → cross-cluster coherency granularity may be smaller (64 B?)
    run_false_sharing_qos("P+E  (USER_INTERACTIVE + BACKGROUND, cross-cluster)", QOS_CLASS_USER_INTERACTIVE,
                          QOS_CLASS_BACKGROUND);

#else
    char *buffer = nullptr;
    if (posix_memalign(reinterpret_cast<void **>(&buffer), 4096, 4096) != 0) {
        fprintf(stderr, "posix_memalign failed\n");
        return;
    }
    memset(buffer, 0, 4096);

    printf("\n");
    printf("  %10s  %10s  %8s  %s\n", "Stride(B)", "Time(ms)", "Ratio", "Note");
    printf("  ----------  ----------  --------  ----\n");

    double prev_ms = 0;

    for (int stride = 4; stride <= 512; stride *= 2) {
        memset(buffer, 0, 4096);

        volatile int *a = reinterpret_cast<volatile int *>(buffer);
        volatile int *b = reinterpret_cast<volatile int *>(buffer + stride);

        auto t0 = std::chrono::steady_clock::now();
        std::thread t1([&]() {
            for (int i = 0; i < ITERATIONS; ++i)
                ++(*a);
        });
        std::thread t2([&]() {
            for (int i = 0; i < ITERATIONS; ++i)
                ++(*b);
        });
        t1.join();
        t2.join();
        auto t1_end = std::chrono::steady_clock::now();

        double ms = std::chrono::duration<double, std::milli>(t1_end - t0).count();
        double ratio = (prev_ms > 0) ? ms / prev_ms : 0.0;
        const char *note = "";
        if (prev_ms > 0 && ratio < 0.5)
            note = "<-- SPEEDUP (crossed cache line boundary)";

        printf("  %10d  %10.1f  %8.2f  %s\n", stride, ms, ratio, note);
        prev_ms = ms;
    }

    free(buffer);
#endif
}

// ============================================================
//  Write-Allocate Policy Probe
// ============================================================
//
// Two complementary methods to detect the D-cache write-allocate policy:
//
// Method A (DRAM bandwidth ratio)
//   Buffer >> all caches (DRAM-bound).  Compare streaming write BW to read BW.
//   Write-allocate     : write miss → fetch line first → 2× DRAM traffic
//                        ⟹  write BW ≈ read BW / 2  (ratio ≈ 0.50)
//   Non-write-allocate : write miss → write only     → 1× DRAM traffic
//                        ⟹  write BW ≈ read BW      (ratio ≈ 1.00)
//
// Method B (L1 warmth probe)
//   Buffer fits in L1.  After evicting it from L1, do a cold write-only pass,
//   then immediately measure the read time.  Compare against two baselines:
//     cold-read  : evict → read  (L2→L1 fill, slow)
//     warm-read  : read  → read  (L1 hit, fast)
//   Write-allocate     : writes fill L1 → read-after-write ≈ warm-read (L1 hit)
//   Non-write-allocate : writes bypass L1 → read-after-write ≈ cold-read (L2 hit)

// Marked noinline + opaque sink to prevent the compiler from dead-code-eliminating
// the load loops.  We want real cache traffic, not vectorised memcpy substitutions.
static volatile char g_wa_sink;

__attribute__((noinline)) static char wa_stream_read(const char *buf, size_t n, size_t stride) {
    char s = 0;
    for (size_t i = 0; i < n; i += stride)
        s ^= buf[i];
    return s;
}

__attribute__((noinline)) static void wa_stream_write(char *buf, size_t n, size_t stride) {
    for (size_t i = 0; i < n; i += stride)
        buf[i] = (char)(i >> 7);
}

void test_write_allocate() {
    // Apple Silicon cache line size (confirmed by the false-sharing test above)
    const size_t LINE = 128;

    printf("\n=== Write-Allocate Policy Probe ===\n");

    // ------------------------------------------------------------------
    // Method B: L1 warmth probe
    // ------------------------------------------------------------------
    // M4 P-core L1: 128 KB, M4 E-core L1: 64 KB.
    // Use 32 KB so the probe buffer fits safely in both core types.
    // Eviction buffer: 512 KB (8× E-core L1) evicts L1 but stays in L2.
    const size_t PROBE_SZ = 32 * 1024;
    const size_t EVICT_SZ = 512 * 1024;
    const int REPS_B = 1000;

    char *probe_buf = (char *)aligned_alloc(4096, PROBE_SZ);
    char *evict_buf = (char *)aligned_alloc(4096, EVICT_SZ);
    memset(probe_buf, 0x42, PROBE_SZ);
    memset(evict_buf, 0x55, EVICT_SZ);

    // Pre-warm evict_buf into L2 so eviction is L1-only, not DRAM-cold.
    for (int i = 0; i < 10; i++)
        g_wa_sink = wa_stream_read(evict_buf, EVICT_SZ, LINE);

    // Evict probe_buf from L1 by streaming through evict_buf.
    // evict_buf > L1, so every L1 set gets overwritten.
    auto evict_l1 = [&]() { g_wa_sink = wa_stream_read(evict_buf, EVICT_SZ, LINE); };

    // Baseline 1: cold read (L1 miss → L2 hit)
    double cold_read_us = 0;
    for (int r = 0; r < REPS_B; r++) {
        evict_l1();
        auto t0 = std::chrono::steady_clock::now();
        g_wa_sink = wa_stream_read(probe_buf, PROBE_SZ, LINE);
        auto t1 = std::chrono::steady_clock::now();
        cold_read_us += std::chrono::duration<double, std::micro>(t1 - t0).count();
    }
    cold_read_us /= REPS_B;

    // Baseline 2: warm read (L1 hit) – read twice; second read is fully in L1
    double warm_read_us = 0;
    for (int r = 0; r < REPS_B; r++) {
        g_wa_sink = wa_stream_read(probe_buf, PROBE_SZ, LINE); // fill L1
        auto t0 = std::chrono::steady_clock::now();
        g_wa_sink = wa_stream_read(probe_buf, PROBE_SZ, LINE); // measure
        auto t1 = std::chrono::steady_clock::now();
        warm_read_us += std::chrono::duration<double, std::micro>(t1 - t0).count();
    }
    warm_read_us /= REPS_B;

    // Key test: read immediately after cold writes
    double raw_us = 0;
    for (int r = 0; r < REPS_B; r++) {
        evict_l1();                                 // ensure probe_buf is cold in L1
        wa_stream_write(probe_buf, PROBE_SZ, LINE); // cold write pass
        // Drain the store buffer so subsequent loads come from cache, not forwarding.
        std::atomic_thread_fence(std::memory_order_seq_cst);
        auto t0 = std::chrono::steady_clock::now();
        g_wa_sink = wa_stream_read(probe_buf, PROBE_SZ, LINE);
        auto t1 = std::chrono::steady_clock::now();
        raw_us += std::chrono::duration<double, std::micro>(t1 - t0).count();
    }
    raw_us /= REPS_B;

    // Normalise: 0 = pure L1 hit (warm), 1 = pure L1 miss (cold L2 fill)
    double pos = (cold_read_us > warm_read_us) ? (raw_us - warm_read_us) / (cold_read_us - warm_read_us) : 0.5;

    printf("\n  [Method B: L1 warmth probe | probe=%zuKB  evict=%zuKB  reps=%d]\n", PROBE_SZ / 1024, EVICT_SZ / 1024, REPS_B);
    printf("  Warm read  (L1 hit)        : %7.3f µs\n", warm_read_us);
    printf("  Cold read  (L2 -> L1 fill) : %7.3f µs\n", cold_read_us);
    printf("  Read-after-write           : %7.3f µs\n", raw_us);
    printf("  Normalised [0=L1-hit, 1=L2-fill] : %.3f\n", pos);
    if (pos < 0.30)
        printf("  => WRITE ALLOCATE       (writes filled L1; reads are L1 hits)\n");
    else if (pos > 0.70)
        printf("  => NON-WRITE ALLOCATE   (writes bypassed L1; reads see L2 latency)\n");
    else
        printf("  => AMBIGUOUS            (possible hardware-prefetcher interference)\n");

    free(probe_buf);
    free(evict_buf);

    // ------------------------------------------------------------------
    // Method A: DRAM bandwidth ratio
    // ------------------------------------------------------------------
    // 64 MB >> M4 P-core L2 (16 MB) → DRAM-bound streaming.
    // For write-allocate,  each streaming store causes a read-for-ownership,
    // so effective DRAM traffic = 2×buf → write BW ≈ read BW / 2.
    // For non-write-allocate the line is never fetched → write BW ≈ read BW.
    const size_t DRAM_SZ = 64 * 1024 * 1024;
    const int REPS_A = 5;

    char *dram_buf = (char *)aligned_alloc(4096, DRAM_SZ);
    memset(dram_buf, 0x77, DRAM_SZ);

    // One warm-up pass to fault in pages / populate TLB
    g_wa_sink = wa_stream_read(dram_buf, DRAM_SZ, LINE);

    double read_bw_sum = 0;
    double write_bw_sum = 0;

    for (int r = 0; r < REPS_A; r++) {
        auto t0 = std::chrono::steady_clock::now();
        g_wa_sink = wa_stream_read(dram_buf, DRAM_SZ, LINE);
        auto t1 = std::chrono::steady_clock::now();
        double secs = std::chrono::duration<double>(t1 - t0).count();
        read_bw_sum += (double)DRAM_SZ / secs / (1024.0 * 1024.0 * 1024.0);
    }

    for (int r = 0; r < REPS_A; r++) {
        auto t0 = std::chrono::steady_clock::now();
        wa_stream_write(dram_buf, DRAM_SZ, LINE);
        auto t1 = std::chrono::steady_clock::now();
        double secs = std::chrono::duration<double>(t1 - t0).count();
        write_bw_sum += (double)DRAM_SZ / secs / (1024.0 * 1024.0 * 1024.0);
    }

    double avg_read_bw = read_bw_sum / REPS_A;
    double avg_write_bw = write_bw_sum / REPS_A;
    double bw_ratio = avg_write_bw / avg_read_bw;

    printf("\n  [Method A: DRAM bandwidth ratio | buf=%zuMB  reps=%d]\n", DRAM_SZ / (1024 * 1024), REPS_A);
    printf("  Read  BW          : %6.2f GB/s\n", avg_read_bw);
    printf("  Write BW          : %6.2f GB/s\n", avg_write_bw);
    printf("  Write / Read ratio: %.3f\n", bw_ratio);
    if (bw_ratio < 0.65)
        printf("  => WRITE ALLOCATE       "
               "(write BW ≈ read BW/2; each write fetches the line before writing)\n");
    else
        printf("  => NON-WRITE ALLOCATE   "
               "(write BW ≈ read BW; writes go straight to DRAM without line fetch)\n");

    free(dram_buf);
}

void benchmark_bandwidth() {}

int main() {

    std::pair<size_t, size_t> memory_GB = get_avail_memory();
    std::cout << "Memory Info: " << memory_GB.first << " / " << memory_GB.second << " GB" << std::endl;

    // CPU info
#if defined(__linux__)
    std::vector<int> cpus = get_allowed_cpus();
    std::cout << "Allowed CPUs (" << cpus.size() << "):";
    for (int id : cpus)
        std::cout << " " << id;
    std::cout << std::endl;
#endif
    // benchmark_compute(1, 1000000, peak_sme_fmopa_1_fp32_fp32_fp32, 1, 0);
    // for (int i = 1; i <= 9; i++)
    //     benchmark_compute(i, 500000000, peak_sme_fmopa_4_fp32_fp32_fp32, i, 0);

    print_system_info();
    // test_false_sharing();
    // test_write_allocate();
    L1_miss_Latency();
    test_sector_fill_granularity();
    return 0;
}