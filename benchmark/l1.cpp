#include <numeric>

/*
 * Measure L2 → L1D fill latency on a P-core using random pointer chasing.
 *
 * Why pointer chasing?
 *   - Each load address comes from the *result* of the previous load (data dependency).
 *     This prevents out-of-order execution from overlapping misses; loads are fully
 *     serialized, so elapsed_time / steps = true single-miss latency.
 *   - The random permutation defeats the hardware stride/stream prefetcher.
 *
 * Why NOT the original stride-32 plan?
 *   - A fixed 128-byte stride is immediately recognized by the stride prefetcher,
 *     which pulls data into L1 before it is needed → you measure L1 hit latency.
 *   - Independent stride loads have no data dependency → OOO overlaps dozens of
 *     misses simultaneously → you measure bandwidth, not latency.
 *
 * Measurement phases:
 *   1. Build:   encode random pointer chain in the 8 MB working set.
 *   2. Warmup:  sequential pass over the whole buffer → populates L2.
 *   3. Evict:   access a separate 512 KB buffer (4× L1D) → flushes L1D.
 *   4. Chase:   walk the pointer chain for STEPS steps and time it.
 *               Working set (8 MB) > L1D (128 KB) → almost every step is an L1 miss.
 *               Working set (8 MB) < P-cluster L2 (16 MB) → misses are served by L2.
 */

#ifdef __APPLE__
struct ChaseArg {
    double ns_per_step;
    volatile uint64_t sink;
};

// Run a pointer chase on a working set of `wset_bytes` and return ns per step.
// Precondition: called from a thread already pinned to the desired QoS class.
static double run_chase(size_t wset_bytes, volatile uint64_t &sink_out) {
    constexpr size_t LINE_BYTES = 128;
    const size_t n_lines = wset_bytes / LINE_BYTES;
    // Evict buffer: 4× P-core L1D (512 KB) guarantees full L1D flush regardless of core type.
    constexpr size_t EVICT_BYTES = 512 * 1024;
    const size_t steps = n_lines * 10000;

    char *buf = nullptr;
    char *evict = nullptr;
    if (posix_memalign(reinterpret_cast<void **>(&buf), 4096, wset_bytes) != 0 ||
        posix_memalign(reinterpret_cast<void **>(&evict), LINE_BYTES, EVICT_BYTES) != 0) {
        free(buf);
        return -1.0;
    }
    memset(buf, 0, wset_bytes);
    memset(evict, 1, EVICT_BYTES);

    // Phase 1: build random pointer chain (Fisher-Yates + xorshift64)
    std::vector<uint32_t> perm(n_lines);
    std::iota(perm.begin(), perm.end(), 0u);
    uint64_t rng = 0x9e3779b97f4a7c15ULL;
    for (size_t i = n_lines - 1; i > 0; --i) {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        std::swap(perm[i], perm[static_cast<uint32_t>(rng % (i + 1))]);
    }
    for (size_t i = 0; i < n_lines; ++i) {
        auto *cur = reinterpret_cast<uint64_t *>(buf + (size_t)perm[i] * LINE_BYTES);
        auto *next = reinterpret_cast<uint64_t *>(buf + (size_t)perm[(i + 1) % n_lines] * LINE_BYTES);
        *cur = reinterpret_cast<uint64_t>(next);
    }

    // Phase 2: warmup — sequential pass to populate the cache level under test
    volatile uint64_t sink = 0;
    for (size_t i = 0; i < n_lines; ++i)
        sink ^= *reinterpret_cast<uint64_t *>(buf + i * LINE_BYTES);

    // Phase 3: evict L1D — touch a separate buffer larger than any L1D
    for (size_t i = 0; i < EVICT_BYTES; i += LINE_BYTES)
        sink ^= *reinterpret_cast<volatile uint64_t *>(evict + i);

    // Phase 4: measure — dependent pointer chase
    volatile uint64_t *ptr = reinterpret_cast<volatile uint64_t *>(buf + (size_t)perm[0] * LINE_BYTES);
    auto t0 = std::chrono::steady_clock::now();
    for (size_t i = 0; i < steps; ++i)
        ptr = reinterpret_cast<volatile uint64_t *>(*ptr);
    auto t1 = std::chrono::steady_clock::now();

    sink_out ^= sink ^ reinterpret_cast<uint64_t>(ptr);
    free(evict);
    free(buf);
    return std::chrono::duration<double, std::nano>(t1 - t0).count() / static_cast<double>(steps);
}

/*
 * Double-hop pointer chase: sector0 → sector1 (same node) → sector0 (next node)
 *
 * Node layout (128 B per node):
 *   offset  0  (sector 0): stores address of sector 1 of the SAME node  (= node_base + 64)
 *   offset 64  (sector 1): stores address of sector 0 of the NEXT node  (random permutation)
 *
 * Chase inner loop — one "step" = one node visit = 2 sequential loads:
 *   ptr = *ptr;   // load sector 0 → data gives &sector1 of THIS node
 *   ptr = *ptr;   // load sector 1 → data gives &sector0 of NEXT node
 *
 * Why the two loads are truly sequential:
 *   The address of sector 1 comes from the DATA of sector 0, not from the known
 *   base pointer.  The CPU cannot issue the second load until the first completes.
 *
 * Fill-granularity inference:
 *   fill = 128 B → sector 0 load also fills sector 1 into L1D
 *                  → second load is an L1D hit  (~4 cyc)
 *                  → ns/step ≈ T_L2 + 4 cyc  ≈ 1.1 × T_L2
 *
 *   fill =  64 B → sector 0 load brings only 64 B (sector cache)
 *                  → sector 1 is NOT yet in L1D
 *                  → second load is another L2 miss  (~T_L2)
 *                  → ns/step ≈ T_L2 + T_L2  = 2.0 × T_L2
 *
 * Ratio  double_hop_ns / baseline_ns:
 *   ≈ 1.1  →  128 B fill  (full super-line fetched on every miss)
 *   ≈ 2.0  →   64 B fill  (sector cache, one sector per miss)
 */
static double run_double_hop_chase(size_t wset_bytes, volatile uint64_t &sink_out) {
    constexpr size_t LINE_BYTES = 128;
    constexpr size_t SECTOR_BYTES = 64;
    const size_t n_lines = wset_bytes / LINE_BYTES;
    constexpr size_t EVICT_BYTES = 512 * 1024;
    const size_t steps = n_lines * 10000; // node visits (each = 2 loads)

    char *buf = nullptr;
    char *evict = nullptr;
    if (posix_memalign(reinterpret_cast<void **>(&buf), 4096, wset_bytes) != 0 ||
        posix_memalign(reinterpret_cast<void **>(&evict), LINE_BYTES, EVICT_BYTES) != 0) {
        free(buf);
        return -1.0;
    }
    memset(buf, 0, wset_bytes);
    memset(evict, 1, EVICT_BYTES);

    // Build random permutation (different seed from run_chase to avoid correlated patterns)
    std::vector<uint32_t> perm(n_lines);
    std::iota(perm.begin(), perm.end(), 0u);
    uint64_t rng = 0xdeadbeefcafe1234ULL;
    for (size_t i = n_lines - 1; i > 0; --i) {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        std::swap(perm[i], perm[static_cast<uint32_t>(rng % (i + 1))]);
    }

    // Build double-hop chain
    for (size_t i = 0; i < n_lines; ++i) {
        char *node_k = buf + (size_t)perm[i] * LINE_BYTES;
        char *sec1_k = node_k + SECTOR_BYTES;                                   // sector 1 of same node
        char *node_m_sec0 = buf + (size_t)perm[(i + 1) % n_lines] * LINE_BYTES; // sector 0 of next node
        *reinterpret_cast<uint64_t *>(node_k) = reinterpret_cast<uint64_t>(sec1_k);
        *reinterpret_cast<uint64_t *>(sec1_k) = reinterpret_cast<uint64_t>(node_m_sec0);
    }

    // Warmup: stride 64 B to touch every sector and populate L2
    volatile uint64_t sink = 0;
    for (size_t i = 0; i < wset_bytes; i += SECTOR_BYTES)
        sink ^= *reinterpret_cast<volatile uint64_t *>(buf + i);

    // Evict L1D
    for (size_t i = 0; i < EVICT_BYTES; i += LINE_BYTES)
        sink ^= *reinterpret_cast<volatile uint64_t *>(evict + i);

    // Measure double-hop chase
    volatile uint64_t *ptr = reinterpret_cast<volatile uint64_t *>(buf + (size_t)perm[0] * LINE_BYTES);
    auto t0 = std::chrono::steady_clock::now();
    for (size_t i = 0; i < steps; ++i) {
        ptr = reinterpret_cast<volatile uint64_t *>(*ptr); // sector 0 → &sector1 same node
        ptr = reinterpret_cast<volatile uint64_t *>(*ptr); // sector 1 → &sector0 next node
    }
    auto t1 = std::chrono::steady_clock::now();

    sink_out ^= sink ^ reinterpret_cast<uint64_t>(ptr);
    free(evict);
    free(buf);
    return std::chrono::duration<double, std::nano>(t1 - t0).count() / static_cast<double>(steps);
}

struct ChaseScenario {
    const char *label;
    qos_class_t qos;
    size_t wset_bytes;
    double ns_per_step;
    volatile uint64_t sink;
};

static void *l2_latency_worker(void *p) {
    auto *s = static_cast<ChaseScenario *>(p);
    s->ns_per_step = run_chase(s->wset_bytes, s->sink);
    return nullptr;
}

struct SectorProbeArg {
    size_t wset_bytes;
    double baseline_ns; // run_chase result
    double dblhop_ns;   // run_double_hop_chase result
    volatile uint64_t sink;
};

static void *sector_probe_worker(void *p) {
    auto *a = static_cast<SectorProbeArg *>(p);
    a->baseline_ns = run_chase(a->wset_bytes, a->sink);
    a->dblhop_ns = run_double_hop_chase(a->wset_bytes, a->sink);
    return nullptr;
}
#endif // __APPLE__

void L1_miss_Latency() {
    printf("\n============= Cache Miss Latency  (random pointer chase) ===========\n");
    printf("  Each row: independent load address = value from previous load\n");
    printf("  → serial dependency → OOO cannot overlap → measures true per-miss latency\n");
    printf("  → random permutation → stride prefetcher cannot predict next address\n\n");

#ifdef __APPLE__
    // M4 P-core boost frequency ~4.4-4.5 GHz (not exposed via sysctl)
    constexpr double P_GHZ = 4.4;
    constexpr double E_GHZ = 2.9; // M4 E-core ~2.6 GHz

    //                 label                           QoS                       working set
    ChaseScenario scenarios[] = {
        {"P-core  2 MB  (< P-L2 16MB, < E-L2 4MB)", QOS_CLASS_USER_INTERACTIVE, 2ULL << 20, 0, 0},
        {"P-core  8 MB  (< P-L2 16MB, > E-L2 4MB)", QOS_CLASS_USER_INTERACTIVE, 8ULL << 20, 0, 0},
        // {"E-core  2 MB  (< E-L2 4MB)",               QOS_CLASS_BACKGROUND,       2ULL<<20, 0, 0},
        // {"E-core  8 MB  (> E-L2 4MB → SLC)",         QOS_CLASS_BACKGROUND,       8ULL<<20, 0, 0},
    };

    printf("  %-52s  %8s  %8s  %s\n", "Scenario", "ns/miss", "cyc est.", "Inferred level");
    printf("  %s\n", std::string(100, '-').c_str());

    for (auto &s : scenarios) {
        pthread_attr_t attr;
        pthread_attr_init(&attr);
        pthread_attr_set_qos_class_np(&attr, s.qos, 0);
        pthread_t tid;
        pthread_create(&tid, &attr, l2_latency_worker, &s);
        pthread_join(tid, nullptr);
        pthread_attr_destroy(&attr);

        if (s.ns_per_step < 0.0) {
            printf("  [error] alloc failed\n");
            continue;
        }

        bool is_p = (s.qos == QOS_CLASS_USER_INTERACTIVE);
        double ghz = is_p ? P_GHZ : E_GHZ;
        double cyc = s.ns_per_step * ghz;

        const char *level = cyc < 15 ? "L1D (unexpected warm hit?)" : cyc < 50 ? "L2" : cyc < 130 ? "SLC" : "DRAM";

        printf("  %-52s  %8.2f  %8.1f  %s\n", s.label, s.ns_per_step, cyc, level);
        if (s.sink == 0xdeadbeefULL)
            printf("  sink=%llu\n", (unsigned long long)s.sink);
    }

    printf("\n  Reference: L1D ~4 cyc | L2 ~40-65 cyc | SLC ~80-120 cyc | DRAM ~150+ cyc\n");
    printf("  Note: QoS class is a scheduling hint, not a hard core pin.\n");
    printf("        P-core freq ~%.1f GHz / E-core freq ~%.1f GHz are estimates.\n", P_GHZ, E_GHZ);
#else
    printf("  (macOS / Apple Silicon only)\n");
#endif
}

/*
 * Sector cache fill-granularity probe
 *
 * Runs baseline (sector-0-only) and double-hop (sector0→sector1→next) chains
 * on the same working set and compares ns/node-visit.
 *
 *   ratio = double_hop / baseline
 *   ≈ 1.0-1.2  →  fill unit = 128 B  (full super-line; sector 1 arrives free)
 *   ≈ 1.8-2.0  →  fill unit =  64 B  (sector cache; sector 1 costs a second L2 miss)
 */
void test_sector_fill_granularity() {
    printf("\n========== Sector Cache Fill-Granularity Probe ==========\n");
    printf("  Baseline:    sector-0-only chain  (1 L2 miss / node)\n");
    printf("  Double-hop:  sector0→sector1→next (2 sequential loads / node)\n");
    printf("               sector1 addr comes from sector0 DATA → true serial dependency\n\n");

#ifdef __APPLE__
    constexpr double P_GHZ = 4.4;

    // Use 2 MB: fits comfortably in P-cluster L2 (16 MB), TLB pressure low (128 pages @ 16 KB)
    constexpr size_t WSET = 2ULL * 1024 * 1024;

    SectorProbeArg arg = {WSET, 0.0, 0.0, 0};

    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_set_qos_class_np(&attr, QOS_CLASS_USER_INTERACTIVE, 0);
    pthread_t tid;
    pthread_create(&tid, &attr, sector_probe_worker, &arg);
    pthread_join(tid, nullptr);
    pthread_attr_destroy(&attr);

    if (arg.baseline_ns < 0 || arg.dblhop_ns < 0) {
        printf("  [error] memory allocation failed\n");
        return;
    }

    double ratio = arg.dblhop_ns / arg.baseline_ns;

    printf("  %-30s  %8.2f ns  %6.1f cyc\n", "Baseline (sector 0 only):", arg.baseline_ns, arg.baseline_ns * P_GHZ);
    printf("  %-30s  %8.2f ns  %6.1f cyc\n", "Double-hop (sector 0+1):", arg.dblhop_ns, arg.dblhop_ns * P_GHZ);
    printf("  %-30s  %8.2f\n\n", "Ratio double-hop / baseline:", ratio);

    if (ratio < 1.35) {
        printf("  Inference: fill unit = 128 B\n");
        printf("    Sector 1 was already in L1D after loading sector 0\n");
        printf("    → P-core fetches full 128-byte super-line on every L1D miss\n");
    } else if (ratio > 1.65) {
        printf("  Inference: fill unit = 64 B  (sector cache confirmed)\n");
        printf("    Sector 1 was NOT in L1D after loading sector 0\n");
        printf("    → P-core fetches one 64-byte sector per L1D miss\n");
    } else {
        printf("  Inference: ambiguous (ratio %.2f between 1.35 and 1.65)\n", ratio);
        printf("    Possible causes: L2 pipelining, partial prefetch, or measurement noise\n");
    }

    if (arg.sink == 0xdeadbeefULL)
        printf("  sink=%llu\n", (unsigned long long)arg.sink);
#else
    printf("  (macOS / Apple Silicon only)\n");
#endif
}
