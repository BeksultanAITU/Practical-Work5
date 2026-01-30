#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <queue>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)           \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n";     \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)


// Настройки по умолчанию

static constexpr int DEFAULT_CAPACITY     = 1'000'000;  // ёмкость очереди/стека
static constexpr int BLOCK_SIZE           = 256;
static constexpr int DEFAULT_BLOCKS       = 256;
static constexpr int OPS_PER_THREAD       = 2000;       // сколько операций делает каждый поток в тесте
static constexpr int PREFILL_ITEMS        = 400'000;    // предварительно заполняем структуру, чтобы dequeue/pop не были пустыми


// GPU таймер (CUDA events)

struct GpuTimer {
    cudaEvent_t start{}, stop{};
    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
    }
    ~GpuTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    void tic() { CUDA_CHECK(cudaEventRecord(start)); }
    float toc_ms() {
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        return ms;
    }
};


struct GpuQueue {
    int* data = nullptr;
    int  capacity = 0;

    unsigned int head = 0;
    unsigned int tail = 0;

    int items = 0; // сколько элементов в очереди
    int slots = 0; // сколько свободных мест
};

__device__ __forceinline__ bool queue_enqueue(GpuQueue* q, int value) {
    // Пытаемся занять свободный слот
    int old_slots = atomicSub(&q->slots, 1);
    if (old_slots <= 0) {
        // Слотов не было => откатываем и выходим
        atomicAdd(&q->slots, 1);
        return false;
    }

    // Резервируем индекс (кольцевой буфер)
    unsigned int pos = atomicAdd(&q->tail, 1);
    int idx = (int)(pos % (unsigned int)q->capacity);

    // Записываем значение
    q->data[idx] = value;

    // Гарантируем, что запись в global memory "видна" до увеличения items
    __threadfence();

    // Сообщаем, что появился новый элемент
    atomicAdd(&q->items, 1);
    return true;
}

__device__ __forceinline__ bool queue_dequeue(GpuQueue* q, int* out) {
    // Пытаемся забрать существующий элемент
    int old_items = atomicSub(&q->items, 1);
    if (old_items <= 0) {
        // Элементов не было => откатываем и выходим
        atomicAdd(&q->items, 1);
        return false;
    }

    // Резервируем индекс для чтения
    unsigned int pos = atomicAdd(&q->head, 1);
    int idx = (int)(pos % (unsigned int)q->capacity);

    // Читаем значение
    int v = q->data[idx];

    // Освобождаем слот (теперь место снова доступно)
    atomicAdd(&q->slots, 1);

    *out = v;
    return true;
}


// Реализация MPMC стека 

struct GpuStack {
    int* data = nullptr;
    int  capacity = 0;

    int top = 0;   // следующая свободная позиция (как "size")
    int items = 0; // сколько элементов
    int slots = 0; // сколько свободных мест
};

__device__ __forceinline__ bool stack_push(GpuStack* s, int value) {
    int old_slots = atomicSub(&s->slots, 1);
    if (old_slots <= 0) {
        atomicAdd(&s->slots, 1);
        return false;
    }

    // Резервируем позицию top
    int idx = atomicAdd(&s->top, 1);
    if (idx >= s->capacity) {
        // На всякий случай, если что-то пошло не так
        atomicSub(&s->top, 1);
        atomicAdd(&s->slots, 1);
        return false;
    }

    s->data[idx] = value;
    __threadfence();
    atomicAdd(&s->items, 1);
    return true;
}

__device__ __forceinline__ bool stack_pop(GpuStack* s, int* out) {
    int old_items = atomicSub(&s->items, 1);
    if (old_items <= 0) {
        atomicAdd(&s->items, 1);
        return false;
    }

    // Забираем последний элемент
    int idx = atomicSub(&s->top, 1) - 1;
    if (idx < 0) {
        // Защита от некорректного состояния
        atomicAdd(&s->top, 1);
        atomicAdd(&s->items, 1);
        return false;
    }

    int v = s->data[idx];
    atomicAdd(&s->slots, 1);

    *out = v;
    return true;
}


// Инициализация структуры на GPU (обнуляем индексы и счётчики)

__global__ void init_queue_kernel(GpuQueue* q, int* buffer, int capacity) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        q->data = buffer;
        q->capacity = capacity;
        q->head = 0;
        q->tail = 0;
        q->items = 0;
        q->slots = capacity;
    }
}

__global__ void init_stack_kernel(GpuStack* s, int* buffer, int capacity) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        s->data = buffer;
        s->capacity = capacity;
        s->top = 0;
        s->items = 0;
        s->slots = capacity;
    }
}


// Prefill: предварительно кладём элементы, чтобы consumer-ы могли работать

__global__ void prefill_queue(GpuQueue* q, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // Каждый поток добавляет несколько элементов по stride
    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        // Если вдруг очередь заполнится — просто перестанем добавлять
        queue_enqueue(q, i);
    }
}

__global__ void prefill_stack(GpuStack* s, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        stack_push(s, i);
    }
}


// TASK 2: Ядра, где enqueue/dequeue (или push/pop) работают параллельно

struct GpuCounters {
    unsigned long long enq_ok = 0;
    unsigned long long deq_ok = 0;
    unsigned long long enq_fail = 0;
    unsigned long long deq_fail = 0;
};

__global__ void queue_mpmc_test(GpuQueue* q, int ops_per_thread, GpuCounters* cnt) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int dummy = 0;

    // Чтобы значения отличались у потоков
    int base = tid * 1237;

    for (int it = 0; it < ops_per_thread; ++it) {
        if ((tid & 1) == 0) {
            // Producer: enqueue
            bool ok = queue_enqueue(q, base + it);
            if (ok) atomicAdd(&cnt->enq_ok, 1ULL);
            else    atomicAdd(&cnt->enq_fail, 1ULL);
        } else {
            // Consumer: dequeue
            bool ok = queue_dequeue(q, &dummy);
            if (ok) atomicAdd(&cnt->deq_ok, 1ULL);
            else    atomicAdd(&cnt->deq_fail, 1ULL);
        }
    }
}

__global__ void stack_mpmc_test(GpuStack* s, int ops_per_thread, GpuCounters* cnt) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int dummy = 0;

    int base = tid * 1237;

    for (int it = 0; it < ops_per_thread; ++it) {
        if ((tid & 1) == 0) {
            bool ok = stack_push(s, base + it);
            if (ok) atomicAdd(&cnt->enq_ok, 1ULL);
            else    atomicAdd(&cnt->enq_fail, 1ULL);
        } else {
            bool ok = stack_pop(s, &dummy);
            if (ok) atomicAdd(&cnt->deq_ok, 1ULL);
            else    atomicAdd(&cnt->deq_fail, 1ULL);
        }
    }
}


// CPU (последовательные) версии для сравнения (доп. задание)

struct CpuCounters {
    unsigned long long push_ok = 0;
    unsigned long long pop_ok  = 0;
    unsigned long long push_fail = 0;
    unsigned long long pop_fail  = 0;
};

static double cpu_queue_test(int capacity, int ops, CpuCounters& c) {
    std::queue<int> q;
    auto t0 = std::chrono::high_resolution_clock::now();

    int pre = std::min(PREFILL_ITEMS, capacity);
    for (int i = 0; i < pre; ++i) q.push(i);

    for (int i = 0; i < ops; ++i) {
        if ((i & 1) == 0) {
            // enqueue
            if ((int)q.size() < capacity) { q.push(i); c.push_ok++; }
            else { c.push_fail++; }
        } else {
            // dequeue
            if (!q.empty()) { q.pop(); c.pop_ok++; }
            else { c.pop_fail++; }
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms = t1 - t0;
    return ms.count();
}

static double cpu_stack_test(int capacity, int ops, CpuCounters& c) {
    std::vector<int> s;
    s.reserve(capacity);

    auto t0 = std::chrono::high_resolution_clock::now();

    int pre = std::min(PREFILL_ITEMS, capacity);
    for (int i = 0; i < pre; ++i) s.push_back(i);

    for (int i = 0; i < ops; ++i) {
        if ((i & 1) == 0) {
            // push
            if ((int)s.size() < capacity) { s.push_back(i); c.push_ok++; }
            else { c.push_fail++; }
        } else {
            // pop
            if (!s.empty()) { s.pop_back(); c.pop_ok++; }
            else { c.pop_fail++; }
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms = t1 - t0;
    return ms.count();
}


// Печать счётчиков 

static void print_gpu_counters(const std::string& title, const GpuCounters& c) {
    std::cout << title << "\n";
    std::cout << "Enqueue/Push  OK   : " << c.enq_ok << "\n";
    std::cout << "Enqueue/Push  FAIL : " << c.enq_fail << "\n";
    std::cout << "Dequeue/Pop   OK   : " << c.deq_ok << "\n";
    std::cout << "Dequeue/Pop   FAIL : " << c.deq_fail << "\n";
}


int main() {
    std::cout << "Practical_Work: Queue vs Stack (CUDA)\n";

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "Device: " << prop.name << "\n";
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << "\n\n";

    const int capacity = DEFAULT_CAPACITY;
    const int blocks   = DEFAULT_BLOCKS;
    const int threads  = BLOCK_SIZE;

    // Общее число потоков и операций (для подсчёта throughput)
    const long long total_threads = 1LL * blocks * threads;
    const long long total_ops = total_threads * OPS_PER_THREAD;

    
    // TASK 1: Инициализация очереди и стека (ёмкость задана capacity)
    
    std::cout << "TASK 1\n";
    std::cout << "Initializing queue and stack with capacity = " << capacity << "\n\n";

    // Выделяем память под буферы данных (global memory)
    int* d_queue_buf = nullptr;
    int* d_stack_buf = nullptr;
    CUDA_CHECK(cudaMalloc(&d_queue_buf, capacity * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_stack_buf, capacity * sizeof(int)));

    // Выделяем память под "объекты" структур
    GpuQueue* d_queue = nullptr;
    GpuStack* d_stack = nullptr;
    CUDA_CHECK(cudaMalloc(&d_queue, sizeof(GpuQueue)));
    CUDA_CHECK(cudaMalloc(&d_stack, sizeof(GpuStack)));

    init_queue_kernel<<<1, 1>>>(d_queue, d_queue_buf, capacity);
    init_stack_kernel<<<1, 1>>>(d_stack, d_stack_buf, capacity);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Prefill — чтобы dequeue/pop могли сразу работать (иначе будет много fail по пустоте)
    {
        int pre = std::min(PREFILL_ITEMS, capacity);
        std::cout << "Prefill items: " << pre << "\n\n";
        prefill_queue<<<blocks, threads>>>(d_queue, pre);
        prefill_stack<<<blocks, threads>>>(d_stack, pre);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    
    // TASK 2: Параллельные enqueue + dequeue (MPMC) в одном ядре
    
    std::cout << "TASK 2\n";
    std::cout << "Running parallel enqueue/dequeue kernel (MPMC)\n";
    std::cout << "Blocks = " << blocks << ", Threads = " << threads
              << ", Ops per thread = " << OPS_PER_THREAD << "\n\n";

    // Счётчики для очереди
    GpuCounters* d_cnt_q = nullptr;
    CUDA_CHECK(cudaMalloc(&d_cnt_q, sizeof(GpuCounters)));
    CUDA_CHECK(cudaMemset(d_cnt_q, 0, sizeof(GpuCounters)));

    // Счётчики для стека
    GpuCounters* d_cnt_s = nullptr;
    CUDA_CHECK(cudaMalloc(&d_cnt_s, sizeof(GpuCounters)));
    CUDA_CHECK(cudaMemset(d_cnt_s, 0, sizeof(GpuCounters)));

    float t_queue_ms = 0.0f;
    float t_stack_ms = 0.0f;

    // --- Тест очереди ---
    {
        GpuTimer t;
        t.tic();
        queue_mpmc_test<<<blocks, threads>>>(d_queue, OPS_PER_THREAD, d_cnt_q);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        t_queue_ms = t.toc_ms();
    }

    // --- Тест стека ---
    {
        GpuTimer t;
        t.tic();
        stack_mpmc_test<<<blocks, threads>>>(d_stack, OPS_PER_THREAD, d_cnt_s);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        t_stack_ms = t.toc_ms();
    }

    // Копируем счётчики на CPU для печати
    GpuCounters hq{}, hs{};
    CUDA_CHECK(cudaMemcpy(&hq, d_cnt_q, sizeof(GpuCounters), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&hs, d_cnt_s, sizeof(GpuCounters), cudaMemcpyDeviceToHost));

    
    // TASK 3: Сравнение производительности Queue vs Stack
    
    std::cout << "TASK 3\n";
    std::cout << "Comparing performance: Queue vs Stack\n\n";

    print_gpu_counters("GPU Queue counters:", hq);
    std::cout << "GPU Queue time: " << std::fixed << std::setprecision(3) << t_queue_ms << " ms\n";
    double q_throughput = (double)total_ops / (t_queue_ms / 1000.0) / 1e6;
    std::cout << "GPU Queue throughput: " << std::fixed << std::setprecision(2)
              << q_throughput << " Mops/s\n\n";

    print_gpu_counters("GPU Stack counters:", hs);
    std::cout << "GPU Stack time: " << std::fixed << std::setprecision(3) << t_stack_ms << " ms\n";
    double s_throughput = (double)total_ops / (t_stack_ms / 1000.0) / 1e6;
    std::cout << "GPU Stack throughput: " << std::fixed << std::setprecision(2)
              << s_throughput << " Mops/s\n\n";

    // Дополнительное сравнение с последовательными CPU версиями
    {
        std::cout << "CPU baseline (sequential)\n";
        std::cout << "Total ops (CPU) = " << (int)(total_ops / 1024) * 1024 << " (approx)\n";

        int cpu_ops = (int)std::min<long long>(total_ops, 20'000'000LL); // ограничим, чтобы CPU тест не был слишком долгим

        CpuCounters cq{}, cs{};
        double cpu_q_ms = cpu_queue_test(capacity, cpu_ops, cq);
        double cpu_s_ms = cpu_stack_test(capacity, cpu_ops, cs);

        std::cout << "CPU Queue time: " << std::fixed << std::setprecision(3) << cpu_q_ms << " ms\n";
        std::cout << "CPU Stack time: " << std::fixed << std::setprecision(3) << cpu_s_ms << " ms\n\n";
    }

    // Освобождаем память
    CUDA_CHECK(cudaFree(d_cnt_q));
    CUDA_CHECK(cudaFree(d_cnt_s));
    CUDA_CHECK(cudaFree(d_queue));
    CUDA_CHECK(cudaFree(d_stack));
    CUDA_CHECK(cudaFree(d_queue_buf));
    CUDA_CHECK(cudaFree(d_stack_buf));

    return 0;
}
