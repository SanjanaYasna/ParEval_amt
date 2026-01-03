//doesn't get used in hpx.cc, since the tests are just implementation wise, but this is something that could work 
//based off of: https://mfukar.github.io/2017/09/26/mcs.html 
//NOTE: NOT SUITABLE FOR HPX THREADS, AS HPX RELIES ON USER-LEVEL THREADS (thread can start executing on a different worker if next thread is suspended)
// ...

struct HPX_MCS_Lock
{
    struct qnode
    {
        std::atomic<qnode*> next{nullptr};
        hpx::lcos::local::binary_semaphore gate{0};  // suspends waiter
    };

    std::atomic<qnode*> tail{nullptr};

    static qnode& get_node()
    {
        auto raw = hpx::this_thread::get_thread_data();
        auto* node = reinterpret_cast<qnode*>(raw);
        if (!node)
        {
            node = new qnode;
            hpx::this_thread::set_thread_data(
                reinterpret_cast<std::uintptr_t>(node));
            hpx::this_thread::run_on_exit([node]() { delete node; });
        }
        return *node;
    }

    void lock()
    {
        qnode& my = get_node();
        my.next.store(nullptr, std::memory_order_relaxed);

        qnode* pred = tail.exchange(&my, std::memory_order_acq_rel);
        if (pred != nullptr)
        {
            pred->next.store(&my, std::memory_order_release);

            // Suspend instead of spinning – the semaphore will be released by unlock().
            my.gate.acquire();
        }
    }

    void unlock()
    {
        qnode& my = get_node();

        qnode* succ = my.next.load(std::memory_order_acquire);
        if (succ == nullptr)
        {
            qnode* expected = &my;
            if (tail.compare_exchange_strong(expected, nullptr,
                                             std::memory_order_release,
                                             std::memory_order_relaxed))
            {
                my.next.store(nullptr, std::memory_order_relaxed);
                return;
            }

            do
            {
                succ = my.next.load(std::memory_order_acquire);
                if (succ == nullptr)
                    hpx::this_thread::yield();
            } while (succ == nullptr);
        }

        my.next.store(nullptr, std::memory_order_relaxed);
        succ->gate.release();        // wake the successor instead of writing to a spin flag
    }
};


// struct HPX_MCS_Lock
// {
//     struct hpx_qnode
//     {
//         std::atomic<hpx_qnode*> next{nullptr};  // make next atomic
//         std::atomic<bool>       locked{false};  // locked flag should also be atomic
//     };

//     std::atomic<hpx_qnode*> tail{nullptr};      // tail points to hpx_qnode
//     static thread_local hpx_qnode node;

//     void lock()
//     {
//         //geet predecessor as end of queue, and exchange with tail
//         node.next.store(nullptr, std::memory_order_relaxed);

//         hpx_qnode* pred = tail.exchange(&node, std::memory_order_acq_rel);
//         //if pred, that means we can't aquire lock right away and need to be added 
//         if (pred != nullptr)
//         {
//             //mark node as locked, and move to tail repeatedly 
//             node.locked.store(true, std::memory_order_relaxed);
//             pred->next.store(&node, std::memory_order_release);
//             //spin on node
//             while (node.locked.load(std::memory_order_acquire));
//                 hpx::this_thread::yield();
//         }
//     }

//     void unlock()
//     {   //load_aquire
//         hpx_qnode* succ = node.next.load(std::memory_order_acquire);
//         //no waiting, so node.next is th successor
//         if (succ == nullptr)
//         {
//             hpx_qnode* expected = &node;
//             //if no other waiting, then set to nullptr 
//             if (tail.compare_exchange_strong(expected, nullptr,
//                                              std::memory_order_release,
//                                              std::memory_order_relaxed))
//                 return;

//             do
//             {   //else move to next  ie load_acquire(node.next);
//                 succ = node.next.load(std::memory_order_acquire);
//                 if (succ == nullptr)
//                     hpx::this_thread::yield();
//             } while (succ == nullptr);
//         }

//         succ->locked.store(false, std::memory_order_release);
//         node.next.store(nullptr, std::memory_order_relaxed);
//     }
// };
// thread_local HPX_MCS_Lock::hpx_qnode HPX_MCS_Lock::node{};