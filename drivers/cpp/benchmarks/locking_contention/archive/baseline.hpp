#include "atomic-wrapper.hpp"
#pragma once 


struct MCSLock_baseline {
  struct MCSNode {
    MCSNode *next;
    bool locked;
  };

  MCSNode *tail = nullptr;
  static thread_local MCSNode qnode;

  void lock(){
    store_release(qnode.next, nullptr);
    MCSNode *pred = exchange(tail, &qnode);
    if (pred != nullptr) {
        store_release(qnode.locked, true);
        store_release(pred->next, &qnode);
        while (load_acquire(qnode.locked))
        ;
    }
  }

  void unlock(){
    MCSNode *succ = load_acquire(qnode.next);
    if (succ == nullptr) {
        auto expected = &qnode;
        if (compare_exchange(tail, expected, nullptr)) {
        return;
        }
        while (succ == nullptr) {
        succ = load_acquire(qnode.next);
        }
    }
    store_release(succ->locked, false);
  }
};


thread_local MCSLock_baseline::MCSNode MCSLock_baseline::qnode;

