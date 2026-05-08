#![no_std]

use core::{
    cell::UnsafeCell,
    ops::{Deref, DerefMut},
    sync::atomic::{AtomicBool, Ordering},
};

/// A mutual exclusion primitive based on spinning.
///
/// This lock uses a `Test and Test-and-Set` (TTAS) strategy to minimize cache contention.
#[derive(Debug)]
pub struct SpinLock<T> {
    /// Tracks whether the lock is currently held.
    locked: AtomicBool,
    /// The actual data protected by the lock.
    data: UnsafeCell<T>,
}

impl<T> SpinLock<T> {
    /// Creates a new, unlocked `SpinLock` protecting the provided `data`.
    pub const fn new(data: T) -> Self {
        Self {
            locked: AtomicBool::new(false),
            data: UnsafeCell::new(data),
        }
    }

    /// Acquires the lock, spinning until it becomes available.
    ///
    /// This method uses the TTAS pattern:
    /// 1. It first performs a [Relaxed](Ordering::Relaxed) load to check if the lock is held. This keeps
    ///    the CPU cache line in a `Shared` state to avoid memory bus saturation.
    /// 2. Once the lock appears free, it attempts a [compare_exchange_weak](AtomicBool::compare_exchange_weak) with
    ///    [Acquire](Ordering::Acquire) ordering to officially claim ownership.
    ///
    /// Returns a [`Guard`] which automatically releases the lock when dropped.
    pub fn lock(&self) -> Guard<'_, T> {
        loop {
            // "Test" - Look at the value without trying to write to it.
            if self.locked.load(Ordering::Relaxed) {
                core::hint::spin_loop();
                continue;
            }

            // "Test-and-Set" - Try to grab the lock now that it looks free.
            // We use compare_exchange_weak because we are already in a loop.
            if self
                .locked
                .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
                .is_err()
            {
                core::hint::spin_loop();
                continue;
            }

            return Guard { lock: self };
        }
    }

    /// Acquires the lock using an optimistic strategy.
    ///
    /// This variant assumes that the lock is likely to be free. It attempts an
    /// atomic write immediately upon being called.
    ///
    /// ⚠️ **Warning**: If the lock is held, this core will perform an
    /// unnecessary atomic write attempt, which invalidates the cache lines
    /// of other CPUs before falling back to the `load` loop.
    pub fn optimistic_lock(&self) -> Guard<'_, T> {
        loop {
            // We assume the lock is free and try to grab it immediately.
            if self
                .locked
                .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
                .is_ok()
            {
                return Guard { lock: self };
            }

            // If we failed to grab it, we sit in a read-only loop until the lock appears free again.
            while self.locked.load(Ordering::Relaxed) {
                core::hint::spin_loop();
                continue;
            }
        }
    }

    /// Returns `true` if the lock is currently held by a thread.
    ///
    /// ⚠️ **Warning**: The lock state _**may change**_ immediately after the value is returned.
    pub fn is_locked(&self) -> bool {
        self.locked.load(Ordering::Relaxed)
    }
}

impl<T> From<T> for SpinLock<T> {
    /// Creates a new `SpinLock` from the given data.
    fn from(data: T) -> Self {
        Self::new(data)
    }
}

// SAFETY: We can share the lock across threads as long as the data inside it
// can be safely sent between threads.
unsafe impl<T: Send> Sync for SpinLock<T> {}

/// An RAII implementation of a `scoped lock` of a [SpinLock].
///
/// When this structure is dropped, the lock will be unlocked.
#[derive(Debug)]
pub struct Guard<'a, T> {
    lock: &'a SpinLock<T>,
}

impl<'a, T> Guard<'a, T> {
    /// Manually drops the guard, releasing the lock immediately.
    ///
    /// This is functionally equivalent to letting the guard fall out of scope.
    pub fn unlock(self) {
        // Ownership is consumed, triggering drop()
    }
}

impl<'a, T> Deref for Guard<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        // SAFETY: The existence of the Guard proves we have exclusive access.
        unsafe { &*self.lock.data.get() }
    }
}

impl<'a, T> DerefMut for Guard<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        // SAFETY: The existence of the Guard proves we have exclusive access.
        unsafe { &mut *self.lock.data.get() }
    }
}

impl<'a, T> Drop for Guard<'a, T> {
    fn drop(&mut self) {
        // Use Release ordering to ensure all modifications to the data are
        // visible to the next thread that acquires the lock.
        self.lock.locked.store(false, Ordering::Release);
    }
}

// SAFETY: A Guard can be moved to another thread if the data itself is Send.
unsafe impl<'a, T> Send for Guard<'a, T> where T: Send {}
// SAFETY: A reference to a Guard can be shared if the data itself is Sync.
unsafe impl<'a, T> Sync for Guard<'a, T> where T: Sync {}

#[cfg(test)]
mod tests {
    extern crate std;

    use crate::SpinLock;

    #[test]
    fn base_test() {
        let lock = SpinLock::new(std::vec::Vec::new());
        std::thread::scope(|s| {
            s.spawn(|| lock.lock().push(1));
            s.spawn(|| lock.lock().push(2));
            s.spawn(|| lock.lock().push(3));
        });

        let mut guard = lock.lock();
        guard.sort();

        assert_eq!(guard.as_slice(), &[1, 2, 3]);
    }

    #[test]
    fn ordering_test() {
        let lock = SpinLock::new(std::vec::Vec::new());
        std::thread::scope(|s| {
            s.spawn(|| lock.lock().push(1));
            s.spawn(|| {
                let mut guard = lock.lock();
                guard.push(2);
                guard.push(3);
            });
        });

        let guard = lock.lock();
        assert!(guard.as_slice() == &[1, 2, 3] || guard.as_slice() == &[2, 3, 1]);
    }
}
