use std::{
    cell::UnsafeCell,
    mem::MaybeUninit,
    sync::{
        Arc,
        atomic::{AtomicU32, Ordering},
    },
};

const EMPTY: u32 = 0;
const SENT: u32 = 1;
const RECEIVED: u32 = 2;
const CLOSED: u32 = 3;

/// Creates a single-use channel.
///
/// The channel has exactly one sender and one receiver.
/// [Sender] and [Receiver] are intentionally non-`Clone`.
pub fn channel<T>() -> (Sender<T>, Receiver<T>) {
    let channel = Arc::new(Slot::new());
    (Sender::new(channel.clone()), Receiver::new(channel))
}

#[derive(Debug)]
struct Slot<T> {
    // The value is synchronized by `state`:
    // - EMPTY    => uninitialized
    // - SENT     => initialized and stored in the slot
    // - RECEIVED => moved out by the receiver
    // - CLOSED   => no value is stored
    value: UnsafeCell<MaybeUninit<T>>,

    // Slot state and futex-style wait word.
    //
    // The receiver waits while this is `EMPTY`.
    // Any transition that may wake it must update this word first.
    state: AtomicU32,
}

impl<T> Slot<T> {
    const fn new() -> Self {
        Self {
            value: UnsafeCell::new(MaybeUninit::uninit()),
            state: AtomicU32::new(EMPTY),
        }
    }
}

impl<T> Drop for Slot<T> {
    fn drop(&mut self) {
        // Drop only an unread value.
        if *self.state.get_mut() == SENT {
            unsafe {
                // SAFETY: `SENT` means the value was initialized and not moved out.
                self.value.get_mut().assume_init_drop();
            }
        }
    }
}

// SAFETY:
// The slot transfers `T` between threads, so `T: Send` is required.
// Access to `value` is guarded by the `state` machine.
unsafe impl<T: Send> Send for Slot<T> {}

// SAFETY:
// Shared access is safe because `value` is touched only through the
// single-sender/single-receiver protocol.
unsafe impl<T: Send> Sync for Slot<T> {}

#[derive(Debug)]
pub struct Sender<T> {
    channel: Arc<Slot<T>>,
}

impl<T> Sender<T> {
    #[inline]
    const fn new(channel: Arc<Slot<T>>) -> Self {
        Self { channel }
    }

    /// Sends a value into the channel.
    ///
    /// ### Returns ###
    /// * `Ok(())` means the value was published.
    /// * `Err(value)` if the receiver was already dropped.
    pub fn send(self, value: T) -> Result<(), T> {
        unsafe {
            // SAFETY:
            // The receiver may read `value` only after `SENT` is published with `Release`.
            let ptr = self.channel.value.get();
            (*ptr).write(value);
        }

        match self
            .channel
            .state
            .compare_exchange(EMPTY, SENT, Ordering::Release, Ordering::Relaxed)
        {
            Ok(EMPTY) => {
                // Publish succeeded; wake a possibly waiting receiver.
                atomic_wait::wake_one(&self.channel.state);
                Ok(())
            }
            Err(CLOSED) => {
                // Receiver is gone, so take the value back.
                let value = unsafe {
                    // SAFETY:
                    // We wrote this value above, but never published `SENT`.
                    let ptr = self.channel.value.get();
                    (*ptr).assume_init_read()
                };

                Err(value)
            }
            Ok(_state) | Err(_state) => cfg_select! {
                test => {{
                    // Clean up the written value before panicking in tests.
                    let value = unsafe {
                        // SAFETY:
                        // The value was written above and was not published.
                        let ptr = self.channel.value.get();
                        (*ptr).assume_init_read()
                    };

                    drop(value);

                    unreachable!("wrong channel state: {_state}")
                }}
                _ => unsafe {
                    // SAFETY:
                    // With one non-Clone sender, one non-Clone receiver, and a
                    // private `Slot`, no other state is reachable here.
                    std::hint::unreachable_unchecked();
                }
            },
        }
    }
}

impl<T> Drop for Sender<T> {
    fn drop(&mut self) {
        // If no value was sent, close the channel and wake the receiver.
        if self
            .channel
            .state
            .compare_exchange(EMPTY, CLOSED, Ordering::Relaxed, Ordering::Relaxed)
            .is_ok()
        {
            atomic_wait::wake_one(&self.channel.state);
        }
    }
}

#[derive(Debug)]
pub struct Receiver<T> {
    channel: Arc<Slot<T>>,
}

impl<T> Receiver<T> {
    #[inline]
    const fn new(channel: Arc<Slot<T>>) -> Self {
        Self { channel }
    }

    /// Receives the value from the channel.
    ///
    /// Returns `None` if the sender was dropped without sending.
    pub fn receive(self) -> Option<T> {
        // Wait in a loop: futex-style waits may return spuriously, and wake-up
        // only means the state may have changed.
        loop {
            match self.channel.state.load(Ordering::Acquire) {
                EMPTY => atomic_wait::wait(&self.channel.state, EMPTY),
                SENT => {
                    let value = unsafe {
                        // SAFETY:
                        // `Acquire` on `SENT` synchronizes with sender's `Release`.
                        let ptr = self.channel.value.get();
                        (*ptr).assume_init_read()
                    };

                    // Prevent `Slot::drop` from dropping the value again.
                    //
                    // `Relaxed` is enough because this transition does not publish any
                    // additional memory to the sender. The sender only needs to observe
                    // whether the state changed from `EMPTY`` to `CLOSED`.
                    self.channel.state.store(RECEIVED, Ordering::Relaxed);

                    return Some(value);
                }
                CLOSED => return None,
                _state => cfg_select! {
                    test => unreachable!("wrong channel state: {_state}"),
                    _ => unsafe {
                        // SAFETY:
                        // Only EMPTY, SENT, RECEIVED, and CLOSED are valid states,
                        // and RECEIVED is unreachable before `receive` returns.
                        std::hint::unreachable_unchecked();
                    }
                },
            }
        }
    }
}

impl<T> Drop for Receiver<T> {
    fn drop(&mut self) {
        // If no value was sent, close the channel.
        // No wake-up is needed: the sender never waits.
        //
        // `Relaxed` is enough because this transition does not publish any
        // additional memory to the sender. The sender only needs to observe
        // whether the state changed from `EMPTY`` to `CLOSED`.
        let _ = self.channel.state.compare_exchange(
            EMPTY,
            CLOSED,
            Ordering::Relaxed,
            Ordering::Relaxed,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        },
        thread,
    };

    #[test]
    fn send_from_thread_receive_on_main_thread() {
        let (tx, rx) = channel();

        let sender = thread::spawn(move || {
            tx.send(String::from("hello")).unwrap();
        });
        let value = rx.receive();
        sender.join().unwrap();

        assert_eq!(value.as_deref(), Some("hello"));
    }

    #[test]
    fn receive_from_thread_send_from_main_thread() {
        let (tx, rx) = channel();

        let receiver = thread::spawn(move || rx.receive());
        tx.send(123).unwrap();

        assert_eq!(receiver.join().unwrap(), Some(123));
    }

    #[test]
    fn dropping_sender_wakes_receiver() {
        let (tx, rx) = channel::<String>();

        let receiver = thread::spawn(move || rx.receive());
        drop(tx);

        assert_eq!(receiver.join().unwrap(), None);
    }

    #[test]
    fn dropping_receiver_makes_send_fail() {
        let (tx, rx) = channel();

        let receiver = thread::spawn(move || {
            drop(rx);
        });
        receiver.join().unwrap();

        assert_eq!(tx.send(String::from("value")), Err(String::from("value")));
    }

    #[test]
    fn sent_value_is_dropped_if_receiver_never_receives() {
        #[derive(Debug)]
        struct DropCounter(Arc<AtomicUsize>);

        impl Drop for DropCounter {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::Relaxed);
            }
        }

        let drops = Arc::new(AtomicUsize::new(0));
        {
            let (tx, rx) = channel();
            tx.send(DropCounter(drops.clone())).unwrap();
            drop(rx);
        }

        assert_eq!(drops.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn received_value_is_not_dropped_twice_by_slot() {
        #[derive(Debug)]
        struct DropCounter(Arc<AtomicUsize>);

        impl Drop for DropCounter {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::Relaxed);
            }
        }

        let drops = Arc::new(AtomicUsize::new(0));
        {
            let (tx, rx) = channel();
            tx.send(DropCounter(drops.clone())).unwrap();

            let value = rx.receive().unwrap();
            assert_eq!(drops.load(Ordering::Relaxed), 0);

            drop(value);
            assert_eq!(drops.load(Ordering::Relaxed), 1);
        }

        assert_eq!(drops.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn stress_send_receive_across_threads() {
        const N: usize = 10_000;

        for i in 0..N {
            let (tx, rx) = channel();

            let sender = thread::spawn(move || {
                tx.send(i).unwrap();
            });
            let receiver = thread::spawn(move || rx.receive().unwrap());
            sender.join().unwrap();

            assert_eq!(receiver.join().unwrap(), i);
        }
    }

    #[test]
    fn stress_sender_dropped_before_send() {
        const N: usize = 10_000;

        for _ in 0..N {
            let (tx, rx) = channel::<usize>();

            let receiver = thread::spawn(move || rx.receive());
            drop(tx);

            assert_eq!(receiver.join().unwrap(), None);
        }
    }

    #[test]
    fn stress_receiver_dropped_before_send() {
        const N: usize = 10_000;

        for i in 0..N {
            let (tx, rx) = channel();

            let receiver = thread::spawn(move || {
                drop(rx);
            });
            receiver.join().unwrap();

            assert_eq!(tx.send(i), Err(i));
        }
    }
}
