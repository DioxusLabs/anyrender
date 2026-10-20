//! Wall-clock timing of a frame's GPU work, for `log_frame_times`-style debug output.

#[cfg(feature = "log_frame_times")]
mod real {
    use std::io::{Write, stdout};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant};
    use wgpu::{Device, Queue};

    /// How often the polling thread checks the device while frames are in flight.
    const POLL_INTERVAL: Duration = Duration::from_micros(250);

    /// Measures the wall-clock time from `start` until the GPU has finished the work submitted
    /// by the following `submitted` call, using `Queue::on_submitted_work_done`. CPU work done
    /// between the two calls (e.g. vello's encoding) is included, so this bounds GPU execution
    /// time from above; timestamp queries are needed for exact GPU time.
    ///
    /// wgpu only runs the callback from `Device::poll`/`Queue::submit`, which the render loop
    /// calls about once per frame, so on native a thread polls the device every
    /// `POLL_INTERVAL` while frames are in flight (WebGPU polls by itself). Completed times are
    /// printed on the next frame.
    pub struct GpuFrameTimer {
        start: Option<Instant>,
        shared: Arc<Shared>,
        poller: Option<std::thread::Thread>,
    }

    #[derive(Default)]
    struct Shared {
        completed: Mutex<Vec<(&'static str, Duration)>>,
        in_flight: AtomicUsize,
    }

    impl GpuFrameTimer {
        pub fn new() -> Self {
            Self {
                start: None,
                shared: Arc::new(Shared::default()),
                poller: None,
            }
        }

        /// Mark the start of a piece of GPU work. Call before it is submitted.
        pub fn start(&mut self) {
            self.start = Some(Instant::now());
        }

        /// Register for completion of the work submitted since `start`. Call after its submission.
        pub fn submitted(&mut self, label: &'static str, device: &Device, queue: &Queue) {
            let Some(start) = self.start.take() else {
                return;
            };
            let shared = self.shared.clone();
            shared.in_flight.fetch_add(1, Ordering::AcqRel);
            queue.on_submitted_work_done(move || {
                shared
                    .completed
                    .lock()
                    .unwrap()
                    .push((label, start.elapsed()));
                shared.in_flight.fetch_sub(1, Ordering::AcqRel);
            });
            self.wake_poller(device);
        }

        /// Print the time taken by each piece of work that has completed since the last call.
        pub fn print_completed(&self, message: &str) {
            let completed = std::mem::take(&mut *self.shared.completed.lock().unwrap());
            if !completed.is_empty() {
                let mut out = stdout().lock();
                for (label, duration) in completed {
                    let ms = duration.as_secs_f64() * 1000.0;
                    writeln!(out, "{message}gpu {label}: {ms:.2}ms (previous frame)").unwrap();
                }
            }
        }

        #[cfg(target_arch = "wasm32")]
        fn wake_poller(&mut self, _device: &Device) {}

        #[cfg(not(target_arch = "wasm32"))]
        fn wake_poller(&mut self, device: &Device) {
            let poller = self.poller.get_or_insert_with(|| {
                let device = device.clone();
                let shared = self.shared.clone();
                std::thread::Builder::new()
                    .name("gpu-frame-timer".into())
                    .spawn(move || {
                        loop {
                            if shared.in_flight.load(Ordering::Acquire) == 0 {
                                std::thread::park();
                                continue;
                            }
                            let _ = device.poll(wgpu::PollType::Poll);
                            std::thread::sleep(POLL_INTERVAL);
                        }
                    })
                    .unwrap()
                    .thread()
                    .clone()
            });
            poller.unpark();
        }
    }
}

#[cfg(not(feature = "log_frame_times"))]
mod dummy {
    use wgpu::{Device, Queue};

    /// No-op stand-in for the `log_frame_times` GPU frame timer.
    pub struct GpuFrameTimer;

    impl GpuFrameTimer {
        #[inline(always)]
        pub fn new() -> Self {
            Self
        }
        #[inline(always)]
        pub fn start(&mut self) {}
        #[inline(always)]
        pub fn submitted(&mut self, _label: &'static str, _device: &Device, _queue: &Queue) {}
        #[inline(always)]
        pub fn print_completed(&self, _message: &str) {}
    }
}

#[cfg(not(feature = "log_frame_times"))]
pub use dummy::GpuFrameTimer;
#[cfg(feature = "log_frame_times")]
pub use real::GpuFrameTimer;

impl Default for GpuFrameTimer {
    fn default() -> Self {
        Self::new()
    }
}
