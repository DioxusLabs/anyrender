//! Wall-clock timing of a frame's GPU work, for `log_frame_times`-style debug output.

#[cfg(feature = "log_frame_times")]
mod real {
    use std::io::{Write, stdout};
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant};
    use wgpu::Queue;

    /// Measures the wall-clock time from a frame's first GPU submission until the GPU has
    /// finished its last one, using `Queue::on_submitted_work_done`. The callback runs during a
    /// later `Device::poll`/`Queue::submit`, so a frame's time is available one frame late.
    /// Any swapchain wait made between the submissions is included, so this bounds GPU execution
    /// time from above; timestamp queries are needed for exact GPU time.
    pub struct GpuFrameTimer {
        frame_start: Option<Instant>,
        completed: Arc<Mutex<Option<Duration>>>,
    }

    impl GpuFrameTimer {
        pub fn new() -> Self {
            Self {
                frame_start: None,
                completed: Arc::new(Mutex::new(None)),
            }
        }

        /// Mark the start of this frame's GPU work. Call before the frame's first submission.
        pub fn start_frame(&mut self) {
            self.frame_start = Some(Instant::now());
        }

        /// Register for completion of this frame's GPU work. Call after the frame's last submission.
        pub fn end_frame(&mut self, queue: &Queue) {
            let Some(start) = self.frame_start.take() else {
                return;
            };
            let completed = self.completed.clone();
            queue.on_submitted_work_done(move || {
                *completed.lock().unwrap() = Some(start.elapsed());
            });
        }

        /// Print the time taken by the most recently completed frame, if one has completed
        /// since the last call.
        pub fn print_completed(&self, message: &str) {
            if let Some(duration) = self.completed.lock().unwrap().take() {
                let ms = duration.as_secs_f64() * 1000.0;
                let mut out = stdout().lock();
                writeln!(out, "{message}gpu: {ms:.2}ms (previous frame)").unwrap();
            }
        }
    }
}

#[cfg(not(feature = "log_frame_times"))]
mod dummy {
    use wgpu::Queue;

    /// No-op stand-in for the `log_frame_times` GPU frame timer.
    pub struct GpuFrameTimer;

    impl GpuFrameTimer {
        #[inline(always)]
        pub fn new() -> Self {
            Self
        }
        #[inline(always)]
        pub fn start_frame(&mut self) {}
        #[inline(always)]
        pub fn end_frame(&mut self, _queue: &Queue) {}
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
