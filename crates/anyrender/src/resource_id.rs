use std::sync::atomic::AtomicUsize;

static ID_COUNTER: AtomicUsize = AtomicUsize::new(0);

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ResourceId(u64);

impl ResourceId {
    pub fn new() -> Self {
        Self(ID_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed) as u64)
    }

    pub fn into_ffi(&self) -> u64 {
        self.0
    }

    pub fn from_ffi(raw: u64) -> Self {
        Self(raw)
    }
}
