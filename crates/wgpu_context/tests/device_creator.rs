//! Needs a wgpu adapter (lavapipe is enough); the tests skip themselves when there is none.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use wgpu_context::{DeviceCreator, SharedWGPUContext, WGPUContext};

fn counting_creator(calls: Arc<AtomicUsize>) -> DeviceCreator {
    DeviceCreator::new(move |adapter, descriptor| {
        calls.fetch_add(1, Ordering::SeqCst);
        Ok(pollster::block_on(adapter.request_device(descriptor))?)
    })
}

fn adapter_available() -> bool {
    let available = pollster::block_on(
        WGPUContext::new()
            .instance
            .request_adapter(&Default::default()),
    )
    .is_ok();
    if !available {
        eprintln!("no wgpu adapter, skipping");
    }
    available
}

#[test]
fn context_creates_devices_with_its_creator() {
    if !adapter_available() {
        return;
    }
    let calls = Arc::new(AtomicUsize::new(0));
    let context = SharedWGPUContext::new(WGPUContext::with_device_creator(
        None,
        None,
        counting_creator(calls.clone()),
    ));

    // Every user of the shared context finds the device the hook created
    for _ in 0..2 {
        let mut context = context.lock();
        assert_eq!(
            pollster::block_on(context.find_or_create_device(None)).unwrap(),
            0
        );
    }
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(context.lock().device_pool.len(), 1);

    // A device created outside the pool (as the window renderers do) goes through the hook too
    let device_handle = pollster::block_on(context.lock().create_device_handle(None)).unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    drop(device_handle);
}

#[test]
fn failing_creator_falls_back_to_request_device() {
    if !adapter_available() {
        return;
    }
    let calls = Arc::new(AtomicUsize::new(0));
    let creator = {
        let calls = calls.clone();
        DeviceCreator::new(move |_, _| {
            calls.fetch_add(1, Ordering::SeqCst);
            Err("no video support".into())
        })
    };
    let mut context = WGPUContext::with_device_creator(None, None, creator);

    assert_eq!(
        pollster::block_on(context.find_or_create_device(None)).unwrap(),
        0
    );
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[test]
fn contexts_without_a_creator_have_none() {
    assert!(WGPUContext::new().device_creator().is_none());
    let creator = DeviceCreator::new(|_, _| Err("unused".into()));
    assert!(
        WGPUContext::with_device_creator(None, None, creator)
            .device_creator()
            .is_some()
    );
}
