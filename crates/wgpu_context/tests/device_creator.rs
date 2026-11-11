//! Needs a wgpu adapter (lavapipe is enough); the tests skip themselves when there is none.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use wgpu_context::{DeviceCreator, DeviceHandle, WGPUContext};

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
    let mut context = WGPUContext::new().with_device_creator(Some(counting_creator(calls.clone())));

    assert_eq!(
        pollster::block_on(context.find_or_create_device(None)).unwrap(),
        0
    );
    assert_eq!(
        pollster::block_on(context.find_or_create_device(None)).unwrap(),
        0
    );
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(context.device_pool.len(), 1);
}

#[test]
fn from_device_handle_reuses_the_device_and_creates_later_devices_with_the_creator() {
    if !adapter_available() {
        return;
    }
    let existing = pollster::block_on(DeviceHandle::new_from_compatible_surface(
        WGPUContext::new().instance,
        None,
        None,
        None,
    ))
    .unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let mut context = WGPUContext::from_device_handle(existing.clone())
        .with_device_creator(Some(counting_creator(calls.clone())));

    // The device the context was created from is found without creating a new one
    let idx = pollster::block_on(context.find_or_create_device(None)).unwrap();
    assert!(context.device_pool[idx].device == existing.device);
    assert_eq!(calls.load(Ordering::SeqCst), 0);

    // A later device (e.g. for a surface the existing device can't present to) goes through
    // the creator, with the existing device's features
    let later = pollster::block_on(context.create_device_handle(None)).unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert!(later.device.features().contains(existing.device.features()));
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
    let mut context = WGPUContext::new().with_device_creator(Some(creator));

    assert_eq!(
        pollster::block_on(context.find_or_create_device(None)).unwrap(),
        0
    );
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}
