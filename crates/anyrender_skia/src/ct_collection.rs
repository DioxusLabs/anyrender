//! Loads a face of a font collection (TTC) through CoreText without copying it out.
//!
//! Skia's CoreText backend only opens index 0 from bytes. CoreText itself can open every face of a
//! collection from one `CFData`, which wraps the blob's bytes without copying them.

use std::ffi::c_void;

use peniko::{Blob, FontData};
use skia_safe::Typeface;
use skrifa::raw::TableProvider;
use skrifa::string::StringId;
use skrifa::{FontRef, MetadataProvider};

type CFTypeRef = *const c_void;
type CFIndex = isize;

#[repr(C)]
struct CFAllocatorContext {
    version: CFIndex,
    info: *mut c_void,
    retain: *const c_void,
    release: *const c_void,
    copy_description: *const c_void,
    allocate: *const c_void,
    reallocate: *const c_void,
    deallocate: Option<unsafe extern "C" fn(ptr: *mut c_void, info: *mut c_void)>,
    preferred_size: *const c_void,
}

#[link(name = "CoreFoundation", kind = "framework")]
unsafe extern "C" {
    fn CFAllocatorCreate(allocator: CFTypeRef, context: *mut CFAllocatorContext) -> CFTypeRef;
    fn CFDataCreateWithBytesNoCopy(
        allocator: CFTypeRef,
        bytes: *const u8,
        length: CFIndex,
        bytes_deallocator: CFTypeRef,
    ) -> CFTypeRef;
    fn CFArrayGetCount(array: CFTypeRef) -> CFIndex;
    fn CFArrayGetValueAtIndex(array: CFTypeRef, index: CFIndex) -> CFTypeRef;
    fn CFRelease(cf: CFTypeRef);
    fn CFStringGetLength(string: CFTypeRef) -> CFIndex;
    fn CFStringGetCString(string: CFTypeRef, buffer: *mut u8, size: CFIndex, encoding: u32) -> u8;
}

#[link(name = "CoreText", kind = "framework")]
unsafe extern "C" {
    static kCTFontNameAttribute: CFTypeRef;
    fn CTFontManagerCreateFontDescriptorsFromData(data: CFTypeRef) -> CFTypeRef;
    fn CTFontDescriptorCopyAttribute(descriptor: CFTypeRef, attribute: CFTypeRef) -> CFTypeRef;
    fn CTFontCreateWithFontDescriptor(
        descriptor: CFTypeRef,
        size: f64,
        matrix: *const c_void,
    ) -> CFTypeRef;
}

const K_CF_STRING_ENCODING_UTF8: u32 = 0x0800_0100;

unsafe extern "C" fn drop_blob(_ptr: *mut c_void, info: *mut c_void) {
    drop(unsafe { Box::from_raw(info.cast::<Blob<u8>>()) });
}

/// A `CFData` over the blob's bytes that keeps a clone of the blob alive until CoreText is done.
fn cf_data_no_copy(blob: &Blob<u8>) -> Option<CFTypeRef> {
    let bytes = blob.data();
    let mut context = CFAllocatorContext {
        version: 0,
        info: Box::into_raw(Box::new(blob.clone())).cast(),
        retain: std::ptr::null(),
        release: std::ptr::null(),
        copy_description: std::ptr::null(),
        allocate: std::ptr::null(),
        reallocate: std::ptr::null(),
        deallocate: Some(drop_blob),
        preferred_size: std::ptr::null(),
    };
    unsafe {
        let deallocator = CFAllocatorCreate(std::ptr::null(), &mut context);
        if deallocator.is_null() {
            drop_blob(std::ptr::null_mut(), context.info);
            return None;
        }
        let data = CFDataCreateWithBytesNoCopy(
            std::ptr::null(),
            bytes.as_ptr(),
            bytes.len() as CFIndex,
            deallocator,
        );
        CFRelease(deallocator);
        (!data.is_null()).then_some(data)
    }
}

fn cf_string(string: CFTypeRef) -> Option<String> {
    let mut buffer = vec![0u8; unsafe { CFStringGetLength(string) } as usize * 4 + 1];
    let ok = unsafe {
        CFStringGetCString(
            string,
            buffer.as_mut_ptr(),
            buffer.len() as CFIndex,
            K_CF_STRING_ENCODING_UTF8,
        )
    };
    let len = buffer.iter().position(|&b| b == 0)?;
    buffer.truncate(len);
    (ok != 0).then(|| String::from_utf8(buffer).ok()).flatten()
}

fn descriptor_postscript_name(descriptor: CFTypeRef) -> Option<String> {
    unsafe {
        let name = CTFontDescriptorCopyAttribute(descriptor, kCTFontNameAttribute);
        if name.is_null() {
            return None;
        }
        let string = cf_string(name);
        CFRelease(name);
        string
    }
}

/// The face's PostScript name (name ID 6), if it has one.
fn postscript_name(font: &FontRef<'_>) -> Option<String> {
    font.name().ok()?;
    let name = font
        .localized_strings(StringId::POSTSCRIPT_NAME)
        .english_or_first()?;
    Some(name.chars().collect())
}

/// Opens face `font.index` of a collection, or `None` if CoreText cannot open it from the bytes or
/// no CoreText face can be matched to that index by PostScript name.
pub(crate) fn typeface_from_collection(font: &FontData) -> Option<Typeface> {
    let face = FontRef::from_index(font.data.data(), font.index).ok()?;
    let name = postscript_name(&face);

    unsafe {
        let data = cf_data_no_copy(&font.data)?;
        let descriptors = CTFontManagerCreateFontDescriptorsFromData(data);
        CFRelease(data);
        if descriptors.is_null() {
            return None;
        }

        // CoreText returns the faces in collection order; check by name so a skipped face
        // cannot shift the indices.
        let count = CFArrayGetCount(descriptors);
        let index = font.index as CFIndex;
        let matches = |i: CFIndex| match &name {
            Some(name) => {
                descriptor_postscript_name(CFArrayGetValueAtIndex(descriptors, i)).as_ref()
                    == Some(name)
            }
            None => i == index,
        };
        let found = if index < count && matches(index) {
            Some(index)
        } else if name.is_some() {
            (0..count).find(|&i| matches(i))
        } else {
            None
        };

        let typeface = found.and_then(|i| {
            let ct_font = CTFontCreateWithFontDescriptor(
                CFArrayGetValueAtIndex(descriptors, i),
                0.0,
                std::ptr::null(),
            );
            if ct_font.is_null() {
                return None;
            }
            let typeface = Typeface::from_ct_font(ct_font);
            CFRelease(ct_font);
            typeface
        });
        CFRelease(descriptors);
        typeface
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[link(name = "CoreText", kind = "framework")]
    unsafe extern "C" {
        fn CTFontCopyTable(font: CFTypeRef, tag: u32, options: u32) -> CFTypeRef;
    }
    #[link(name = "CoreFoundation", kind = "framework")]
    unsafe extern "C" {
        fn CFDataGetBytePtr(data: CFTypeRef) -> *const u8;
    }

    fn face_count(data: &[u8]) -> u32 {
        match skrifa::raw::FileRef::new(data).unwrap() {
            skrifa::raw::FileRef::Collection(collection) => collection.len(),
            skrifa::raw::FileRef::Font(_) => 1,
        }
    }

    #[test]
    fn opens_every_face_of_system_collections_without_copying() {
        for path in [
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/AppleSDGothicNeo.ttc",
            "/System/Library/Fonts/Menlo.ttc",
            "/System/Library/Fonts/Apple Color Emoji.ttc",
        ] {
            let Ok(bytes) = std::fs::read(path) else {
                continue;
            };
            let blob = Blob::new(Arc::new(bytes));
            let range = blob.data().as_ptr_range();
            for index in 0..face_count(blob.data()) {
                let font = FontData::new(blob.clone(), index);
                let typeface = typeface_from_collection(&font).expect(path);
                let face = FontRef::from_index(blob.data(), index).unwrap();
                assert_eq!(
                    typeface.post_script_name(),
                    postscript_name(&face),
                    "{path} {index}"
                );

                let tag = u32::from_be_bytes(*b"head");
                let expected = face
                    .table_data(skrifa::raw::types::Tag::new(b"head"))
                    .unwrap();
                let table = unsafe { CTFontCopyTable(typeface.ct_font().unwrap(), tag, 0) };
                let ptr = unsafe { CFDataGetBytePtr(table) };
                assert!(range.contains(&ptr), "{path} {index}: table was copied");
                assert_eq!(ptr, expected.as_bytes().as_ptr(), "{path} {index}");
                unsafe { CFRelease(table) };

                let mut surface = skia_safe::surfaces::raster_n32_premul((64, 64)).unwrap();
                let skia_font = skia_safe::Font::from_typeface(typeface, 48.0);
                let glyph = ['H', '\u{1F600}']
                    .into_iter()
                    .map(|c| skia_font.unichar_to_glyph(c as i32))
                    .find(|&glyph| glyph != 0)
                    .unwrap_or(0);
                assert_ne!(glyph, 0, "{path} {index}");
                let paint = skia_safe::Paint::default();
                surface.canvas().draw_glyphs_at(
                    &[glyph],
                    &[skia_safe::Point::new(4.0, 52.0)][..],
                    (0.0, 0.0),
                    &skia_font,
                    &paint,
                );
                let pixmap = surface.peek_pixels().unwrap();
                let inked = pixmap
                    .bytes()
                    .unwrap()
                    .chunks(4)
                    .filter(|px| px[3] != 0)
                    .count();
                assert!(inked > 50, "{path} {index}: nothing drawn");
            }
        }
    }
}

#[cfg(test)]
mod cache_tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn separate_blobs_with_same_names_get_separate_typefaces() {
        let a = std::fs::read("/System/Library/Fonts/Helvetica.ttc").unwrap();
        let mut b = a.clone();
        // Change one byte of face 1's glyph data, leaving every name the same.
        let face = FontRef::from_index(&a, 1).unwrap();
        let glyf = face
            .table_data(skrifa::raw::types::Tag::new(b"glyf"))
            .unwrap();
        let offset = glyf.as_bytes().as_ptr() as usize - a.as_ptr() as usize;
        b[offset + 100] ^= 0xff;
        let a = Blob::new(Arc::new(a));
        let b = Blob::new(Arc::new(b));
        let ta = typeface_from_collection(&FontData::new(a.clone(), 1)).unwrap();
        let tb = typeface_from_collection(&FontData::new(b.clone(), 1)).unwrap();
        let tb_same = typeface_from_collection(&FontData::new(b.clone(), 1)).unwrap();
        eprintln!(
            "ids: a={} b={} b again={}",
            ta.unique_id(),
            tb.unique_id(),
            tb_same.unique_id()
        );
        let glyf_tag = u32::from_be_bytes(*b"glyf");
        let tb_glyf = tb.copy_table_data(glyf_tag).unwrap();
        assert_eq!(
            tb_glyf.as_bytes()[100],
            b.data()[offset + 100],
            "typeface for blob b reads blob a's bytes"
        );
        assert_ne!(ta.unique_id(), tb.unique_id());
    }

    /// `SkMakeTypefaceFromCTFont` adds every typeface to Skia's global typeface cache, which only
    /// drops unreferenced entries once it holds `typeface_cache_count_limit` (1024) of them.
    #[test]
    fn releases_blob_once_skia_no_longer_caches_typeface() {
        let bytes = Arc::new(std::fs::read("/System/Library/Fonts/Menlo.ttc").unwrap());
        let weak = Arc::downgrade(&bytes);
        let font = FontData::new(Blob::new(bytes), 2);
        let old = skia_safe::graphics::set_typeface_cache_count_limit(0);
        let typeface = typeface_from_collection(&font).unwrap();
        skia_safe::graphics::set_typeface_cache_count_limit(old);
        drop(font);
        assert_eq!(typeface.post_script_name().as_deref(), Some("Menlo-Italic"));
        assert!(weak.upgrade().is_some());
        drop(typeface);
        assert!(
            weak.upgrade().is_none(),
            "CoreText did not release the blob"
        );
    }
}
