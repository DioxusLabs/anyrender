//! Loads a face of a font collection (TTC) through CoreText without copying it out.
//!
//! Skia's CoreText backend only opens index 0 from bytes. CoreText itself can open every face of a
//! collection from one `CFData`, which wraps the blob's bytes without copying them.

use std::ffi::c_void;
use std::ptr::NonNull;

use objc2_core_foundation::{
    CFAllocator, CFAllocatorContext, CFArray, CFData, CFIndex, CFRetained, CFString, CFType,
};
use peniko::{Blob, FontData};
#[cfg(test)]
use skia_safe::Data;
use skia_safe::Typeface;
use skrifa::raw::TableProvider;
use skrifa::string::StringId;
use skrifa::{FontRef, MetadataProvider};

#[link(name = "CoreText", kind = "framework")]
unsafe extern "C" {
    static kCTFontNameAttribute: &'static CFString;
    fn CTFontManagerCreateFontDescriptorsFromData(data: &CFData) -> Option<NonNull<CFArray>>;
    fn CTFontDescriptorCopyAttribute(
        descriptor: &CFType,
        attribute: &CFString,
    ) -> Option<NonNull<CFType>>;
    fn CTFontCreateWithFontDescriptor(
        descriptor: &CFType,
        size: f64,
        matrix: *const c_void,
    ) -> Option<NonNull<CFType>>;
}

unsafe extern "C-unwind" fn release_blob(info: *const c_void) {
    drop(unsafe { Box::from_raw(info.cast_mut().cast::<Blob<u8>>()) });
}

/// A `CFData` over the blob's bytes. Its deallocator owns a clone of the blob and drops it when
/// the deallocator itself is freed: after the data, or straight away if the data is not created.
fn cf_data_no_copy(blob: &Blob<u8>) -> Option<CFRetained<CFData>> {
    let bytes = blob.data();
    let info = Box::into_raw(Box::new(blob.clone()));
    let mut context = CFAllocatorContext {
        version: 0,
        info: info.cast(),
        retain: None,
        release: Some(release_blob),
        copyDescription: None,
        allocate: None,
        reallocate: None,
        deallocate: None,
        preferredSize: None,
    };
    unsafe {
        let Some(deallocator) = CFAllocator::new(None, &mut context) else {
            release_blob(info.cast());
            return None;
        };
        CFData::with_bytes_no_copy(
            None,
            bytes.as_ptr(),
            bytes.len() as CFIndex,
            Some(&deallocator),
        )
    }
}

fn descriptor_postscript_name(descriptor: &CFType) -> Option<String> {
    let name = unsafe { CTFontDescriptorCopyAttribute(descriptor, kCTFontNameAttribute) }?;
    let name = unsafe { CFRetained::from_raw(name) };
    Some(name.downcast_ref::<CFString>()?.to_string())
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
/// no CoreText face can be matched to it by PostScript name.
pub(crate) fn typeface_from_collection(font: &FontData) -> Option<Typeface> {
    let face = FontRef::from_index(font.data.data(), font.index).ok()?;
    // Without a name the index is no guide: a variable face has a descriptor per named instance.
    let name = Some(postscript_name(&face)?);

    let data = cf_data_no_copy(&font.data)?;
    let descriptors =
        unsafe { CFRetained::from_raw(CTFontManagerCreateFontDescriptorsFromData(&data)?) };
    let descriptor = |i: CFIndex| unsafe { &*descriptors.value_at_index(i).cast::<CFType>() };
    let matches = |i: CFIndex| descriptor_postscript_name(descriptor(i)) == name;

    // CoreText returns the faces in collection order, but gives a variable face one descriptor
    // per named instance, so search by PostScript name where the index misses.
    let count = descriptors.count();
    let index = font.index as CFIndex;
    let found = if index < count && matches(index) {
        index
    } else {
        (0..count).find(|&i| matches(i))?
    };

    let ct_font = unsafe {
        CFRetained::from_raw(CTFontCreateWithFontDescriptor(
            descriptor(found),
            0.0,
            std::ptr::null(),
        )?)
    };
    unsafe { Typeface::from_ct_font(CFRetained::as_ptr(&ct_font).as_ptr().cast()) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};
    use std::sync::Arc;

    #[link(name = "CoreText", kind = "framework")]
    unsafe extern "C" {
        fn CTFontCopyTable(font: *const c_void, tag: u32, options: u32) -> Option<NonNull<CFData>>;
    }

    fn find_collections(dir: &Path, out: &mut Vec<PathBuf>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for path in entries.flatten().map(|entry| entry.path()) {
            if path.is_dir() {
                find_collections(&path, out);
            } else if path
                .extension()
                .and_then(|ext| ext.to_str())
                .is_some_and(|ext| {
                    ext.eq_ignore_ascii_case("ttc") || ext.eq_ignore_ascii_case("otc")
                })
            {
                out.push(path);
            }
        }
    }

    fn variation_position(typeface: &Typeface) -> Vec<(u32, f32)> {
        typeface
            .variation_design_position()
            .unwrap_or_default()
            .iter()
            .map(|coord| (*coord.axis, coord.value))
            .collect()
    }

    fn assert_draws(typeface: Typeface, what: &str) {
        let font = skia_safe::Font::from_typeface(typeface, 48.0);
        let Some(glyph) = ['H', '0', '\u{1F600}', '\u{4E00}', '\u{0915}']
            .into_iter()
            .map(|c| font.unichar_to_glyph(c as i32))
            .find(|&glyph| glyph != 0)
        else {
            return;
        };
        let mut surface = skia_safe::surfaces::raster_n32_premul((64, 64)).unwrap();
        surface.canvas().draw_glyphs_at(
            &[glyph],
            &[skia_safe::Point::new(4.0, 52.0)][..],
            (0.0, 0.0),
            &font,
            &skia_safe::Paint::default(),
        );
        let pixmap = surface.peek_pixels().unwrap();
        let inked = pixmap
            .bytes()
            .unwrap()
            .chunks(4)
            .filter(|px| px[3] != 0)
            .count();
        assert!(inked > 0, "{what}: nothing drawn");
    }

    /// Every face of every system collection opens without copying, as the same face (by
    /// PostScript name and variation position) that the copying fallback would give.
    #[test]
    fn opens_every_face_of_system_collections_without_copying() {
        let mut paths = Vec::new();
        for dir in ["/System/Library/Fonts", "/System/Library/AssetsV2"] {
            find_collections(Path::new(dir), &mut paths);
        }
        let font_mgr = skia_safe::FontMgr::new();
        for path in &paths {
            let blob = Blob::new(Arc::new(std::fs::read(path).unwrap()));
            let range = blob.data().as_ptr_range();
            let Some(collection) = oaty::Collection::new(blob.data()) else {
                continue;
            };
            let skrifa::raw::FileRef::Collection(faces) =
                skrifa::raw::FileRef::new(blob.data()).unwrap()
            else {
                continue;
            };
            for index in 0..faces.len() {
                let what = format!("{} {index}", path.display());
                let typeface = typeface_from_collection(&FontData::new(blob.clone(), index))
                    .unwrap_or_else(|| panic!("{what}: not opened"));
                let face = FontRef::from_index(blob.data(), index).unwrap();
                assert_eq!(
                    typeface.post_script_name(),
                    postscript_name(&face),
                    "{what}"
                );

                let copied = collection
                    .get_font(index)
                    .and_then(|font| font.copy_data())
                    .and_then(|data| font_mgr.new_from_data(Data::new_copy(&data), 0))
                    .unwrap();
                assert_eq!(
                    variation_position(&typeface),
                    variation_position(&copied),
                    "{what}"
                );

                let head = face
                    .table_data(skrifa::raw::types::Tag::new(b"head"))
                    .unwrap();
                let table = unsafe {
                    CFRetained::from_raw(
                        CTFontCopyTable(
                            typeface.ct_font().unwrap(),
                            u32::from_be_bytes(*b"head"),
                            0,
                        )
                        .unwrap(),
                    )
                };
                let ptr = table.byte_ptr();
                assert!(range.contains(&ptr), "{what}: table was copied");
                assert_eq!(ptr, head.as_bytes().as_ptr(), "{what}");

                assert_draws(typeface, &what);
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
