//! Rendering of text nodes based on `usvg`'s layouted glyphs.

use std::collections::HashMap;
use std::sync::Arc;

use crate::util;
use anyrender::{Glyph, NormalizedCoord, PaintScene};
use kurbo::{Affine, Vec2};
use peniko::{Blob, FontData, StyleRef};
use skrifa::raw::TableProvider as _;
use skrifa::{FontRef, MetadataProvider as _, Tag};
use usvg::fontdb;
use usvg::layout::{PositionedGlyph, Span};

const OPSZ: Tag = Tag::new(b"opsz");

#[derive(Clone)]
struct CachedFont {
    data: FontData,
    upem: u16,
    has_axes: bool,
    has_opsz: bool,
}

pub(crate) struct TextRenderer<'a> {
    fontdb: &'a fontdb::Database,
    fonts: HashMap<fontdb::ID, Option<CachedFont>>,
}

impl<'a> TextRenderer<'a> {
    pub(crate) fn new(fontdb: &'a fontdb::Database) -> Self {
        Self {
            fontdb,
            fonts: HashMap::new(),
        }
    }

    fn font(&mut self, id: fontdb::ID) -> Option<CachedFont> {
        let fontdb = self.fontdb;
        self.fonts
            .entry(id)
            .or_insert_with(|| load_font(fontdb, id))
            .clone()
    }

    pub(crate) fn render_text<S: PaintScene, F: FnMut(&mut S, &usvg::Node)>(
        &mut self,
        scene: &mut S,
        node: &usvg::Node,
        text: &usvg::Text,
        transform: Affine,
        error_handler: &mut F,
    ) {
        for span in text.layouted() {
            if !span.visible {
                continue;
            }

            for path in [&span.overline, &span.underline].into_iter().flatten() {
                render_decoration(scene, node, path, transform, error_handler);
            }

            self.render_span_glyphs(scene, node, span, transform, error_handler);

            if let Some(path) = &span.line_through {
                render_decoration(scene, node, path, transform, error_handler);
            }
        }
    }

    fn render_span_glyphs<S: PaintScene, F: FnMut(&mut S, &usvg::Node)>(
        &mut self,
        scene: &mut S,
        node: &usvg::Node,
        span: &Span,
        transform: Affine,
        error_handler: &mut F,
    ) {
        let font_size = span.font_size.get();

        let mut run_font: Option<CachedFont> = None;
        let mut run_font_id: Option<fontdb::ID> = None;
        let mut run_coords: Vec<NormalizedCoord> = Vec::new();
        let mut run_glyphs: Vec<Glyph> = Vec::new();

        macro_rules! flush_run {
            () => {
                if let Some(font) = run_font.take() {
                    let glyphs = std::mem::take(&mut run_glyphs);
                    draw_glyph_run(
                        scene,
                        node,
                        span,
                        &font,
                        &run_coords,
                        transform,
                        Affine::IDENTITY,
                        font_size,
                        &glyphs,
                        error_handler,
                    );
                }
            };
        }

        for glyph in &span.positioned_glyphs {
            let Some(font) = self.font(glyph.font) else {
                flush_run!();
                error_handler(scene, node);
                continue;
            };
            let coords = normalized_coords(&font, span, glyph);

            let glyph_ts = util::to_affine(&glyph.transform());
            match batched_position(&glyph_ts, font_size, font.upem) {
                Some((x, y)) => {
                    if run_font.is_none() || run_font_id != Some(glyph.font) || run_coords != coords
                    {
                        flush_run!();
                        run_font_id = Some(glyph.font);
                        run_font = Some(font);
                        run_coords = coords;
                    }
                    run_glyphs.push(Glyph {
                        id: glyph.id.0,
                        x,
                        y,
                    });
                }
                None => {
                    flush_run!();
                    // The glyph has a transform that cannot be represented as a
                    // glyph position within a run (e.g. rotation for text on a
                    // path), so draw it as a single-glyph run with the transform
                    // folded into the run transform.
                    let run_ts = glyph_ts * Affine::scale(font.upem as f64 / font_size as f64);
                    draw_glyph_run(
                        scene,
                        node,
                        span,
                        &font,
                        &coords,
                        transform * run_ts,
                        run_ts.inverse(),
                        font_size,
                        &[Glyph {
                            id: glyph.id.0,
                            x: 0.0,
                            y: 0.0,
                        }],
                        error_handler,
                    );
                }
            }
        }

        flush_run!();
    }
}

#[allow(clippy::too_many_arguments)]
fn draw_glyph_run<S: PaintScene, F: FnMut(&mut S, &usvg::Node)>(
    scene: &mut S,
    node: &usvg::Node,
    span: &Span,
    font: &CachedFont,
    coords: &[NormalizedCoord],
    transform: Affine,
    brush_space_correction: Affine,
    font_size: f32,
    glyphs: &[Glyph],
    error_handler: &mut F,
) {
    let fill_paint = span
        .fill
        .as_ref()
        .and_then(|fill| util::to_brush(fill.paint(), fill.opacity()))
        .map(|(paint, brush_ts)| apply_brush_transform(paint, brush_space_correction * brush_ts));
    let stroke_paint = span
        .stroke
        .as_ref()
        .and_then(|stroke| util::to_brush(stroke.paint(), stroke.opacity()))
        .map(|(paint, brush_ts)| apply_brush_transform(paint, brush_space_correction * brush_ts));

    // Report unsupported paints (e.g. patterns) to the error handler
    if (span.fill.is_some() && fill_paint.is_none())
        || (span.stroke.is_some() && stroke_paint.is_none())
    {
        error_handler(scene, node);
    }

    let fill_style = span
        .fill
        .as_ref()
        .map(|fill| match fill.rule() {
            usvg::FillRule::NonZero => peniko::Fill::NonZero,
            usvg::FillRule::EvenOdd => peniko::Fill::EvenOdd,
        })
        .unwrap_or(peniko::Fill::NonZero);
    let stroke_style = span.stroke.as_ref().map(util::to_stroke);

    let draw = |scene: &mut S, style: StyleRef<'_>, paint: &anyrender::Paint| {
        scene.draw_glyphs(
            &font.data,
            font_size,
            false,
            coords,
            Vec2::ZERO,
            style,
            paint.as_ref(),
            1.0,
            transform,
            None,
            glyphs.iter().copied(),
        );
    };

    let draw_fill = |scene: &mut S| {
        if let Some(paint) = &fill_paint {
            draw(scene, StyleRef::Fill(fill_style), paint);
        }
    };
    let draw_stroke = |scene: &mut S| {
        if let (Some(paint), Some(stroke)) = (&stroke_paint, &stroke_style) {
            draw(scene, StyleRef::Stroke(stroke), paint);
        }
    };

    match span.paint_order {
        usvg::PaintOrder::FillAndStroke => {
            draw_fill(scene);
            draw_stroke(scene);
        }
        usvg::PaintOrder::StrokeAndFill => {
            draw_stroke(scene);
            draw_fill(scene);
        }
    }
}

/// `PaintScene::draw_glyphs` has no `brush_transform` parameter, so bake the
/// brush transform (used by usvg for gradient units conversion) into the
/// gradient geometry instead. Radius scaling is exact only for uniform scales;
/// non-uniform scales are approximated.
fn apply_brush_transform(mut paint: anyrender::Paint, brush_transform: Affine) -> anyrender::Paint {
    if brush_transform == Affine::IDENTITY {
        return paint;
    }
    if let anyrender::Paint::Gradient(gradient) = &mut paint {
        match &mut gradient.kind {
            peniko::GradientKind::Linear(linear) => {
                linear.start = brush_transform * linear.start;
                linear.end = brush_transform * linear.end;
            }
            peniko::GradientKind::Radial(radial) => {
                radial.start_center = brush_transform * radial.start_center;
                radial.end_center = brush_transform * radial.end_center;
                let [a, b, c, d, _, _] = brush_transform.as_coeffs();
                let scale = (((a * a + b * b).sqrt() + (c * c + d * d).sqrt()) / 2.0) as f32;
                radial.start_radius *= scale;
                radial.end_radius *= scale;
            }
            peniko::GradientKind::Sweep(sweep) => {
                sweep.center = brush_transform * sweep.center;
            }
        }
    }
    paint
}

/// If the glyph's transform is a pure translation + the uniform `font_size / upem`
/// scale that glyph runs already apply, return the glyph position within the run.
fn batched_position(transform: &Affine, font_size: f32, upem: u16) -> Option<(f32, f32)> {
    const EPSILON: f64 = 1e-4;
    let [a, b, c, d, e, f] = transform.as_coeffs();
    let sx = font_size as f64 / upem as f64;
    let is_batchable = b.abs() < EPSILON
        && c.abs() < EPSILON
        && (a - sx).abs() < EPSILON * sx
        && (d - sx).abs() < EPSILON * sx;
    is_batchable.then_some((e as f32, f as f32))
}

fn render_decoration<S: PaintScene, F: FnMut(&mut S, &usvg::Node)>(
    scene: &mut S,
    node: &usvg::Node,
    path: &usvg::Path,
    transform: Affine,
    error_handler: &mut F,
) {
    if !path.is_visible() {
        return;
    }
    let local_path = util::to_bez_path(path);
    match path.paint_order() {
        usvg::PaintOrder::FillAndStroke => {
            crate::render::fill(scene, error_handler, path, transform, &local_path, node);
            crate::render::stroke(scene, error_handler, path, transform, &local_path, node);
        }
        usvg::PaintOrder::StrokeAndFill => {
            crate::render::stroke(scene, error_handler, path, transform, &local_path, node);
            crate::render::fill(scene, error_handler, path, transform, &local_path, node);
        }
    }
}

fn load_font(fontdb: &fontdb::Database, id: fontdb::ID) -> Option<CachedFont> {
    let (source, index) = fontdb.face_source(id)?;
    let data: Blob<u8> = match source {
        fontdb::Source::Binary(data) => Blob::new(data),
        fontdb::Source::File(path) => Blob::new(Arc::new(std::fs::read(path).ok()?)),
        fontdb::Source::SharedFile(_, data) => Blob::new(data),
    };

    let font_ref = FontRef::from_index(data.as_ref(), index).ok()?;
    let upem = font_ref.head().ok()?.units_per_em();
    let axes = font_ref.axes();
    let has_axes = !axes.is_empty();
    let has_opsz = axes.iter().any(|axis| axis.tag() == OPSZ);

    Some(CachedFont {
        data: FontData::new(data, index),
        upem,
        has_axes,
        has_opsz,
    })
}

/// Compute the normalized variation coordinates for a glyph, taking both the span's
/// explicit variations and automatic optical sizing (`font-optical-sizing: auto`)
/// into account.
fn normalized_coords(
    font: &CachedFont,
    span: &Span,
    glyph: &PositionedGlyph,
) -> Vec<NormalizedCoord> {
    if !font.has_axes {
        return Vec::new();
    }

    let auto_opsz = span.font_optical_sizing == usvg::FontOpticalSizing::Auto
        && font.has_opsz
        && !span.variations.iter().any(|v| &v.tag == b"opsz");

    let variations = span
        .variations
        .iter()
        .map(|v| (Tag::new(&v.tag), v.value))
        .chain(auto_opsz.then(|| (OPSZ, glyph.font_size())));

    let Ok(font_ref) = FontRef::from_index(font.data.data.as_ref(), font.data.index) else {
        return Vec::new();
    };
    font_ref
        .axes()
        .location(variations)
        .coords()
        .iter()
        .map(|coord| coord.to_bits())
        .collect()
}
