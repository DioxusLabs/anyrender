use peniko::Color;
use vello_gpu::RenderSettings;
use wgpu::{Features, Limits};

#[derive(Clone)]
#[non_exhaustive]
pub struct VelloHybridRendererOptions {
    pub features: Option<Features>,
    pub limits: Option<Limits>,
    pub render_settings: RenderSettings,
    pub base_color: Color,
    /// Alpha mode used when compositing the window surface.
    pub composite_alpha_mode: anyrender::CompositeAlphaMode,
    /// Maximum number of frames the presentation engine may queue ahead of the
    /// display. Each queued frame requires a window-sized swapchain surface, so
    /// lower values reduce memory usage and input latency at the cost of less
    /// buffering to absorb slow frames.
    pub desired_maximum_frame_latency: u32,
    /// Whether to cache rasterized glyphs in Vello's glyph atlas.
    ///
    /// Defaults to `false`.
    ///
    /// Note: Vello considers atlas-backed glyph caching experimental.
    pub glyph_caching: bool,
}

impl Default for VelloHybridRendererOptions {
    fn default() -> Self {
        Self {
            features: None,
            limits: None,
            render_settings: RenderSettings::default(),
            base_color: Color::WHITE,
            composite_alpha_mode: anyrender::CompositeAlphaMode::Auto,
            desired_maximum_frame_latency: 1,
            glyph_caching: false,
        }
    }
}

impl VelloHybridRendererOptions {
    pub fn new() -> Self {
        // Within default of RenderSettings there are calls to non const methods so no const for new
        Self::default()
    }

    pub const fn features(self, features: Features) -> Self {
        Self {
            features: Some(features),
            ..self
        }
    }

    pub const fn limits(self, limits: Limits) -> Self {
        Self {
            limits: Some(limits),
            ..self
        }
    }

    pub const fn render_settings(self, render_settings: RenderSettings) -> Self {
        Self {
            render_settings,
            ..self
        }
    }

    pub const fn base_color(self, base_color: Color) -> Self {
        Self { base_color, ..self }
    }

    pub const fn composite_alpha_mode(
        self,
        composite_alpha_mode: anyrender::CompositeAlphaMode,
    ) -> Self {
        Self {
            composite_alpha_mode,
            ..self
        }
    }

    pub const fn desired_maximum_frame_latency(self, desired_maximum_frame_latency: u32) -> Self {
        Self {
            desired_maximum_frame_latency,
            ..self
        }
    }

    /// Enable or disable caching of rasterized glyphs in Vello's glyph atlas.
    ///
    /// Defaults to `false`.
    ///
    /// Note: Vello considers atlas-backed glyph caching experimental.
    pub const fn glyph_caching(self, glyph_caching: bool) -> Self {
        Self {
            glyph_caching,
            ..self
        }
    }
}

impl From<anyrender::RendererConfig> for VelloHybridRendererOptions {
    fn from(config: anyrender::RendererConfig) -> Self {
        Self {
            base_color: config.base_color.unwrap_or(Color::WHITE),
            composite_alpha_mode: config.composite_alpha_mode.unwrap_or_default(),
            ..Default::default()
        }
    }
}
