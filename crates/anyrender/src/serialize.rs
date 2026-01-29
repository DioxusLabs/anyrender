use std::{io::BufWriter, io::Write};

use peniko::{FontData, ImageData};
use serde::{Deserialize, Serialize};
use serde_jsonlines::JsonLinesWriter;

use crate::recording::RenderCommand;

#[derive(Copy, Clone, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ResourceId(usize);

#[derive(Copy, Clone, Serialize, Deserialize)]
pub enum ResourceKind {
    Image,
    Font,
}

#[derive(Serialize, Deserialize)]
pub struct Resource {
    /// The kind of resource (image, font, etc)
    kind: ResourceKind,
    /// The size of the resource in bytes
    size: usize,
    /// SHA-256 hash of the resource's raw content
    sha256_hash: Option<String>,
}

pub type SerializableRenderCommand = RenderCommand<ResourceId, ResourceId>;

#[derive(Serialize, Deserialize)]
pub struct SerializableSceneHeader {
    /// The number of commands in the file
    command_count: usize,
    /// When the scene was serialized
    created_at: u64,

    /// Metadata for the resources
    resources: Vec<Resource>,
}

pub struct SerializableScene {
    pub header: SerializableSceneHeader,
    pub commands: Vec<SerializableRenderCommand>,
    pub fonts: Vec<FontData>,
    pub images: Vec<ImageData>,
}

struct SceneSerializer<'s, W: Write> {
    writer: BufWriter<W>,
    scene: &'s SerializableScene,
}

impl SerializableScene {
    pub fn serialize<W: Write>(&self, writer: BufWriter<W>) -> std::io::Result<()> {
        // Write JSON
        let mut out = JsonLinesWriter::new(writer);
        out.write(&self.header)?;
        out.write_all(self.commands.iter())?;

        // Write resources
        let mut out = out.into_inner();

        out.flush()?;

        Ok(())
    }
}
