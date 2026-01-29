use std::{io::Write, io::BufWriter};

use peniko::{FontData, ImageData};
use serde_jsonlines::JsonLinesWriter;
use serde::{Serialize, Deserialize};

use crate::recording::RenderCommand;

#[derive(Copy, Clone, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ResourceId(usize);

#[derive(Serialize, Deserialize)]
pub struct ResourcePosition {
    offset: usize,
    length: usize,
}

pub type SerializableRenderCommand = RenderCommand<ResourceId, ResourceId>;

#[derive(Serialize, Deserialize)]
pub struct SerializableSceneHeader {
    /// The number of commands in the file
    command_count: usize,
    /// When the scene was serialized
    created_at: u64,
    /// The number of commands in the file
    images: Vec<ResourcePosition>,
    fonts: Vec<ResourcePosition>,
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