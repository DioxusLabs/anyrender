use serde::Serialize;

pub(crate) fn to_json_depth_limited<T: Serialize>(
    value: &T,
    max_depth: usize,
) -> Result<String, serde_json::Error> {
    let mut buf = Vec::new();
    let mut serializer =
        serde_json::Serializer::with_formatter(&mut buf, DepthLimitedFormatter::new(max_depth));
    value.serialize(&mut serializer)?;
    Ok(String::from_utf8(buf).unwrap())
}

/// A JSON formatter that pretty-prints up to a nesting depth before
/// falling back to inline formatting (no whitespace).
struct DepthLimitedFormatter {
    max_depth: usize,
    current_depth: usize,
    /// Stack tracking per-level state: (is_pretty, has_value).
    stack: Vec<(bool, bool)>,
}

impl DepthLimitedFormatter {
    fn new(max_depth: usize) -> Self {
        Self {
            max_depth,
            current_depth: 0,
            stack: Vec::new(),
        }
    }

    fn indent<W: ?Sized + std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        for _ in 0..self.current_depth {
            writer.write_all(b"  ")?;
        }
        Ok(())
    }

    fn push(&mut self) {
        self.current_depth += 1;
        let pretty = self.current_depth <= self.max_depth;
        self.stack.push((pretty, false));
    }

    fn pop(&mut self) -> (bool, bool) {
        let result = self.stack.pop().unwrap_or((false, false));
        self.current_depth -= 1;
        result
    }

    fn set_has_value(&mut self) {
        if let Some(entry) = self.stack.last_mut() {
            entry.1 = true;
        }
    }

    fn current_is_pretty(&self) -> bool {
        self.stack.last().is_some_and(|&(pretty, _)| pretty)
    }
}

impl serde_json::ser::Formatter for DepthLimitedFormatter {
    fn begin_array<W: ?Sized + std::io::Write>(&mut self, writer: &mut W) -> std::io::Result<()> {
        self.push();
        writer.write_all(b"[")
    }

    fn end_array<W: ?Sized + std::io::Write>(&mut self, writer: &mut W) -> std::io::Result<()> {
        let (pretty, had_value) = self.pop();
        if pretty && had_value {
            writer.write_all(b"\n")?;
            self.indent(writer)?;
        }
        writer.write_all(b"]")
    }

    fn begin_array_value<W: ?Sized + std::io::Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> std::io::Result<()> {
        if self.current_is_pretty() {
            if !first {
                writer.write_all(b",")?;
            }
            writer.write_all(b"\n")?;
            self.indent(writer)?;
        } else if !first {
            writer.write_all(b", ")?;
        }
        self.set_has_value();
        Ok(())
    }

    fn begin_object<W: ?Sized + std::io::Write>(&mut self, writer: &mut W) -> std::io::Result<()> {
        self.push();
        writer.write_all(b"{")
    }

    fn end_object<W: ?Sized + std::io::Write>(&mut self, writer: &mut W) -> std::io::Result<()> {
        let (pretty, had_value) = self.pop();
        if pretty && had_value {
            writer.write_all(b"\n")?;
            self.indent(writer)?;
        }
        writer.write_all(b"}")
    }

    fn begin_object_key<W: ?Sized + std::io::Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> std::io::Result<()> {
        if self.current_is_pretty() {
            if !first {
                writer.write_all(b",")?;
            }
            writer.write_all(b"\n")?;
            self.indent(writer)?;
        } else if !first {
            writer.write_all(b", ")?;
        }
        self.set_has_value();
        Ok(())
    }

    fn begin_object_value<W: ?Sized + std::io::Write>(
        &mut self,
        writer: &mut W,
    ) -> std::io::Result<()> {
        writer.write_all(b": ")
    }
}
