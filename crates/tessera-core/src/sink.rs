//! Destinations for finished output rows.
//!
//! The blend accumulator costs 20 bytes per output pixel, so holding a whole
//! result in memory puts a hard ceiling on image size: a ×4 upscale of a
//! 20,000 × 20,000 map would need about 128 GiB. Handing finished rows straight
//! to a sink lets the pipeline keep only the rows still being worked on.
//!
//! Rows arrive strictly top to bottom and are never revisited, which is exactly
//! what an image encoder wants.

use std::fs::File;
use std::io::BufWriter;
use std::io::Write;
use std::path::Path;

use anyhow::{ensure, Context, Result};
use image::RgbaImage;

/// Somewhere finished RGBA8 rows can go.
pub trait RowSink {
    /// Accept one or more complete rows, top to bottom, 4 bytes per pixel.
    fn write_rows(&mut self, rgba: &[u8]) -> Result<()>;

    /// Flush and close. Must be called once, after the last row.
    fn finish(&mut self) -> Result<()>;
}

/// Collects rows into an image in memory.
///
/// Suitable for images that comfortably fit, and for intermediate passes.
pub struct MemorySink {
    width: u32,
    height: u32,
    buf: Vec<u8>,
}

impl MemorySink {
    pub fn new(width: u32, height: u32) -> Self {
        let bytes = width as usize * height as usize * 4;
        Self {
            width,
            height,
            buf: Vec::with_capacity(bytes),
        }
    }

    /// Consume the sink, producing the assembled image.
    pub fn into_image(self) -> Result<RgbaImage> {
        let expected = self.width as usize * self.height as usize * 4;
        ensure!(
            self.buf.len() == expected,
            "sink holds {} bytes but {}x{} needs {expected}",
            self.buf.len(),
            self.width,
            self.height
        );
        RgbaImage::from_raw(self.width, self.height, self.buf)
            .context("assembling image from sink buffer")
    }
}

impl RowSink for MemorySink {
    fn write_rows(&mut self, rgba: &[u8]) -> Result<()> {
        self.buf.extend_from_slice(rgba);
        Ok(())
    }

    fn finish(&mut self) -> Result<()> {
        Ok(())
    }
}

/// Streams rows straight into a PNG file.
///
/// Memory stays flat regardless of output size, so this is what makes very
/// large results possible.
pub struct PngSink {
    // Taken in `finish`, which consumes the writer.
    stream: Option<png::StreamWriter<'static, BufWriter<File>>>,
    path: String,
}

impl PngSink {
    pub fn create(path: &Path, width: u32, height: u32) -> Result<Self> {
        ensure!(width > 0 && height > 0, "cannot write a zero-sized image");
        let file = File::create(path)
            .with_context(|| format!("creating {}", path.display()))?;
        let mut encoder = png::Encoder::new(BufWriter::new(file), width, height);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        let writer = encoder
            .write_header()
            .with_context(|| format!("writing PNG header for {}", path.display()))?;
        let stream = writer
            .into_stream_writer()
            .context("opening PNG stream writer")?;
        Ok(Self {
            stream: Some(stream),
            path: path.display().to_string(),
        })
    }
}

impl RowSink for PngSink {
    fn write_rows(&mut self, rgba: &[u8]) -> Result<()> {
        let stream = self
            .stream
            .as_mut()
            .context("PNG sink already finished")?;
        stream
            .write_all(rgba)
            .with_context(|| format!("writing pixels to {}", self.path))
    }

    fn finish(&mut self) -> Result<()> {
        // The PNG stream must be finished explicitly; dropping it would leave a
        // truncated file behind.
        if let Some(stream) = self.stream.take() {
            stream
                .finish()
                .with_context(|| format!("finalising {}", self.path))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_sink_assembles_rows_in_order() {
        let mut sink = MemorySink::new(2, 2);
        sink.write_rows(&[1, 2, 3, 4, 5, 6, 7, 8]).unwrap();
        sink.write_rows(&[9, 10, 11, 12, 13, 14, 15, 16]).unwrap();
        sink.finish().unwrap();
        let img = sink.into_image().unwrap();
        assert_eq!(img.get_pixel(0, 0).0, [1, 2, 3, 4]);
        assert_eq!(img.get_pixel(1, 1).0, [13, 14, 15, 16]);
    }

    #[test]
    fn memory_sink_rejects_a_short_image() {
        let mut sink = MemorySink::new(4, 4);
        sink.write_rows(&[0; 16]).unwrap();
        assert!(sink.into_image().is_err(), "must not pad a partial image");
    }

    #[test]
    fn png_sink_round_trips_through_a_file() {
        let dir = std::env::temp_dir().join("tessera-png-sink-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("out.png");

        let mut sink = PngSink::create(&path, 3, 2).unwrap();
        sink.write_rows(&[10; 3 * 4]).unwrap();
        sink.write_rows(&[200; 3 * 4]).unwrap();
        sink.finish().unwrap();

        let img = image::open(&path).unwrap().to_rgba8();
        assert_eq!(img.dimensions(), (3, 2));
        assert_eq!(img.get_pixel(0, 0).0, [10, 10, 10, 10]);
        assert_eq!(img.get_pixel(2, 1).0, [200, 200, 200, 200]);
        let _ = std::fs::remove_file(&path);
    }
}
