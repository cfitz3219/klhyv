//! Turning images into something the interface can display.

use anyhow::{Context, Result};
use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use image::{imageops::FilterType, ImageFormat, RgbaImage};

/// Encode an image as a PNG data URL.
pub fn to_data_url(img: &RgbaImage) -> Result<String> {
    let mut bytes = Vec::new();
    img.write_to(&mut std::io::Cursor::new(&mut bytes), ImageFormat::Png)
        .context("encoding preview PNG")?;
    Ok(format!("data:image/png;base64,{}", STANDARD.encode(&bytes)))
}

/// Shrink so neither side exceeds `max`, leaving smaller images alone.
pub fn fit(img: &RgbaImage, max: u32) -> RgbaImage {
    let (w, h) = img.dimensions();
    if w <= max && h <= max {
        return img.clone();
    }
    let factor = (max as f32 / w.max(h) as f32).min(1.0);
    let (nw, nh) = (
        ((w as f32 * factor).round() as u32).max(1),
        ((h as f32 * factor).round() as u32).max(1),
    );
    image::imageops::resize(img, nw, nh, FilterType::Lanczos3)
}

/// A square region of `img` centred on normalised coordinates.
///
/// The comparison view works at full size rather than fitting the whole result
/// to the window: shrinking a 4x upscale back down to fit hides exactly the
/// detail the user is trying to judge.
pub fn region(img: &RgbaImage, cx: f32, cy: f32, size: u32) -> RgbaImage {
    let (w, h) = img.dimensions();
    let size = size.min(w).min(h).max(1);
    // Clamp so the window stays inside the image.
    let max_x = w.saturating_sub(size);
    let max_y = h.saturating_sub(size);
    let x = ((cx.clamp(0.0, 1.0) * w as f32) as u32).saturating_sub(size / 2).min(max_x);
    let y = ((cy.clamp(0.0, 1.0) * h as f32) as u32).saturating_sub(size / 2).min(max_y);
    image::imageops::crop_imm(img, x, y, size, size).to_image()
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgba;

    fn img(w: u32, h: u32) -> RgbaImage {
        RgbaImage::from_fn(w, h, |x, y| Rgba([(x % 256) as u8, (y % 256) as u8, 0, 255]))
    }

    #[test]
    fn fit_leaves_small_images_untouched() {
        let src = img(100, 80);
        assert_eq!(fit(&src, 400).dimensions(), (100, 80));
    }

    #[test]
    fn fit_preserves_aspect_ratio() {
        let out = fit(&img(1000, 500), 200);
        assert_eq!(out.dimensions(), (200, 100));
    }

    #[test]
    fn region_stays_inside_the_image() {
        let src = img(300, 200);
        // Ask for a region hanging off the bottom-right corner.
        let r = region(&src, 1.0, 1.0, 128);
        assert_eq!(r.dimensions(), (128, 128));
    }

    #[test]
    fn region_shrinks_to_fit_a_small_image() {
        let r = region(&img(60, 40), 0.5, 0.5, 256);
        assert_eq!(r.dimensions(), (40, 40), "region cannot exceed the image");
    }

    #[test]
    fn data_url_is_a_png() {
        let url = to_data_url(&img(4, 4)).unwrap();
        assert!(url.starts_with("data:image/png;base64,"));
        assert!(url.len() > 40);
    }
}
