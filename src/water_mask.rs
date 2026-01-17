use std::collections::HashMap;

use image::{GenericImageView, ImageReader, Pixel};
use nalgebra::Vector2;
use rayon::iter::{FromParallelIterator, ParallelBridge, ParallelIterator};

use crate::tensor_field::{DesignElement, Point};

fn mask_edge_pixel_directions(path: &str) -> HashMap<(i32, i32), Vector2<f32>> {
    let img = ImageReader::open(path).unwrap().decode().unwrap();

    HashMap::from_par_iter(img.pixels().par_bridge().flat_map(|(x, y, pix)| {
        let mut x_diff: i32 = 0;
        let mut x_samples = 0;
        let mut y_diff: i32 = 0;
        let mut y_samples = 0;

        let pix_luma = pix.to_luma().0[0] as i32;

        if x < img.width() - 1 {
            x_diff += img.get_pixel(x + 1, y).to_luma().0[0] as i32 - pix_luma;
            x_samples += 1;
        }
        if x > 0 {
            x_diff += pix_luma - img.get_pixel(x - 1, y).to_luma().0[0] as i32;
            x_samples += 1;
        }

        if y < img.height() - 1 {
            y_diff += img.get_pixel(x, y + 1).to_luma().0[0] as i32 - pix_luma;
            y_samples += 1;
        }
        if y > 0 {
            y_diff += pix_luma - img.get_pixel(x, y - 1).to_luma().0[0] as i32;
            y_samples += 1;
        }

        if x_diff != 0 || y_diff != 0 {
            Some((
                (x as i32, y as i32),
                Vector2::new((x_diff / x_samples) as f32, (y_diff / y_samples) as f32).normalize(),
            ))
        } else {
            None
        }
    }))
}

pub fn mask_to_elements(mask_path: &str) -> Vec<DesignElement> {
    let skip_factor = 5;
    let edge_pixels = mask_edge_pixel_directions(mask_path);
    edge_pixels
        .iter()
        .enumerate()
        .flat_map(|(i, ((x, y), dir))| {
            let (x, y) = (*x as f32, *y as f32);
            if i % skip_factor == 0 {
                Some(DesignElement::Grid {
                    center: Point::new(x, y),
                    theta: dir.y.atan2(dir.x) + std::f32::consts::FRAC_PI_2,
                    length: skip_factor as f32,
                })
            } else {
                None
            }
        })
        .collect()
}
