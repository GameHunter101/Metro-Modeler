use std::collections::{HashMap, HashSet};

use cool_utils::data_structures::dcel::DCEL;
use image::{GenericImage, GenericImageView, ImageBuffer, ImageReader, Pixel};
use nalgebra::Vector2;
use rayon::iter::{FromParallelIterator, ParallelBridge, ParallelIterator};

use crate::{
    street_graph::{OrderedPoint, order_point},
    tensor_field::{DesignElement, Point},
};

fn mask_faces(path: &str) -> (Vec<Point>, DCEL) {
    let img = ImageReader::open(path).unwrap().decode().unwrap();

    let mut grad_tex = ImageBuffer::new(img.width(), img.height());

    let all_points: Vec<Point> = img
        .pixels()
        .flat_map(|(x, y, pix)| {
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

            if (x_diff != 0 || y_diff != 0) && pix_luma == 255 {
                grad_tex.put_pixel(
                    x,
                    y,
                    image::Rgba([
                        (y_diff / y_samples) as u8,
                        (x_diff / x_samples) as u8,
                        0,
                        255,
                    ]),
                );
                Some(Point::new(x as f32, y as f32))
            } else {
                None
            }
        })
        .collect();

    let inverse_map: HashMap<(i32, i32), usize> = all_points
        .iter()
        .enumerate()
        .map(|(i, point)| ((point.x as i32, point.y as i32), i))
        .collect();

    let adjacency_list: HashMap<usize, HashSet<usize>> =
        HashMap::from_iter(all_points.iter().enumerate().map(|(i, point)| {
            let mut neighbors = HashSet::new();
            for i in -1..=1 {
                for j in -1..=1 {
                    if i == 0 && j == 0 {
                        continue;
                    }
                    if all_points.contains(&Point::new(point.x + j as f32, point.y + i as f32)) {
                        neighbors.insert(inverse_map[&(point.x as i32 + j, point.y as i32 + i)]);
                    }
                }
            }
            (i, neighbors)
        }));

    let dcel = DCEL::new(&all_points, &adjacency_list);

    grad_tex.save("./grad_res.png").unwrap();
    (all_points, dcel)
}

pub fn mask_to_elements(mask_path: &str) -> (Vec<(Point, Point)>, Vec<DesignElement>) {
    let skip_factor = 5;
    let (points, dcel) = mask_faces(mask_path);
    dcel.faces()
        .iter()
        .flat_map(|face| {
            face.iter().enumerate().flat_map(|(i, &point_idx)| {
                if point_idx % skip_factor == 0 {
                    let dir = points[face[(i + skip_factor) % points.len()]] - points[point_idx];
                    let theta = dir.y.atan2(dir.x);// + std::f32::consts::FRAC_PI_2;
                    if points[point_idx] == Point::new(66.0, 463.0) {
                        println!("Angle: {theta}");
                    }
                    Some((
                        (points[point_idx], dir),
                        DesignElement::Grid {
                            center: points[point_idx],
                            theta,
                            length: skip_factor as f32,
                        },
                    ))
                } else {
                    None
                }
            })
        })
        .unzip()
}
