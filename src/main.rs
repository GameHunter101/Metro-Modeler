use core::f32;
use std::collections::BinaryHeap;

use image::ImageBuffer;
use nalgebra::Vector2;
use rand::Rng;
use tensor_field::{DesignElement, SeedPoint, TensorField, GRID_SIZE};

mod tensor_field;

fn main() {
    let grid_element = DesignElement::Grid {
        center: Vector2::new(100.0, 100.0),
        theta: -std::f32::consts::FRAC_PI_3 * 2.0,
        // theta: 0.0,
        length: 500.0,
    };

    let grid_element_2 = DesignElement::Grid {
        center: Vector2::new(300.0, 400.0),
        theta: 0.1,
        length: 200.0,
    };

    let radial_element = DesignElement::Radial {
        center: Vector2::new(200.0, 200.0),
    };

    let grid_element_3 = DesignElement::Grid {
        center: Vector2::new(0.0, 400.0),
        theta: 0.7,
        length: 10.0,
    };

    let tensor_field = TensorField::new(
        vec![grid_element, radial_element, grid_element_2, grid_element_3],
        0.005,
    );

    let degenerate_points: Vec<Vector2<f32>> = (0..GRID_SIZE)
        .flat_map(|x| {
            (0..GRID_SIZE)
                .flat_map(|y| {
                    if tensor_field
                        .evaluate_field_at_point(Vector2::new(x as f32, y as f32))
                        .norm_squared()
                        <= 0.0001
                    {
                        Some(Vector2::new(x as f32, y as f32))
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect();

    let city_center = Vector2::new(150.0_f32, 240.0);

    let mut rng = rand::rng();
    let mut seed_points: BinaryHeap<SeedPoint> = (0..100)
        .flat_map(|_| {
            let point = Vector2::new(
                rng.random_range(0..GRID_SIZE) as f32,
                rng.random_range(0..GRID_SIZE) as f32,
            );

            if tensor_field.evaluate_field_at_point(point).norm_squared() <= 0.00001 {
                return None;
            }

            let closest_degen = degenerate_points
                .iter()
                .map(|p| (p - point).norm_squared())
                .fold(f32::MAX, |acc, e| if e < acc { e } else { acc });

            if closest_degen <= 0.00001 {
                None
            } else {
                Some(SeedPoint {
                    seed: point,
                    priority: (-closest_degen).exp() + (-(city_center - point).norm_squared()).exp(),
                })
            }
        })
        .collect();

    dbg!(seed_points.len());

    let mut trace = Vec::new();

    let mut count = 0;
    while let Some(SeedPoint { seed, .. }) = seed_points.pop() {
        count += 1;
        trace.append(&mut tensor_field.trace(seed, 3.0));
    }

    dbg!(count);


    let mut image = ImageBuffer::new(GRID_SIZE as u32, GRID_SIZE as u32);
    image.enumerate_pixels_mut().for_each(|(x, y, pixel)| {
        let vector = Vector2::new(x as f32, y as f32);
        let dist = trace.iter().fold(f32::MAX, |acc, point| {
            let dist = (point - vector).norm_squared();
            if dist < acc { dist } else { acc }
        });

        if dist < 8.0 {
            *pixel = image::Rgb([255_u8, 0, 0])
        } else {
            *pixel = image::Rgb([0, 0, 0])
        }
    });

    /* image.enumerate_pixels_mut().for_each(|(x, y, pixel)| {
        let mut tensor = tensor_field.evaluate_field_at_point(Vector2::new(x as f32, y as f32));
        let mut neighbor_count = 1;
        if x != 0 {
            neighbor_count += 1;
            tensor += tensor_field.evaluate_field_at_point(Vector2::new(x as f32 - 1.0, y as f32));
        }
        if x != GRID_SIZE as u32 - 1 {
            neighbor_count += 1;
            tensor += tensor_field.evaluate_field_at_point(Vector2::new(x as f32 + 1.0, y as f32));
        }
        if y != 0 {
            neighbor_count += 1;
            tensor += tensor_field.evaluate_field_at_point(Vector2::new(x as f32, y as f32 - 1.0));
        }
        if x != GRID_SIZE as u32 - 1 {
            neighbor_count += 1;
            tensor += tensor_field.evaluate_field_at_point(Vector2::new(x as f32, y as f32 + 1.0));
        }

        tensor /= neighbor_count as f32;

        let eigenvalues = tensor.eigenvalues().unwrap();
        let eigenvectors = tensor.eigenvectors();
        let major_eigenvector = eigenvectors.major;
        let angle = major_eigenvector.normalize().dot(&Vector2::x()).acos();
        let c = (eigenvalues.x - eigenvalues.y).abs() / 4.0;
        *pixel = image::Rgb([
            (angle / std::f32::consts::PI * 255.0) as u8,
            (c * 255.0) as u8,
            0,
        ]);
    }); */

    image.save("test.png").unwrap();
}
