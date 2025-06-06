use core::f32;
use std::collections::BinaryHeap;

use image::ImageBuffer;
use nalgebra::Vector2;
use rand::Rng;
use tensor_field::{DesignElement, GRID_SIZE, SeedPoint, TensorField};
use v4::{
    builtin_components::mesh_component::{MeshComponent, VertexDescriptor},
    scene,
};
use wgpu::vertex_attr_array;

mod tensor_field;

#[tokio::main]
async fn main() {
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
        0.0001,
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
                    priority: (-closest_degen).exp()
                        + (-(city_center - point).norm_squared()).exp(),
                    follow_major_eigenvector: true,
                })
            }
        })
        .collect();

    let mut trace = Vec::new();

    let mut count = 0;
    while let Some(SeedPoint {
        seed,
        follow_major_eigenvector,
        ..
    }) = seed_points.pop()
    {
        count += 1;
        let mut trace_res = tensor_field.trace(seed, 3.0, 50.0, follow_major_eigenvector, 200.0);
        // dbg!(trace_res.0.len());
        trace.append(&mut trace_res.0);
        seed_points.push(SeedPoint {
            seed: trace_res.1.unwrap_or_default(),
            priority: f32::MAX,
            follow_major_eigenvector: !follow_major_eigenvector,
        });

        // dbg!(count);

        if count == 200 {
            break;
        }
    }

    // dbg!(trace.len());
    /* let trace = tensor_field
    .trace(seed_points.pop().unwrap().seed, 3.0, 50.0, true)
    .0; */
    dbg!(trace.len());

    // dbg!(&trace[0..(40).min(trace.len())]);

    let mut image = ImageBuffer::new(GRID_SIZE as u32, GRID_SIZE as u32);
    for pos in &trace {
        if pos.x < GRID_SIZE as f32 && pos.y < GRID_SIZE as f32 {
            *image.get_pixel_mut(pos.x as u32, pos.y as u32) = image::Rgb([255_u8, 0, 0]);
        }
    }
    /* image.enumerate_pixels_mut().for_each(|(x, y, pixel)| {
        // let vector = Vector2::new(x as f32, y as f32);
        if trace.contains(&Vector2::new(x as f32, y as f32)) {
            *pixel = image::Rgb([255_u8, 0, 0]);
        }
        /* let dist = trace.iter().fold(f32::MAX, |acc, point| {
            let dist = (point - vector).norm_squared();
            if dist < acc { dist } else { acc }
        }); */

        /* if dist < 8.0 {
            *pixel = image::Rgb([255_u8, 0, 0])
        } else {
            *pixel = image::Rgb([0, 0, 0])
        } */
    }); */

    image.save("test.png").unwrap();

    /* let trace = vec![
        Vector2::new(0.0, 0.0),
        Vector2::new(0.0, GRID_SIZE as f32),
        Vector2::new(GRID_SIZE as f32, 0.0),
        Vector2::new(GRID_SIZE as f32, GRID_SIZE as f32),
    ]; */

    let mut engine = v4::V4::builder()
        .features(wgpu::Features::POLYGON_MODE_POINT)
        .window_settings(GRID_SIZE as u32, GRID_SIZE as u32, "Visualizer", None)
        .build()
        .await;

    scene! {
        scene: visualizer,
        _ = {
            material: {
                pipeline: {
                    vertex_shader_path: "./shaders/visualizer_vertex.wgsl",
                    fragment_shader_path: "./shaders/visualizer_fragment.wgsl",
                    vertex_layouts: [Vertex::vertex_layout()],
                    uses_camera: false,
                    geometry_details: {
                        topology: wgpu::PrimitiveTopology::PointList,
                        polygon_mode: wgpu::PolygonMode::Point,
                    }
                }
            },
            components: [
                MeshComponent(
                    vertices: vec![
                        trace.into_iter().map(|vec| Vertex {
                            pos: [
                                2.0 * vec.x / GRID_SIZE as f32 - 1.0,
                                -(2.0 * vec.y / GRID_SIZE as f32 - 1.0),
                                0.0
                            ]
                        }).collect()
                    ],
                    enabled_models: vec![(0, None)]
                )
            ]
        }
    }

    engine.attach_scene(visualizer);

    engine.main_loop().await;
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    pos: [f32; 3],
}

impl VertexDescriptor for Vertex {
    const ATTRIBUTES: &[wgpu::VertexAttribute] = &vertex_attr_array![0 => Float32x3];

    fn from_pos_normal_coords(pos: Vec<f32>, _normal: Vec<f32>, _tex_coords: Vec<f32>) -> Self {
        Self {
            pos: pos.try_into().unwrap(),
        }
    }
}
