use image::{EncodableLayout, ImageBuffer};
use rand::Rng;
use rayon::prelude::*;
use street_graph::path_to_graph;
use street_plan::{HermiteCurve, SeedPoint, merge_road_endings, resample_curve, trace_street_plan};
use tensor_field::{DesignElement, GRID_SIZE, Point, TensorField};
use v4::{
    builtin_components::{
        camera_component::CameraComponent,
        mesh_component::{MeshComponent, VertexDescriptor},
        transform_component::TransformComponent,
    },
    engine_support::texture_support::Texture,
    scene,
};
use wgpu::vertex_attr_array;

use crate::{
    building_generation::footprint_to_building, tensor_field::EvalEigenvectors,
    triangulation::triangulate_faces, water_mask::mask_to_elements,
};

mod building_generation;
mod intersections;
mod street_graph;
mod street_plan;
mod tensor_field;
mod triangulation;
mod water_mask;

#[tokio::main]
async fn main() {
    let start_time = std::time::Instant::now();
    let grid_element = DesignElement::Grid {
        center: Point::new(100.0, 100.0),
        // theta: -std::f32::consts::FRAC_PI_3 * 2.0,
        theta: std::f32::consts::FRAC_PI_2,
        length: 100.0,
    };

    let grid_element_2 = DesignElement::Grid {
        center: Point::new(300.0, 400.0),
        theta: 0.1,
        length: 200.0,
    };

    let radial_element = DesignElement::Radial {
        center: Point::new(200.0, 200.0),
    };

    let grid_element_3 = DesignElement::Grid {
        center: Point::new(0.0, 400.0),
        theta: 0.7,
        length: 10.0,
    };

    let city_center = *radial_element.center().as_ref().unwrap();

    let (water_edge, edge_elements) = mask_to_elements("water_mask.png");
    println!("Water edge length: {}", water_edge.len());

    let tensor_field = TensorField::new(
        edge_elements,
        /* vec![
            grid_element, /* , radial_element, grid_element_2, grid_element_3 */
        ], */
        /* .into_iter()
        .chain(edge_elements)
        .collect(), */
        0.0004,
    );

    let tensor_field_grid_size = 128_usize;
    let mut grid = (-(tensor_field_grid_size as i32)..=tensor_field_grid_size as i32)
        .map(|y| {
            (-(tensor_field_grid_size as i32)..=tensor_field_grid_size as i32)
                .map(|x| Point::new(x as f32, y as f32) / (tensor_field_grid_size as f32))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    for _ in 0..5 {
        for row in &mut grid {
            for point in row {
                let field_value = tensor_field.evaluate_smoothed_field_at_point(*point);
                if field_value.norm() > 0.0001 {
                    println!("Valid at: {point:?}");
                }
                *point += field_value.eigenvectors().major.normalize() / 10.0;
            }
        }
    }

    let mut white_noise =
        ImageBuffer::new(GRID_SIZE, GRID_SIZE);
    let mut rng = rand::rng();
    for pix in white_noise.pixels_mut() {
        let val =
            (rng.sample::<f32, _>(rand_distr::Normal::new(1.0, 1.0).unwrap()) * 255.0).abs() as u8;
        *pix = image::Rgba([val, val, val, 255]);
    }

    let tiles: Vec<Vertex> = (0..(2 * tensor_field_grid_size))
        .flat_map(|y| {
            (0..(2 * tensor_field_grid_size))
                .flat_map(|x| {
                    let x_coord = x as f32 / (2.0 * tensor_field_grid_size as f32);
                    let y_coord = y as f32 / (2.0 * tensor_field_grid_size as f32);
                    let offset = 1.0 / tensor_field_grid_size as f32;
                    [
                        Vertex {
                            pos: [grid[y][x].x, grid[y][x].y, 0.1],
                            normal: [0.0; 3],
                            tex_coords: [x_coord, y_coord],
                        },
                        Vertex {
                            pos: [grid[y][x + 1].x, grid[y][x + 1].y, 0.1],
                            normal: [0.0; 3],
                            tex_coords: [x_coord + offset, y_coord],
                        },
                        Vertex {
                            pos: [grid[y + 1][x].x, grid[y + 1][x].y, 0.1],
                            normal: [0.0; 3],
                            tex_coords: [x_coord, y_coord + offset],
                        },
                        Vertex {
                            pos: [grid[y + 1][x + 1].x, grid[y + 1][x + 1].y, 0.1],
                            normal: [0.0; 3],
                            tex_coords: [x_coord + offset, y_coord + offset],
                        },
                    ]
                })
                .collect::<Vec<_>>()
        })
        .collect();

    let tile_indices: Vec<u32> = (0..=tensor_field_grid_size as u32 * tensor_field_grid_size as u32 * 4)
        .flat_map(|i| vec![4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 1, 4 * i + 3, 4 * i + 2])
        .collect();

    let (major_network_major_curves_unconnected, major_network_minor_curves_unconnected) =
        trace_street_plan(
            &tensor_field,
            "water_mask.png",
            street_plan::TraceSeeds::Random(30),
            city_center,
            30.0,
            5,
            Vec::new(),
            Vec::new(),
        );

    let major_network_major_curves_len = major_network_major_curves_unconnected.len();

    let major_network_curves_unconnected: Vec<HermiteCurve> =
        major_network_major_curves_unconnected
            .into_iter()
            .chain(major_network_minor_curves_unconnected)
            .collect();
    let major_network_merge_distance = 5.0;
    let major_network_curves = merge_road_endings(
        &major_network_curves_unconnected,
        major_network_merge_distance,
    );

    let minor_network_seed_points: Vec<SeedPoint> = major_network_curves
        .iter()
        .flat_map(|curve| {
            (0..curve.len() - 1)
                .map(|i| SeedPoint {
                    seed: (curve[i].position + curve[i + 1].position) / 2.0,
                    priority: 0.0,
                    follow_major_eigenvectors: i > major_network_major_curves_len,
                })
                .collect::<Vec<_>>()
        })
        .collect();

    let major_network: Vec<Vec<Point>> = major_network_curves
        .par_iter()
        .map(|curve| resample_curve(curve, 20))
        .collect();

    let (minor_network_major_curves_unconnected, minor_network_minor_curves_unconnected) =
        trace_street_plan(
            &tensor_field,
            "water_mask.png",
            street_plan::TraceSeeds::Specific(minor_network_seed_points),
            city_center,
            5.0,
            3,
            major_network_curves[..major_network_major_curves_len].to_vec(),
            major_network_curves[major_network_major_curves_len..].to_vec(),
        );

    let minor_network_curves_unconnected: Vec<HermiteCurve> =
        minor_network_major_curves_unconnected
            .into_iter()
            .chain(minor_network_minor_curves_unconnected)
            .collect();
    let minor_network_merge_distance = 3.0;

    let minor_network_curves = merge_road_endings(
        &minor_network_curves_unconnected,
        minor_network_merge_distance,
    );

    let minor_network: Vec<Vec<Point>> = minor_network_curves
        .par_iter()
        .map(|curve| resample_curve(curve, 20))
        .collect();

    println!("{}", start_time.elapsed().as_millis() as f32 / 1000.0);

    /* let all_curves: Vec<HermiteCurve> = minor_network_curves
        .into_iter()
        .chain(major_network_curves)
        .collect();

    let footprints = path_to_graph(&all_curves, 15.0, 20.0);

    let faces: Vec<Vec<Vertex>> = footprints
        .iter()
        .flat_map(|footprint| {
            footprint_to_building(
                footprint,
                ((55.0
                    - (footprint.iter().copied().sum::<Point>() / footprint.len() as f32)
                        .metric_distance(&city_center)
                        .min(50.0))
                    / 5.0)
                    .floor()
                    * 5.0,
            )
        })
        .collect();

    let triangulated_faces = triangulate_faces(&faces);
    let verts_for_triangulation: Vec<Vertex> = faces.into_iter().flatten().collect(); */

    let mut engine = v4::V4::builder()
        .features(wgpu::Features::POLYGON_MODE_LINE | wgpu::Features::POLYGON_MODE_POINT)
        .window_settings(
            GRID_SIZE as u32 * 2,
            GRID_SIZE as u32 * 2,
            "Visualizer",
            None,
        )
        .hide_cursor(true)
        .antialiasing_enabled(true)
        .build()
        .await;

    let rendering_manager = engine.rendering_manager();
    let device = rendering_manager.device();
    let queue = rendering_manager.queue();

    let sample_factor = 14;
    let vector_opacity = 0.2;

    scene! {
        scene: visualizer,
        active_camera: "cam",
        "eigenvectors" = {
            material: {
                pipeline: {
                    vertex_shader_path: "./shaders/visualizer_vertex.wgsl",
                    fragment_shader_path: "./shaders/visualizer_fragment.wgsl",
                    vertex_layouts: [LineVertex::vertex_layout(), TransformComponent::vertex_layout::<2>()],
                    uses_camera: false,
                    geometry_details: {
                        topology: wgpu::PrimitiveTopology::LineList,
                        polygon_mode: wgpu::PolygonMode::Line,
                    },
                },
            },
            components: [
                MeshComponent(
                    vertices: vec![
                        (0..GRID_SIZE / sample_factor).flat_map(|x| (0..GRID_SIZE / sample_factor).flat_map(|y| {
                            let point = Point::new(x as f32 * sample_factor as f32, y as f32 * sample_factor as f32);
                            let tensor = tensor_field.evaluate_smoothed_field_at_point(point);
                            let eigenvectors = tensor.eigenvectors();
                            let maj = eigenvectors.major.normalize() * (sample_factor - 1) as f32;
                            let min = eigenvectors.minor.normalize() * (sample_factor - 1) as f32;
                            let maj_point = normalize_vector(point + maj);
                            let min_point = normalize_vector(point + min);
                            let norm_point = normalize_vector(point);
                            [
                                LineVertex {pos: [norm_point.x, norm_point.y, 0.0], col: [1.0, 0.0, 0.0, vector_opacity]}, LineVertex {pos: [maj_point.x, maj_point.y, 0.0], col: [1.0, 0.0, 0.0, vector_opacity]},
                                LineVertex {pos: [norm_point.x, norm_point.y, 0.0], col: [0.0, 1.0, 0.0, vector_opacity]}, LineVertex {pos: [min_point.x, min_point.y, 0.0], col: [0.0, 1.0, 0.0, vector_opacity]}
                            ]
                        }).collect::<Vec<_>>()).collect(),
                        water_edge.iter().flat_map(|&(point, dir)| {
                            let start = normalize_vector(point);
                            let end = normalize_vector(point + dir.normalize() * 15.0);
                            [
                                LineVertex {pos: [start.x, start.y, 0.0], col: [1.0, 1.0, 1.0, vector_opacity]},
                                LineVertex {pos: [end.x, end.y, 0.0], col: [1.0, 1.0, 1.0, vector_opacity]}
                            ]
                        }).collect()
                    ],
                    enabled_models: vec![(0, None), (1, None)]
                ),
                TransformComponent(position: nalgebra::Vector3::zeros())
            ]
        },
        /* "tensorfield" = {
            material: {
                pipeline: {
                    vertex_shader_path: "./shaders/tensorfield_vertex.wgsl",
                    fragment_shader_path: "./shaders/tensorfield_fragment.wgsl",
                    vertex_layouts: [Vertex::vertex_layout()],
                    uses_camera: false,
                },
                attachments: [
                    Texture(
                        texture: Texture::from_bytes(
                            white_noise.as_bytes(),
                            (GRID_SIZE, GRID_SIZE),
                            device,
                            queue,
                            wgpu::TextureFormat::Rgba8Unorm,
                            None,
                            true,
                            wgpu::TextureUsages::empty()
                        ),
                        visibility: wgpu::ShaderStages::FRAGMENT
                    ),
                ],
            },
            components: [
                MeshComponent(
                    vertices: vec![tiles],
                    indices: vec![tile_indices],
                    enabled_models: vec![(0, None)]
                )
            ]
        }, */
        "major_network" = {
            material: {
                pipeline: {
                    vertex_shader_path: "./shaders/visualizer_vertex.wgsl",
                    fragment_shader_path: "./shaders/visualizer_fragment.wgsl",
                    vertex_layouts: [LineVertex::vertex_layout(), TransformComponent::vertex_layout::<2>()],
                    uses_camera: true,
                    geometry_details: {
                        topology: wgpu::PrimitiveTopology::LineList,
                        polygon_mode: wgpu::PolygonMode::Line,
                    },
                    ident: "network_pipeline",
                }
            },
            components: [
                MeshComponent(
                    vertices:
                        vec![major_network.iter().flat_map(|arr| {
                            (0..arr.len() - 1).flat_map(|i| {
                                let current_pos = normalize_vector(arr[i]);
                                let next_pos = normalize_vector(arr[i + 1]);
                                [
                                    LineVertex {
                                        pos: [ current_pos.x, current_pos.y, 0.0,],
                                        col: [0.0, 0.0, 1.0, 1.0]
                                    },
                                    LineVertex {
                                        pos: [ next_pos.x, next_pos.y, 0.0, ],
                                        col: [0.0, 0.0, 1.0, 1.0]
                                    }
                                ]
                            })
                        }).collect()],
                    enabled_models: vec![(0, None)]
                ),
                TransformComponent(position: nalgebra::Vector3::new(0.0, 0.0, 0.0))
            ]
        },
        "minor_network" = {
            material: {
                pipeline: ident("network_pipeline"),
            },
            components: [
                MeshComponent(
                    vertices:
                        vec![minor_network.iter().flat_map(|arr| {
                            (0..arr.len() - 1).flat_map(|i| {
                                let current_pos = normalize_vector(arr[i]);
                                let next_pos = normalize_vector(arr[i + 1]);
                                [
                                    LineVertex {
                                        pos: [ current_pos.x, current_pos.y, 0.0,],
                                        col: [1.0, 0.0, 0.0, 1.0]
                                    },
                                    LineVertex {
                                        pos: [ next_pos.x, next_pos.y, 0.0, ],
                                        col: [1.0, 0.0, 0.0, 1.0]
                                    }
                                ]
                            })
                        }).collect()],
                    enabled_models: vec![(0, None)]
                ),
                TransformComponent(position: nalgebra::Vector3::new(0.0, 0.0, 0.0))
            ]
        },
        /* "plots" = {
            material: {
                pipeline: {
                    vertex_shader_path: "./shaders/buildings_vertex.wgsl",
                    fragment_shader_path: "./shaders/buildings_fragment.wgsl",
                    vertex_layouts: [Vertex::vertex_layout(), TransformComponent::vertex_layout::<3>()],
                    uses_camera: true,
                }
            },
            components: [
                MeshComponent(
                    vertices: vec![verts_for_triangulation],
                    indices: vec![triangulated_faces],
                    enabled_models: vec![(0, None)]
                ),
                TransformComponent(position: nalgebra::Vector3::new(0.0, 0.0, 0.0))
            ]
        }, */
        "cam_ent" = {
            components: [
                CameraComponent(field_of_view: 80.0, aspect_ratio: 1.0, near_plane: 0.1, far_plane: GRID_SIZE as f32, sensitivity: 0.002, movement_speed: 0.05, ident: "cam"),
                TransformComponent(position: nalgebra::Vector3::new(50.0, 30.0, 50.0))
            ]
        }
    }

    engine.attach_scene(visualizer);

    engine.main_loop().await;
}

fn normalize_vector(vec: Point) -> Point {
    Point::new(
        2.0 * vec.x / GRID_SIZE as f32 - 1.0,
        2.0 * vec.y / GRID_SIZE as f32 - 1.0,
    )
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct LineVertex {
    pos: [f32; 3],
    col: [f32; 4],
}

impl VertexDescriptor for LineVertex {
    const ATTRIBUTES: &[wgpu::VertexAttribute] =
        &vertex_attr_array![0 => Float32x3, 1 => Float32x4];

    fn from_pos_normal_coords(pos: Vec<f32>, _normal: Vec<f32>, _tex_coords: Vec<f32>) -> Self {
        Self {
            pos: pos.try_into().unwrap(),
            col: [1.0; 4],
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    pos: [f32; 3],
    normal: [f32; 3],
    tex_coords: [f32; 2],
}

impl VertexDescriptor for Vertex {
    const ATTRIBUTES: &[wgpu::VertexAttribute] =
        &vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x2];

    fn from_pos_normal_coords(pos: Vec<f32>, normal: Vec<f32>, tex_coords: Vec<f32>) -> Self {
        Self {
            pos: pos.try_into().unwrap(),
            normal: normal.try_into().unwrap(),
            tex_coords: tex_coords.try_into().unwrap(),
        }
    }
}
