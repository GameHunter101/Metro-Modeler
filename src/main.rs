use nalgebra::Vector2;
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
    scene,
};
use wgpu::vertex_attr_array;

use crate::triangulation::triangulate_faces;

mod building_generation;
mod intersections;
mod street_graph;
mod street_plan;
mod tensor_field;
mod triangulation;

#[tokio::main]
async fn main() {
    let start_time = std::time::Instant::now();
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

    let city_center = *radial_element.center().as_ref().unwrap();

    let tensor_field = TensorField::new(
        vec![grid_element, radial_element, grid_element_2, grid_element_3],
        0.0004,
    );

    let (major_network_major_curves_unconnected, major_network_minor_curves_unconnected) =
        trace_street_plan(
            &tensor_field,
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

    let all_curves: Vec<HermiteCurve> = minor_network_curves
        .into_iter()
        .chain(major_network_curves)
        .collect();

    let faces = path_to_graph(&all_curves, 15.0, 20.0);

    let triangulated_faces = triangulate_faces(&faces);
    let verts_for_triangulation: Vec<Vertex> = faces
        .into_iter()
        .flatten()
        .map(|point| Vertex {
            pos: [point.x, 0.0, point.y],
            col: [0.0, 1.0, 0.0, 1.0],
        })
        .collect();

    let mut engine = v4::V4::builder()
        .features(wgpu::Features::POLYGON_MODE_LINE | wgpu::Features::POLYGON_MODE_POINT)
        .window_settings(
            GRID_SIZE as u32 * 2,
            GRID_SIZE as u32 * 2,
            "Visualizer",
            None,
        )
        .hide_cursor(true)
        .build()
        .await;


    scene! {
        scene: visualizer,
        active_camera: "cam",
        "major_network" = {
            material: {
                pipeline: {
                    vertex_shader_path: "./shaders/visualizer_vertex.wgsl",
                    fragment_shader_path: "./shaders/visualizer_fragment.wgsl",
                    vertex_layouts: [Vertex::vertex_layout(), TransformComponent::vertex_layout::<2>()],
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
                                [
                                    Vertex {
                                        pos: [ arr[i].x, 0.0, arr[i].y, ],
                                        col: [0.0, 0.0, 1.0, 1.0]
                                    },
                                    Vertex {
                                        pos: [ arr[i + 1].x, 0.0, arr[i + 1].y, ],
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
                                [
                                    Vertex {
                                        pos: [ arr[i].x, 0.0, arr[i].y, ],
                                        col: [1.0, 0.0, 0.0, 1.0]
                                    },
                                    Vertex {
                                        pos: [ arr[i + 1].x, 0.0, arr[i + 1].y, ],
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
        "plots" = {
            material: {
                pipeline: {
                    vertex_shader_path: "./shaders/visualizer_vertex.wgsl",
                    fragment_shader_path: "./shaders/visualizer_fragment.wgsl",
                    vertex_layouts: [Vertex::vertex_layout(), TransformComponent::vertex_layout::<2>()],
                    uses_camera: true,
                    ident: "network_pipeline",
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
        },
        "cam_ent" = {
            components: [
                CameraComponent(field_of_view: 80.0, aspect_ratio: 1.0, near_plane: 0.1, far_plane: GRID_SIZE as f32, sensitivity: 0.002, movement_speed: 0.05, ident: "cam"),
                TransformComponent(position: nalgebra::Vector3::new(0.0, 0.0, 0.0))
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
    col: [f32; 4],
}

impl VertexDescriptor for Vertex {
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
struct TexVertex {
    pos: [f32; 3],
    tex_coords: [f32; 2],
}

impl VertexDescriptor for TexVertex {
    const ATTRIBUTES: &[wgpu::VertexAttribute] =
        &vertex_attr_array![0 => Float32x3, 1 => Float32x2];

    fn from_pos_normal_coords(pos: Vec<f32>, _normal: Vec<f32>, tex_coords: Vec<f32>) -> Self {
        Self {
            pos: pos.try_into().unwrap(),
            tex_coords: tex_coords.try_into().unwrap(),
        }
    }
}
