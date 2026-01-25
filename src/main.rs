#![allow(unstable_name_collisions)]

use image::{EncodableLayout, GenericImage, ImageBuffer, ImageReader, Rgba};
use rand::Rng;
use rayon::prelude::*;
use street_graph::path_to_graph;
use street_plan::{HermiteCurve, SeedPoint, merge_road_endings, resample_curve, trace_street_plan};
use tensor_field::{DesignElement, GRID_SIZE, Point, TensorField};
use v4::ecs::compute::Compute;
use v4::ecs::material::{GeneralTexture, ShaderBufferAttachment};
use v4::{
    builtin_components::{
        camera_component::CameraComponent,
        mesh_component::{MeshComponent, VertexDescriptor},
        transform_component::TransformComponent,
    },
    ecs::material::{ShaderAttachment, ShaderTextureAttachment},
    engine_support::texture_support::Texture,
    scene,
};
use wgpu::util::DeviceExt;
use wgpu::vertex_attr_array;

use crate::field_visualization_component::FieldVisualizationComponent;
use crate::{
    building_generation::footprint_to_building, tensor_field::EvalEigenvectors,
    triangulation::triangulate_faces, water_mask::mask_to_elements,
};

mod building_generation;
mod field_visualization_component;
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
        theta: std::f32::consts::FRAC_PI_2,
        length: 100.0,
    };

    let radial_element = DesignElement::Radial {
        center: Point::new(256.0, 256.0),
    };

    let grid_element_2 = DesignElement::Grid {
        center: Point::new(0.0, 400.0),
        theta: 0.7,
        length: 10.0,
    };

    let city_center = *radial_element.center().as_ref().unwrap();

    let mut water_mask_image = ImageReader::open("water_mask.png")
        .unwrap()
        .decode()
        .unwrap();

    water_mask_image = water_mask_image.resize(
        water_mask_image.width() + 2,
        water_mask_image.height() + 2,
        image::imageops::FilterType::Nearest,
    );
    for x in 0..water_mask_image.width() {
        water_mask_image.put_pixel(x, 0, Rgba([0, 0, 0, 255]));
        water_mask_image.put_pixel(x, water_mask_image.height() - 1, Rgba([0, 0, 0, 255]));
    }
    for y in 0..water_mask_image.height() {
        water_mask_image.put_pixel(0, y, Rgba([0, 0, 0, 255]));
        water_mask_image.put_pixel(water_mask_image.width() - 1, y, Rgba([0, 0, 0, 255]));
    }

    let (water_edge, edge_elements) = mask_to_elements(&water_mask_image, 10);
    println!("Water edge length: {}", water_edge.len());

    let tensor_field = TensorField::new(
        vec![grid_element, radial_element, grid_element_2]
        .into_iter()
        .chain(edge_elements)
        .collect(),
        0.0004,
    );

    println!("Field creation: {}", start_time.elapsed().as_millis() as f32 / 1000.0);

    let (major_network_major_curves_unconnected, major_network_minor_curves_unconnected) =
        trace_street_plan(
            &tensor_field,
            "water_mask.png",
            street_plan::TraceSeeds::Random(30),
            city_center,
            |_| 20.0,
            300.0,
            100.0,
            5,
            Vec::new(),
            Vec::new(),
            &water_mask_image,
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
            |_| 5.0,
            100.0,
            50.0,
            4,
            major_network_curves[..major_network_major_curves_len].to_vec(),
            major_network_curves[major_network_major_curves_len..].to_vec(),
            &water_mask_image,
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
    let verts_for_triangulation: Vec<Vertex> = faces.into_iter().flatten().collect();

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
        .features(
            wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
                | wgpu::Features::POLYGON_MODE_LINE,
        )
        .limits(wgpu::Limits {
            max_bind_groups: 5,
            ..Default::default()
        })
        .build()
        .await;

    let sample_factor = 14;
    let vector_opacity = 0.2;

    let rendering_manager = engine.rendering_manager();
    let device = rendering_manager.device();
    let queue = rendering_manager.queue();

    let mut visualization_input_image =
        ImageBuffer::from_pixel(GRID_SIZE, GRID_SIZE, Rgba([0_u8, 0, 0, 255]));

    let mut rng = rand::rng();

    for x in 0..GRID_SIZE {
        for y in 0..GRID_SIZE {
            let val1 = rng.random();
            let val2 = rng.random();
            visualization_input_image.put_pixel(x, y, Rgba([val1, val2, 0, 0]));
        }
    }

    let visualization_input_texture = Texture::from_bytes(
        visualization_input_image.as_bytes(),
        (GRID_SIZE, GRID_SIZE),
        device,
        queue,
        wgpu::TextureFormat::Rgba8Unorm,
        Some(wgpu::StorageTextureAccess::ReadWrite),
        false,
        wgpu::TextureUsages::COPY_DST,
    );

    let visualization_output_image =
        ImageBuffer::from_pixel(GRID_SIZE, GRID_SIZE, Rgba([0_u8, 0, 0, 255]));

    let visualization_output_texture = Texture::from_bytes(
        visualization_output_image.as_bytes(),
        (GRID_SIZE, GRID_SIZE),
        device,
        queue,
        wgpu::TextureFormat::Rgba8Unorm,
        Some(wgpu::StorageTextureAccess::ReadWrite),
        false,
        wgpu::TextureUsages::COPY_SRC,
    );

    let mut major_eigenvector_image = ImageBuffer::new(GRID_SIZE, GRID_SIZE);
    let mut minor_eigenvector_image = ImageBuffer::new(GRID_SIZE, GRID_SIZE);
    let mut blending_image = ImageBuffer::new(GRID_SIZE, GRID_SIZE);

    let smooth_field = |eigenvector: Point| {
        let field_x = if eigenvector.x >= 0.0 {
            eigenvector
        } else {
            -eigenvector
        };

        let field_y = if eigenvector.y >= 0.0 {
            eigenvector
        } else {
            -eigenvector
        };

        (field_x * 255.0, field_y * 255.0)
    };

    for y in 0..GRID_SIZE {
        for x in 0..GRID_SIZE {
            let eigenvectors = tensor_field
                .evaluate_smoothed_field_at_point(Point::new(x as f32, y as f32))
                .eigenvectors();
            let major_eigenvector = eigenvectors.major.normalize();
            let minor_eigenvector = eigenvectors.minor.normalize();
            let (major_field_x, major_field_y) = smooth_field(major_eigenvector);
            let (minor_field_x, minor_field_y) = smooth_field(minor_eigenvector);

            major_eigenvector_image.put_pixel(
                x,
                y,
                Rgba([
                    major_field_x.x as u8,
                    major_field_x.y as u8,
                    major_field_y.x as u8,
                    major_field_y.y as u8,
                ]),
            );

            minor_eigenvector_image.put_pixel(
                x,
                y,
                Rgba([
                    minor_field_x.x as u8,
                    minor_field_x.y as u8,
                    minor_field_y.x as u8,
                    minor_field_y.y as u8,
                ]),
            );

            let major_w_x = ((major_eigenvector.x * major_eigenvector.x) * 255.0) as u8;
            let minor_w_x = ((minor_eigenvector.x * minor_eigenvector.x) * 255.0) as u8;

            blending_image.put_pixel(x, y, Rgba([major_w_x, minor_w_x, 0, 0]));
        }
    }

    let major_eigenvector_texture = Texture::from_bytes(
        major_eigenvector_image.as_bytes(),
        (GRID_SIZE, GRID_SIZE),
        device,
        queue,
        wgpu::TextureFormat::Rgba8Unorm,
        None,
        false,
        wgpu::TextureUsages::empty(),
    );

    let minor_eigenvector_texture = Texture::from_bytes(
        minor_eigenvector_image.as_bytes(),
        (GRID_SIZE, GRID_SIZE),
        device,
        queue,
        wgpu::TextureFormat::Rgba8Unorm,
        None,
        false,
        wgpu::TextureUsages::empty(),
    );

    let blending_texture = Texture::from_bytes(
        blending_image.as_bytes(),
        (GRID_SIZE, GRID_SIZE),
        device,
        queue,
        wgpu::TextureFormat::Rgba8Unorm,
        None,
        true,
        wgpu::TextureUsages::empty(),
    );

    let tensorfield_vis_img =
        ImageBuffer::from_pixel(GRID_SIZE, GRID_SIZE, Rgba([0_u8, 0, 0, 255]));
    let tensorfield_vis_tex = Texture::from_bytes(
        tensorfield_vis_img.as_bytes(),
        (GRID_SIZE, GRID_SIZE),
        device,
        queue,
        wgpu::TextureFormat::Rgba8Unorm,
        None,
        true,
        wgpu::TextureUsages::COPY_DST,
    );

    scene! {
        scene: visualizer,
        active_camera: "cam",
        "eigenvectors" = {
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
                },
            },
            components: [
                MeshComponent(
                    vertices: vec![
                        /* (0..GRID_SIZE / sample_factor).flat_map(|x| (0..GRID_SIZE / sample_factor).flat_map(|y| {
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
                        }).collect::<Vec<_>>()).collect(), */
                        water_edge.iter().flat_map(|&(point, dir)| {
                            let start = normalize_vector(point);
                            let end = normalize_vector(point + dir.normalize() * 15.0);
                            [
                                LineVertex {pos: [start.x, start.y, 0.0], col: [1.0, 1.0, 1.0, vector_opacity]},
                                LineVertex {pos: [end.x, end.y, 0.0], col: [1.0, 1.0, 1.0, vector_opacity]}
                            ]
                        }).collect()
                    ],
                    enabled_models: vec![(0, None)/* , (1, None) */]
                ),
                TransformComponent(position: nalgebra::Vector3::zeros())
            ]
        },
        "field_visualization" = {
            material: {
                pipeline: {
                    vertex_shader_path: "./shaders/tensorfield_vertex.wgsl",
                    fragment_shader_path: "./shaders/tensorfield_fragment.wgsl",
                    vertex_layouts: [Vertex::vertex_layout()],
                    uses_camera: false,
                },
                attachments: [
                    Texture(
                        texture: tensorfield_vis_tex,
                        visibility: wgpu::ShaderStages::FRAGMENT
                    ),
                    Texture(
                        texture: blending_texture,
                        visibility: wgpu::ShaderStages::FRAGMENT
                    ),
                ],
                ident: "vis_mat"
            },
            components: [
                MeshComponent(
                    vertices: vec![
                        vec![
                            Vertex {
                                pos: [-1.0, 1.0, 0.2],
                                normal: [0.0; 3],
                                tex_coords: [0.0, 1.0],
                            },
                            Vertex {
                                pos: [-1.0, -1.0, 0.2],
                                normal: [0.0; 3],
                                tex_coords: [0.0, 0.0],
                            },
                            Vertex {
                                pos: [1.0, -1.0, 0.2],
                                normal: [0.0; 3],
                                tex_coords: [1.0, 0.0],
                            },
                            Vertex {
                                pos: [1.0, 1.0, 0.2],
                                normal: [0.0; 3],
                                tex_coords: [1.0, 1.0],
                            },
                        ]
                    ],
                    indices: vec![
                        vec![0, 1, 2, 0, 2, 3],
                    ],
                    enabled_models: vec![(0, None)]
                ),
                FieldVisualizationComponent(compute: ident("compute"), material: ident("vis_mat"))
            ],
            computes: [
                Compute(
                    input: vec![
                        ShaderAttachment::Texture(ShaderTextureAttachment {
                            texture: visualization_input_texture,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            extra_usages: wgpu::TextureUsages::empty(),
                        }),
                        ShaderAttachment::Texture(ShaderTextureAttachment {
                            texture: major_eigenvector_texture,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            extra_usages: wgpu::TextureUsages::empty(),
                        }),
                        ShaderAttachment::Texture(ShaderTextureAttachment {
                            texture: minor_eigenvector_texture,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            extra_usages: wgpu::TextureUsages::empty(),
                        }),
                        ShaderAttachment::Buffer(ShaderBufferAttachment::new(
                            device,
                            bytemuck::cast_slice(&[0_u32]),
                            wgpu::BufferBindingType::Uniform,
                            wgpu::ShaderStages::COMPUTE,
                            wgpu::BufferUsages::COPY_DST
                        )),
                    ],
                    output: ShaderAttachment::Texture(ShaderTextureAttachment {
                        texture: visualization_output_texture,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        extra_usages: wgpu::TextureUsages::empty(),
                    }),
                    shader_path: "shaders/field_visualization_compute.wgsl",
                    workgroup_counts: (4, GRID_SIZE, 1),
                    ident: "compute"
                )
            ],
        },
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
                                        pos: [ current_pos.x, current_pos.y, 0.0],
                                        col: [0.0, 0.0, 1.0, 1.0]
                                    },
                                    LineVertex {
                                        pos: [ next_pos.x, next_pos.y, 0.0,],
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
                                        pos: [ current_pos.x, current_pos.y, 0.0],
                                        col: [1.0, 0.0, 0.0, 1.0]
                                    },
                                    LineVertex {
                                        pos: [ next_pos.x, next_pos.y, 0.0],
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
