#![allow(unstable_name_collisions)]

use image::{GenericImage, ImageReader, Rgba};
use street_graph::path_to_graph;
use tensor_field::{DesignElement, GRID_SIZE, Point, TensorField};
use v4::ecs::compute::Compute;
use v4::ecs::material::ShaderBufferAttachment;
use v4::{
    builtin_components::{
        camera_component::CameraComponent,
        mesh_component::{MeshComponent, VertexDescriptor},
        transform_component::TransformComponent,
    },
    ecs::material::{ShaderAttachment, ShaderTextureAttachment},
    scene,
};
use wgpu::{Features, vertex_attr_array};

use crate::field_visualization_component::{
    FieldVisualizationComponent, create_visualizer_textures,
};
use crate::street_plan::{DSepFunc, StreetPlanNetworks, TraceParams, generate_street_plan};
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

    println!(
        "Field creation: {}",
        start_time.elapsed().as_millis() as f32 / 1000.0
    );

    let major_params = TraceParams {
        d_sep: DSepFunc::new(Box::new(|_| 20.0)),
        max_len: 300.0,
        min_len: 100.0,
        iter_count: 5,
    };

    let minor_params = TraceParams {
        d_sep: DSepFunc::new(Box::new(|_| 5.0)),
        max_len: 100.0,
        min_len: 50.0,
        iter_count: 4,
    };

    let StreetPlanNetworks {
        major_network,
        minor_network,
        all_curves,
    } = generate_street_plan(
        major_params,
        minor_params,
        &tensor_field,
        &water_mask_image,
        Some(city_center),
        street_plan::TraceSeeds::Random(30),
    );

    println!("{}", start_time.elapsed().as_millis() as f32 / 1000.0);

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
        .window_settings(
            GRID_SIZE as u32 * 2,
            GRID_SIZE as u32 * 2,
            "Visualizer",
            None,
        )
        .hide_cursor(true)
        .antialiasing_enabled(true)
        .features(
            Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
                | Features::POLYGON_MODE_LINE
                | Features::IMMEDIATES,
        )
        .limits(wgpu::Limits {
            max_bind_groups: 5,
            max_immediate_size: 4,
            ..Default::default()
        })
        .build()
        .await;

    let sample_factor = 14;
    let vector_opacity = 0.2;

    let rendering_manager = engine.rendering_manager();
    let device = rendering_manager.device();
    let queue = rendering_manager.queue();

    let (compute_textures, [blending_tex, tensorfield_vis_tex], compute_output_tex) =
        create_visualizer_textures(device, queue, &tensor_field);

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
                        texture: blending_tex,
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
                FieldVisualizationComponent(
                    compute: ident("compute"),
                    material: ident("vis_mat"),
                    street_mat: ident("network_mat"),
                    street_transform: ident("street_transform"),
                    plot_entity: ident("plots"),
                )
            ],
            computes: [
                Compute(
                    input: compute_textures.into_iter().map(|tex|
                        ShaderAttachment::Texture(ShaderTextureAttachment {
                            texture: tex,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            extra_usages: wgpu::TextureUsages::empty(),
                        })
                    ).chain(Some(ShaderAttachment::Buffer(ShaderBufferAttachment::new(
                        device,
                        bytemuck::cast_slice(&[0_u32]),
                        wgpu::BufferBindingType::Uniform,
                        wgpu::ShaderStages::COMPUTE,
                        wgpu::BufferUsages::COPY_DST
                    )))).collect(),
                    output: ShaderAttachment::Texture(ShaderTextureAttachment {
                        texture: compute_output_tex,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        extra_usages: wgpu::TextureUsages::empty(),
                    }),
                    shader_path: "shaders/field_visualization_compute.wgsl",
                    workgroup_counts: (4, GRID_SIZE, 1),
                    ident: "compute"
                )
            ],
        },
        "street_network" = {
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
                    immediate_size: 4,
                    ident: "network_pipeline",
                },
                immediate_data: bytemuck::cast_slice(&[0_u32]).to_vec(),
                ident: "network_mat",
            },
            components: [
                MeshComponent(
                    vertices:
                        vec![
                            major_network.iter().flat_map(|arr| {
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
                            }).collect(),
                            minor_network.iter().flat_map(|arr| {
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
                            }).collect()
                        ],
                    enabled_models: vec![(0, None), (1, None)]
                ),
                TransformComponent(position: nalgebra::Vector3::new(0.0, 0.0, 0.0), ident: "street_transform")
            ]
        },
        "plots" = {
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
            ],
            is_enabled: false,
        },
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
