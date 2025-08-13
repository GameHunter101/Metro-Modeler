use image::{EncodableLayout, ImageBuffer};
use nalgebra::Vector2;
use street_graph::path_to_graph;
use street_plan::{HermiteCurve, SeedPoint, merge_road_endings, resample_curve, trace_street_plan};
use tensor_field::{DesignElement, EvalEigenvectors, GRID_SIZE, Point, TensorField};
use v4::{
    builtin_components::mesh_component::{MeshComponent, VertexDescriptor},
    engine_support::texture_support::Texture,
    scene,
};
use wgpu::vertex_attr_array;

mod event_queue;
mod status;
mod street_graph;
mod street_plan;
mod tensor_field;

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

    let major_network: Result<Vec<Vec<Point>>, tokio::task::JoinError> = futures::future::join_all(
        major_network_curves
            .clone()
            .into_iter()
            .map(|curve| tokio::spawn(async { resample_curve(curve, 20) })),
    )
    .await
    .into_iter()
    .collect();

    let major_network = major_network.unwrap();

    let major_network_dcel = path_to_graph(&major_network_curves);

    println!("Faces: {:?}", major_network_dcel.faces());

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
    let minor_network_major_curves = merge_road_endings(
        &minor_network_curves_unconnected,
        minor_network_merge_distance,
    );
    let minor_network_minor_curves = merge_road_endings(
        &minor_network_curves_unconnected,
        minor_network_merge_distance,
    );

    let minor_network: Result<Vec<Vec<Point>>, tokio::task::JoinError> = futures::future::join_all(
        minor_network_major_curves
            .into_iter()
            .chain(minor_network_minor_curves)
            .map(|curve| tokio::spawn(async { resample_curve(curve, 20) })),
    )
    .await
    .into_iter()
    .collect();

    let minor_network = minor_network.unwrap();

    println!("{}", start_time.elapsed().as_millis() as f32 / 1000.0);

    let mut engine = v4::V4::builder()
        .features(wgpu::Features::POLYGON_MODE_LINE | wgpu::Features::POLYGON_MODE_POINT)
        .window_settings(
            GRID_SIZE as u32 * 2,
            GRID_SIZE as u32 * 2,
            "Visualizer",
            None,
        )
        .build()
        .await;

    let sample_factor = 14;

    let mut norm_tex = ImageBuffer::new(GRID_SIZE, GRID_SIZE);

    for (x, y, pix) in norm_tex.enumerate_pixels_mut() {
        let val = (tensor_field
            .evaluate_smoothed_field_at_point(Vector2::new(x as f32, y as f32))
            .norm()
            > 0.0001) as u8
            * 255;
        *pix = image::Rgba([val, val, val, 50]);
    }

    let rendering_manager = engine.rendering_manager();
    let device = rendering_manager.device();
    let queue = rendering_manager.queue();

    let vector_opacity = 0.2;

    scene! {
        scene: visualizer,
        "eigenvectors" = {
            material: {
                pipeline: {
                    vertex_shader_path: "./shaders/visualizer_vertex.wgsl",
                    fragment_shader_path: "./shaders/visualizer_fragment.wgsl",
                    vertex_layouts: [Vertex::vertex_layout()],
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
                            let point = Vector2::new(x as f32 * sample_factor as f32, y as f32 * sample_factor as f32);
                            let tensor = tensor_field.evaluate_smoothed_field_at_point(point);
                            let eigenvectors = tensor.eigenvectors();
                            let maj = eigenvectors.major.normalize() * (sample_factor - 1) as f32;
                            let min = eigenvectors.minor.normalize() * (sample_factor - 1) as f32;
                            let maj_point = normalize_vector(point + maj);
                            let min_point = normalize_vector(point + min);
                            let norm_point = normalize_vector(point);
                            [
                                Vertex {pos: [norm_point.x, norm_point.y, 0.0], col: [1.0, 0.0, 0.0, vector_opacity]}, Vertex {pos: [maj_point.x, maj_point.y, 0.0], col: [1.0, 0.0, 0.0, vector_opacity]},
                                Vertex {pos: [norm_point.x, norm_point.y, 0.0], col: [0.0, 1.0, 0.0, vector_opacity]}, Vertex {pos: [min_point.x, min_point.y, 0.0], col: [0.0, 1.0, 0.0, vector_opacity]}
                            ]
                        }).collect::<Vec<_>>()).collect()
                    ],
                    enabled_models: vec![(0, None)]
                ),
            ]
        },
        "degenerate_points" = {
            material: {
                pipeline: {
                    vertex_shader_path: "./shaders/degenerate_point_vert.wgsl",
                    fragment_shader_path: "./shaders/degenerate_point_frag.wgsl",
                    vertex_layouts: [TexVertex::vertex_layout()],
                    uses_camera: false,
                },
                attachments: [
                    Texture(
                    texture: v4::ecs::material::GeneralTexture::Regular(
                        Texture::from_bytes(
                            norm_tex.as_bytes(),
                            (GRID_SIZE, GRID_SIZE),
                            device,
                            queue,
                            wgpu::TextureFormat::Rgba8Unorm,
                            false,
                            true,
                        )
                    ),
                    visibility: wgpu::ShaderStages::FRAGMENT,
                )],
            },
            components: [
                MeshComponent(
                    vertices: vec![vec![
                        TexVertex {
                            pos: [-1.0, 1.0, 0.1],
                            tex_coords: [0.0, 1.0]
                        },
                        TexVertex {
                            pos: [-1.0, -1.0, 0.1],
                            tex_coords: [0.0, 0.0]
                        },
                        TexVertex {
                            pos: [1.0, -1.0, 0.1],
                            tex_coords: [1.0, 0.0]
                        },
                        TexVertex {
                            pos: [1.0, 1.0, 0.1],
                            tex_coords: [1.0, 1.0]
                        },
                    ]],
                    indices: vec![vec![0,1,2,0,2,3]],
                    enabled_models: vec![(0, None)]
                )
            ]
        },
        "major_network" = {
            material: {
                pipeline: {
                    vertex_shader_path: "./shaders/visualizer_vertex.wgsl",
                    fragment_shader_path: "./shaders/visualizer_fragment.wgsl",
                    vertex_layouts: [Vertex::vertex_layout()],
                    uses_camera: false,
                    geometry_details: {
                        topology: wgpu::PrimitiveTopology::LineStrip,
                        polygon_mode: wgpu::PolygonMode::Line,
                    },
                    ident: "network_pipeline",
                }
            },
            components: [
                MeshComponent(
                    vertices:
                        major_network.iter().map(|arr| {
                            arr.iter().map(|vec| {
                                    Vertex {
                                        pos: [
                                            2.0 * vec.x / GRID_SIZE as f32 - 1.0,
                                            2.0 * vec.y / GRID_SIZE as f32 - 1.0,
                                            0.0,
                                        ],
                                        col: [0.0, 0.0, 1.0, 1.0]
                                    }
                            }).collect::<Vec<_>>()
                        }).collect(),
                    enabled_models: major_network.iter().enumerate().map(|(i, _)| (i, None)).collect()
                )
            ]
        },
        "minor_network" = {
            material: {
                pipeline: ident("network_pipeline"),
            },
            components: [
                MeshComponent(
                    vertices:
                        minor_network.iter().map(|arr| {
                            arr.iter().map(|vec| {
                                    Vertex {
                                        pos: [
                                            2.0 * vec.x / GRID_SIZE as f32 - 1.0,
                                            2.0 * vec.y / GRID_SIZE as f32 - 1.0,
                                            0.0,
                                        ],
                                        col: [1.0, 0.0, 0.0, 1.0]
                                    }
                            }).collect::<Vec<_>>()
                        }).collect(),
                    enabled_models: minor_network.iter().enumerate().map(|(i, _)| (i, None)).collect()
                )
            ]
        }
    }

    engine.attach_scene(visualizer);

    engine.main_loop().await;
}

fn normalize_vector(vec: Vector2<f32>) -> Vector2<f32> {
    Vector2::new(
        2.0 * vec.x / GRID_SIZE as f32 - 1.0,
        2.0 * vec.y / GRID_SIZE as f32 - 1.0,
    )
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
