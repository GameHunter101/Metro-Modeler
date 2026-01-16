struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) col: vec4<f32>,
    @builtin(vertex_index) index: u32,
}

struct TransformData {
    @location(3) mat_0: vec4<f32>,
    @location(4) mat_1: vec4<f32>,
    @location(5) mat_2: vec4<f32>,
    @location(6) mat_3: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
    @location(0) real_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) col: vec4<f32>,
    @location(3) cam_pos: vec3<f32>,
}


struct Camera {
    mat: mat4x4<f32>,
    pos: vec3<f32>,
}

@group(0) @binding(0)
var<uniform> camera: Camera;

@vertex
fn main(input: VertexInput, transform: TransformData) -> VertexOutput {
    let transform_mat = mat4x4<f32>(
        transform.mat_0,
        transform.mat_1,
        transform.mat_2,
        transform.mat_3,
    );

    var out: VertexOutput;
    out.pos = camera.mat * (transform_mat * vec4f(input.position, 1.0));
    out.real_pos = input.position;
    out.normal = input.normal;
    out.col = input.col;
    out.cam_pos = camera.pos;
    return out;
}
