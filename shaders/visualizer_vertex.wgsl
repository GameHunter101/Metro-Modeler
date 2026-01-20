struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) col: vec4<f32>,
    @builtin(vertex_index) index: u32,
}

struct TransformData {
    @location(2) mat_0: vec4<f32>,
    @location(3) mat_1: vec4<f32>,
    @location(4) mat_2: vec4<f32>,
    @location(5) mat_3: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
    @location(0) col: vec4<f32>,
}


struct Camera {
    mat: mat4x4<f32>
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
    // out.pos = camera.mat * (transform_mat * vec4f(input.position, 1.0));
    out.pos = vec4f(input.position, 1.0);
    out.col = input.col;
    return out;
}
