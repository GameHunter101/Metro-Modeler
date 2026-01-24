@group(0) @binding(0) var tex: texture_2d<f32>;
@group(1) @binding(0) var blend: texture_2d<f32>;
@group(2) @binding(0) var sample: sampler;

struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
}

@fragment
fn main(input: VertexOutput) -> @location(0) vec4<f32> {
    let major_w_x = textureSample(blend, sample, input.tex_coords).x;
    let minor_w_x = textureSample(blend, sample, input.tex_coords).y;

    let major_w_y = 1.0 - major_w_x;
    let minor_w_y = 1.0 - minor_w_x;

    let sample = textureSample(tex, sample, input.tex_coords);

    let major_val = major_w_x * sample.x + major_w_y * sample.y;
    // let minor_val = minor_w_x * sample.z + minor_w_y * sample.w;

    return vec4f(vec3f(major_val), 1.0);
}
