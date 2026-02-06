@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var blend: texture_2d<f32>;
@group(0) @binding(2) var sample: sampler;

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

    let major_val = (major_w_x * sample.x + major_w_y * sample.y - 0.3) * 2.0;
    let minor_val = (minor_w_x * sample.z + minor_w_y * sample.w - 0.3) * 2.0;

    var blend_fac = 0.0;
    if major_val > minor_val {
        blend_fac = 0.0;
    } else {
        blend_fac = 1.0;
    }

    let col_major = vec3f(0.0, 1.0, 0.0) * major_val;
    let col_minor = vec3f(1.0, 0.0, 1.0) * minor_val;

    return vec4f(col_major + col_minor, 1.0);
}
