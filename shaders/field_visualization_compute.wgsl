@group(0) @binding(0) var input: texture_storage_2d<rgba8unorm, read_write>;
@group(1) @binding(0) var eigenvectors: texture_2d<f32>;
// @group(2) @binding(0) var minor_eigenvectors: texture_2d<f32>;
@group(2) @binding(0) var<uniform> time: u32;

@group(3) @binding(0) var output: texture_storage_2d<rgba8unorm, read_write>;

fn random(coord: vec2<u32>) -> f32 {
    let input = (coord.x + coord.y * 512) * time;
    let state = input * 747796405 + 2891336453;
    let word = ((state >> ((state >> 28) + 4)) ^ state) * 277803737;
    return f32((word >> 22) ^ word) / f32(0xffffffff);
}

fn bilerp(coord: vec2<f32>) -> vec4<f32> {
    let coord_floor = vec2u(coord);
    let bl = textureLoad(input, coord_floor);
    let br = textureLoad(input, coord_floor + vec2u(1, 0));
    let tl = textureLoad(input, coord_floor + vec2u(0, 1));
    let tr = textureLoad(input, coord_floor + vec2u(1, 1));

    return mix(mix(bl, br, fract(coord.x)), mix(tl, tr, fract(coord.x)), fract(coord.y));
}

@compute
@workgroup_size(128)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let major_prev_position = vec2f(id.xy) - 0.5 * textureLoad(eigenvectors, id.xy, 0).xy;
    let major_prev_val = bilerp(major_prev_position).x;

    let minor_prev_position = vec2f(id.xy) - 0.5 * textureLoad(eigenvectors, id.xy, 0).zw;
    let minor_prev_val = bilerp(minor_prev_position).y;

    let noise_x = 0.05 * mix(-1.0, 1.0, random(id.xy));
    let noise_y = 0.05 * mix(-1.0, 1.0, random(id.xy * 2));

    textureStore(output, id.xy, vec4f(major_prev_val + noise_x, minor_prev_val + noise_y, 0.0, 1.0));
}
