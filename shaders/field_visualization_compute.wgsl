@group(0) @binding(0) var input: texture_storage_2d<rgba8unorm, read_write>;
@group(0) @binding(1) var major_eigenvector: texture_2d<f32>;
@group(0) @binding(2) var minor_eigenvector: texture_2d<f32>;

@group(0) @binding(3) var output: texture_storage_2d<rgba8unorm, read_write>;

fn random(coord: vec2<u32>) -> f32 {
    let input = (coord.x + coord.y * 512u);
    let state = input * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return f32((word >> 22u) ^ word) / f32(0xffffffff);
}

fn bilerp(coord: vec2<f32>) -> vec4<f32> {
    let coord_floor = vec2u(coord);
    let bl = textureLoad(input, coord_floor);
    let br = textureLoad(input, coord_floor + vec2u(1u, 0u));
    let tl = textureLoad(input, coord_floor + vec2u(0u, 1u));
    let tr = textureLoad(input, coord_floor + vec2u(1u, 1u));

    return mix(mix(bl, br, fract(coord.x)), mix(tl, tr, fract(coord.x)), fract(coord.y));
}

fn lic(origin_pos: vec2<f32>, len: u32, is_x: bool, tex: texture_2d<f32>) -> f32 {
    var sum = textureLoad(input, vec2u(origin_pos)).x;
    var pos_1 = vec2f(origin_pos) + vec2f(0.5, 0.5);
    var pos_2 = pos_1;

    let delta_s = 1.0;

    var i = 1u;

    for (; i < len; i++) {
        if (pos_1.x < 0.0 || pos_1.x > 512.0) || (pos_2.x < 0.0 || pos_2.x > 512.0) || 
            (pos_1.y < 0.0 || pos_1.y > 512.0) || (pos_2.y < 0.0 || pos_2.y > 512.0) {
            break;
        }

        let noise_fac = 0.1;
        let noise_1 = mix(-1.0, 1.0, random(vec2u(pos_1)));
        let noise_2 = mix(-1.0, 1.0, random(vec2u(pos_2)));

        sum += (bilerp(pos_1).x + bilerp(pos_2).x) + noise_fac * (noise_1 + noise_2);

        var pos_1_dir = textureLoad(tex, vec2u(pos_1), 0);
        var pos_2_dir = textureLoad(tex, vec2u(pos_2), 0);
        if is_x {
            pos_1 += normalize(pos_1_dir.xy) * delta_s;
            pos_2 -= normalize(pos_2_dir.xy) * delta_s;
        } else {
            pos_1 += normalize(pos_1_dir.zw) * delta_s;
            pos_2 -= normalize(pos_2_dir.zw) * delta_s;
        }
    }

    return sum / (2.0 * f32(i) + 1.0);
}

@compute
@workgroup_size(128)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let major_dir = textureLoad(major_eigenvector, id.xy, 0);
    let lic_len = 7u;

    let major_prev_val_x = lic(vec2f(id.xy), lic_len, true, major_eigenvector);
    let major_prev_val_y = lic(vec2f(id.xy), lic_len, false, major_eigenvector);

    let minor_dir = textureLoad(minor_eigenvector, id.xy, 0);

    let minor_prev_val_x = lic(vec2f(id.xy), lic_len, true, minor_eigenvector);
    let minor_prev_val_y = lic(vec2f(id.xy), lic_len, false, minor_eigenvector);

    textureStore(output, vec2i(id.xy), vec4f(
        major_prev_val_x, major_prev_val_y, minor_prev_val_x, minor_prev_val_y
    ));
}
