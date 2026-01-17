struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
    @location(0) real_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tex_coords: vec2<f32>,
    @location(3) cam_pos: vec3<f32>,
}

@fragment
fn main(in: VertexOutput) -> @location(0) vec4<f32> {
    let light_pos = vec3(0.0, 150.0, 0.0);
    let dir_to_light = normalize(light_pos - in.real_pos);

    /* let c_cool = vec3f(0.0, 0.0, 0.55) + 0.25 * in.col.xyz;
    let c_warm = vec3f(0.3, 0.3, 0.0) + 0.25 * in.col.xyz;
    let c_highlight = vec3f(1.0);

    let t = (dot(in.normal, dir_to_light) + 1.0)/2.0;
    let r = 2.0 * (dot(in.normal, dir_to_light)) * in.normal - dir_to_light;
    let s = clamp(100.0 * dot(r, in.cam_pos - in.real_pos), 0.0, 1.0); */

    // return vec4f(s * c_highlight + (1.0 - s) * (t * c_warm + (1 - t) * c_cool), 1.0);
    let coord = vec2f((in.tex_coords.x % 0.5) * 2.0, in.tex_coords.y % 1.0);
    let shifted_coord = (coord - vec2f(0.5)) * 2.0;
    let window_mask = f32(abs(shifted_coord.x) > 0.1 && abs(shifted_coord.y) > 0.1);
    let window_color = vec3f(86.0, 166.0, 235.0) / 255.0;
    let frame_color = vec3f(0.542, 0.497, 0.449);

    let albedo = window_mask * window_color + (1.0 - window_mask) * frame_color;
    return vec4f(albedo* dot(dir_to_light, in.normal), 1.0);
}
