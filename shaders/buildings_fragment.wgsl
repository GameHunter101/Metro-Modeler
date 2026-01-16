struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
    @location(0) real_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) col: vec4<f32>,
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
    return vec4f(in.col.xyz * dot(dir_to_light, in.normal), 1.0);
}
