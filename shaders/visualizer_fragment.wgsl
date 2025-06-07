struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
    @location(0) col: vec4<f32>,
}

@fragment
fn main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.col;
}
