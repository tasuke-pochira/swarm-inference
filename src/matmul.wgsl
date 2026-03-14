@group(0) @binding(4) var<uniform> uniforms: Uniforms;

@group(0) @binding(0) var<storage, read> weights: array<f32>;
@group(0) @binding(1) var<storage, read> input_vec: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output_vec: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= uniforms.out_dim) {
        return;
    }

    var sum: f32 = bias[idx];
    let in_dim = uniforms.in_dim;
    for (var i: u32 = 0u; i < in_dim; i = i + 1u) {
        let weight = weights[idx * in_dim + i];
        sum = sum + weight * input_vec[i];
    }
    output_vec[idx] = sum;
}

struct Uniforms {
    in_dim: u32,
    out_dim: u32,
};
