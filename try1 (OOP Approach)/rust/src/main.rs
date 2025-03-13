mod mlp;
fn main() {
    let mut layers: Vec<u16> = Vec::new();
    let mut inputs: Vec<f32> = Vec::new();
    layers.push(40);
    layers.push(20);
    inputs.push(0.12);
    inputs.push(0.42);
    inputs.push(0.11);
    let mut mlp = mlp::Mlp::new(10, &layers, 0.03);
    mlp.describe();
    mlp.feed_forward(&inputs);
}
