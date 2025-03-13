mod mlp;
fn main() {
    let layer = mlp::Layer::new(65000, 20);
    layer.show_neurons();
}
