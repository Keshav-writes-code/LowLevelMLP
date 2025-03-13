use rand::Rng;

fn random_float(rand_range: u8) -> f32 {
    let mut rng = rand::rng();
    let random_val: f32 = rng.random::<f32>();

    (random_val * (rand_range as f32) * 2.0) - 1.0
}
struct Neuron {
    pub value: f32,
    pub bias: f32,
    pub weights: Vec<f32>,
    pub prev_layer_neurons_count: u8,
}
impl Neuron {
    fn new(prev_layer_neurons_count: u8) -> Neuron {
        let rand_range: u8 = 1;
        let mut weights: Vec<f32> = Vec::new();
        for _ in 0..prev_layer_neurons_count {
            weights.push(random_float(rand_range));
        }

        Neuron {
            value: 0.0,
            bias: random_float(rand_range),
            weights,
            prev_layer_neurons_count,
        }
    }
}
struct Layer {
    size: u8,
    neurons: Vec<Neuron>,
}
impl Layer {
    fn new(size: u8, prev_layer_neurons_count: u8) -> Layer {
        let mut neurons: Vec<Neuron> = Vec::new();
        for _ in 0..size {
            neurons.push(Neuron::new(prev_layer_neurons_count));
        }

        Layer { size, neurons }
    }
    // TODO: Add Other Function of Layer Struct
}
// TODO: Add remaining Structs like MLP,
