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
        }
    }
}
pub struct Layer {
    neurons: Vec<Neuron>,
}
impl Layer {
    pub fn new(size: u16, prev_layer_neurons_count: u8) -> Layer {
        let mut neurons: Vec<Neuron> = Vec::new();
        for _ in 0..size {
            neurons.push(Neuron::new(prev_layer_neurons_count));
        }

        Layer { neurons }
    }
    pub fn show_neurons(&self) {
        for (i, neuron) in self.neurons.iter().enumerate() {
            println!("Neuron :- {}", i);
            println!("value :- {}", neuron.value);
            println!("bias :- {}", neuron.bias);
            println!("weights :");
            for weight in neuron.weights.iter() {
                print!("{:.2}, ", weight);
            }
            println!("\n");
        }
    }
}
// TODO: Add remaining Structs like MLP,
