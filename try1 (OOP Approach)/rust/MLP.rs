use lazy_static::lazy_static;
use std::sync::Mutex;
use rand::Rng;

struct Neuron {
    pub value: f32,
    pub bias: f32,
    pub weights: Vec<f32>,
    pub prevLayerNeurons_count: u8;
}
impl Neuron{
    fn new(prevLayerNeurons_count:u8)->Neuron{
        let mut rng = rand::rng();
        
        let randRange:f32= 1.0;


        Neuron{
            value: 0.0,
            bias: rng.random.
            prevLayerNeurons_count,
            // TODO: Figure out a way to generate random values in rust


        }


    }
}
