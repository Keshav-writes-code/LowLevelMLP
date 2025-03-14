use rand::Rng;

mod mlp;

fn gen_random_ints(arr_len: usize, min: u16, max: u16) -> Vec<u16> {
    let mut rng = rand::rng();
    let mut vec: Vec<u16> = Vec::new();
    for _ in 0..arr_len {
        vec.push(rng.random_range(min..max));
    }
    vec
}
fn gen_random_floats_vector(vector_len: usize, min: f32, max: f32) -> Vec<f32> {
    let mut rng = rand::rng();
    let mut vec: Vec<f32> = Vec::new();
    for _ in 0..vector_len {
        vec.push(rng.random_range(min..max));
    }
    vec
}
fn gen_random_floats_2d(rows: usize, cols: usize, min: f32, max: f32) -> Vec<Vec<f32>> {
    let mut vec: Vec<Vec<f32>> = Vec::new();
    for _ in 0..rows {
        vec.push(gen_random_floats_vector(cols, min, max));
    }
    vec
}

fn gen_identity_2d_matrix(rows: usize, cols: usize) -> Vec<Vec<f32>> {
    let mut vec: Vec<Vec<f32>> = Vec::new();
    for i in 0..rows {
        let mut row: Vec<f32> = Vec::new();
        for j in 0..cols {
            if i == j {
                row.push(1.0);
            } else {
                row.push(0.0);
            }
        }
        vec.push(row);
    }
    vec
}
fn print_2d_matrix(matrix: &[Vec<f32>]) {
    println!("2D Array : ");
    for row in matrix.iter() {
        for val in row.iter() {
            print!("{:.2}, ", val);
        }
        println!();
    }
}
fn main() {
    let input_layer_size = 10;
    let hid_out_layer_count = 2;
    let hid_out_layer_sizes = gen_random_ints(hid_out_layer_count, 10, 20);

    // Create a new MLP
    let mut mlp = mlp::Mlp::new(input_layer_size, &hid_out_layer_sizes, 0.1);

    //Dataset with labels
    let inputs = gen_random_floats_2d(
        *hid_out_layer_sizes.last().expect("Array Empty") as usize,
        input_layer_size as usize,
        0.0,
        100.0,
    );
    let targets = gen_identity_2d_matrix(
        *hid_out_layer_sizes.last().expect("Array Empty") as usize,
        *hid_out_layer_sizes.last().expect("Array Empty") as usize,
    );
    mlp.describe();
    mlp.print_params_count();

    mlp.train(&inputs, &targets, 1000);

    mlp.predict(&inputs, &targets);
    println!("Done");
}
