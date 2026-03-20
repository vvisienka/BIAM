// use std::collections::HashMap;
mod utils;

fn main() {
    let mut init_solution: [i32; 5]  = [1,2,3,4,5];
    println!("Unshuffled: {:?}", init_solution);
    println!("\n--- Testing Permutations ---");
    utils::generate_permutations(&mut init_solution);
    println!("Shuffled: {:?}", init_solution);

    println!("\n--- Testing Unique Pairs ---");
    let pair = utils::generate_unique_pairs(init_solution.len());
    println!("Random pair: {:?}", pair);

    println!("\n--- Testing Time Measurement ---");

    //func wrapper for time measurement
    fn tested_func() {
        let mut arr = [1, 2, 3, 4];
        utils::generate_permutations(&mut arr);
    }

    let end_time = 100;
    let min_iterations = 1;
    utils::measure_time(tested_func, end_time, min_iterations);

    // ===RANDOMNESS CHECK===

    // let mut distribution: HashMap<[i32; 4], usize> = HashMap::new();
    // for _ in 0..1000{
    //     utils::generate_permutations(&mut init_solution);
    //     if distribution.contains_key(&init_solution) {
    //         *distribution.get_mut(&init_solution).unwrap() += 1;
    //     } else {
    //         distribution.insert(init_solution, 1);
    // }}
    // println!("Distribution: {:?}", distribution);
}