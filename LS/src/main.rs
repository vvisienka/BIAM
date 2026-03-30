// use std::collections::HashMap;
mod utils;
use std::fs;
use std::error::Error;
use rand::Rng;

fn main() -> Result<(), Box <dyn Error>>{
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
    let min_iterations = 10;
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

    // === LOAD DATA ===
    let mut instances: [String; 10] = Default::default(); //intialized with empty strings
    let mut solutions: [String; 10] = Default::default();
    let mut instance_count = 0;
    let mut solution_count = 0;

    //1. Collect the file names
    for entry in fs::read_dir("../data")?{
        
        let path = entry?.path();
        if let Some(path_str) = path.to_str(){

        if path_str.contains("_solution.dat"){
            solutions[solution_count] = path_str.to_string();
            solution_count += 1;
        } else if path_str.contains(".dat"){
            instances[instance_count] = path_str.to_string();
            instance_count += 1;
        }
    }}

    
    //2. Load the actual data
    for file in instances{
        println!("\nLoading: {}", file);
        let size = utils::load_data(&file)?;
        // // Test if matrices are symmetric
        // if utils::test_symmetry(size) {
        //     println!("Validation: Matrices are symmetric.");
        // } else {
        //     println!("Validation WARNING: Matrices are NOT symmetric.");
        // }

        let mut heuristic_solution = [0i32; utils::MAX_SIZE]; //defult with 0s
        print!("\nSolution before heuristic: {:?}", heuristic_solution);
        utils::heuristic(size, &mut heuristic_solution[0..size]);
        print!("\n\nSolution after heuristic: {:?}", heuristic_solution);
        break;
    }
    Ok(())
}