use crate::utils;
use rand::Rng;
// use std::time::Instant;

pub struct RunResult {
    pub best_cost: i64,
    pub best_solution: [i32; crate::utils::MAX_SIZE],
    pub steps: u64,
    pub evaluations: u64,
    pub time_micros: u128,
}

pub fn simulated_annealing(
    size: usize,
    current_solution: &mut [i32],
    mut current_cost: i64,
    distances: &[[i32; utils::MAX_SIZE]; utils::MAX_SIZE],
    flows: &[[i32; utils::MAX_SIZE]; utils::MAX_SIZE],
    cooling_rate: f64,
    l_factor: usize,
    p: usize,
    mut temperature: f64,
    end_temperature: f64
) -> RunResult {

    //Init solution parameters
    let start_time = std::time::Instant::now();
    let mut best_cost = current_cost;
    let mut evaluations: u64 = 0;
    let mut steps: u64 = 0;
    let mut best_solution = [0i32; crate::utils::MAX_SIZE];
    best_solution[..size].copy_from_slice(&current_solution[..size]);
    let mut rng = rand::thread_rng();

    //Init search parameters
    let l = l_factor * (size * (size - 1)/2);
    let mut no_improvement = 0;

    while (temperature > end_temperature) || (no_improvement < p*l){
        for _ in 0..l{
            let (i, j) = utils::generate_unique_pairs(size);
            let delta = utils::calculate_delta(size, current_solution, distances, flows, i, j);
            evaluations += 1;

            if (delta < 0) || (rng.gen_range(0.0..1.0) < (-delta as f64 / temperature).exp()) {
                current_solution.swap(i, j);
                current_cost += delta;
                steps += 1;
                
            if current_cost < best_cost {
                    best_cost = current_cost;
                    best_solution[..size].copy_from_slice(&current_solution[..size]);
                    no_improvement = 0;
                } else {no_improvement += 1;}
            } else{no_improvement += 1;}
        }
        temperature *= cooling_rate;
    }

    RunResult{
        best_cost: best_cost,
        best_solution: best_solution,
        steps,
        evaluations,
        time_micros: start_time.elapsed().as_micros(),
    }
}