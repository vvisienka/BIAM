use std::thread::current;

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

pub fn tabu_search(
    size: usize,
    current_solution: &mut [i32],
    mut current_cost: i64,
    distances: &[[i32; utils::MAX_SIZE]; utils::MAX_SIZE],
    flows: &[[i32; utils::MAX_SIZE]; utils::MAX_SIZE],
    p: usize,
    tabu_tenure: usize,
    candidate_list_size: usize,
    elite_size: usize,
) -> RunResult{

    //Init solution parameters
    let start_time = std::time::Instant::now();
    let mut best_cost = current_cost;
    let mut tabu_matrix = [[0i32; utils::MAX_SIZE]; utils::MAX_SIZE];
    let mut best_solution = [0i32; crate::utils::MAX_SIZE];
    best_solution[..size].copy_from_slice(&current_solution[..size]);

    //Init search parameters
    let l = size * (size - 1) / 2;
    let mut no_improvement = 0;
    let mut steps = 0;
    let mut evaluations = 0;
    let mut iteration = 0;

    while no_improvement < p*l {
        iteration+=1;
        let (candidate_list, evals_made) = utils::get_elite_candidates(size, candidate_list_size, elite_size, &current_solution, &distances, &flows);
        evaluations += evals_made;

        let mut best_move = None;
        let mut best_delta = i64::MAX;

        //find the best move
        for (i,j,delta) in candidate_list{
            let is_tabu = tabu_matrix[i][j] > iteration as i32;
            let satisfies_aspiration = current_cost + delta < best_cost;

            if (!is_tabu || satisfies_aspiration) && delta < best_delta {
                best_move = Some((i, j));
                best_delta = delta;
            }
        }

        //execute the best move
        if let Some((i, j)) = best_move {
            current_solution.swap(i, j);
            current_cost += best_delta;
            steps += 1;

            //update tabu list
            tabu_matrix[i][j] = iteration as i32 + tabu_tenure as i32;
            tabu_matrix[j][i] = iteration as i32 + tabu_tenure as i32;

            if current_cost < best_cost {
                best_cost = current_cost;
                best_solution[..size].copy_from_slice(&current_solution[..size]);
                no_improvement = 0;
            } else { no_improvement += 1; }
        } else {no_improvement += 1; }
    }


    RunResult{
        best_cost: best_cost,
        best_solution: best_solution,
        steps,
        evaluations,
        time_micros: start_time.elapsed().as_micros(),
    }

}