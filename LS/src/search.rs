use crate::utils;
// use std::time::Instant;

pub struct RunResult {
    pub best_cost: i64,
    pub best_solution: [i32; utils::MAX_SIZE],
    pub steps: u64,
    pub evaluations: u64,
    pub time_micros: u128,
}

pub fn local_search(size: usize, 
    mut current_solution: [i32; 64],
    mut current_cost: i64,
    distances: &[[i32; 64]; 64],
    flows: &[[i32; 64]; 64],
    is_greedy: bool) -> RunResult{
        let start_time = std::time::Instant::now();
        let mut steps: u64 = 0;
        let mut evaluations: u64 = 0;
        let mut improvement = true;

        while improvement {
            improvement = false;
            //add randomization for greedy search

            let mut best_delta: i64 = 0;
            let mut best_pair: Option<(usize, usize)> = None;

            'outer: for i in 0..size-1 {
                for j in i+1..size {
                    evaluations += 1;
                    let delta = utils::calculate_delta(size, &current_solution, distances, flows, i, j);
                    if delta < 0 {
                        if is_greedy {
                            current_solution.swap(i, j);
                            current_cost += delta;
                            steps += 1;
                            improvement = true;
                            break 'outer; // break both loops to start search from the beginning
                        } else {
                            if delta < best_delta {
                                best_delta = delta;
                                best_pair = Some((i, j)); //Store the best pair
                            }
                        }
                    }
                }
            }
            if !is_greedy {
                if let Some((i, j)) = best_pair {
                    current_solution.swap(i, j);
                    current_cost += best_delta;
                    steps += 1;
                    improvement = true;
                }
            }

        }
        RunResult {
            best_cost: current_cost,
                best_solution: current_solution,
            steps,
            evaluations,
            time_micros: start_time.elapsed().as_micros(),
        }
}

pub fn random_search(
    size: usize,
    time_limit_micros: u128,
    distances: &[[i32; utils::MAX_SIZE]; utils::MAX_SIZE],
    flows: &[[i32; utils::MAX_SIZE]; utils::MAX_SIZE],
) -> RunResult {
    let start_time = std::time::Instant::now();
    let mut best_cost = i64::MAX;
    let mut best_solution = [0i32; utils::MAX_SIZE];
    let mut evaluations: u64 = 0;
    let mut steps: u64 = 0;

    //1-based
    let mut current_solution = [0i32; utils::MAX_SIZE];
    for i in 0..size {
        current_solution[i] = (i + 1) as i32;
    }

    //Time limit
    while start_time.elapsed().as_micros() < time_limit_micros {
        utils::generate_permutations(&mut current_solution[0..size]);
        let cost = utils::evaluate(size, &current_solution, distances, flows);
        evaluations += 1;

        if cost < best_cost {
            best_cost = cost;
            best_solution = current_solution;
            steps += 1;
        }
    }

    RunResult {
        best_cost,
        best_solution,
        steps,
        evaluations,
        time_micros: start_time.elapsed().as_micros(),
    }
}

pub fn random_walk(
    size: usize,
    mut current_solution: [i32; utils::MAX_SIZE],
    mut current_cost: i64,
    time_limit_micros: u128,
    distances: &[[i32; utils::MAX_SIZE]; utils::MAX_SIZE],
    flows: &[[i32; utils::MAX_SIZE]; utils::MAX_SIZE],
) -> RunResult {
    let start_time = std::time::Instant::now();
    let mut best_cost = current_cost;
    let mut best_solution = current_solution;
    let mut evaluations: u64 = 0;
    let mut steps: u64 = 0;

    //Time limit
    while start_time.elapsed().as_micros() < time_limit_micros {
        let (i, j) = utils::generate_unique_pairs(size);
        
        let delta = utils::calculate_delta(size, &current_solution, distances, flows, i, j);
        evaluations += 1;

        current_solution.swap(i, j);
        current_cost += delta;
        steps += 1;

        if current_cost < best_cost {
            best_cost = current_cost;
            best_solution = current_solution;
        }
    }

    RunResult {
        best_cost,
        best_solution,
        steps,
        evaluations,
        time_micros: start_time.elapsed().as_micros(),
    }
}