use crate::utils;
use rand::Rng;
// use std::time::Instant;

pub struct RunResult {
    pub best_cost: i64,
    pub best_solution: [i32; utils::MAX_SIZE],
    pub steps: u64,
    pub evaluations: u64,
    pub time_micros: u128,
}

//Heuristic - average value for each row and column, match the highest average (from B) with lowest average (from A)
pub fn heuristic(
    size: usize,
    solution: &mut [i32],
    distances: &[[i32; utils::MAX_SIZE]; utils::MAX_SIZE],
    flows: &[[i32; utils::MAX_SIZE]; utils::MAX_SIZE],
) {
    let mut dist_potentials: [i32; utils::MAX_SIZE] = [0; utils::MAX_SIZE];
    let mut flow_potentials: [i32; utils::MAX_SIZE] = [0; utils::MAX_SIZE];

    for i in 0..size{
        let mut dist_sum = 0;
        let mut flow_sum = 0;
        for j in 0..size{
            dist_sum += distances[i][j];
            flow_sum += flows[i][j];
        }
        dist_potentials[i] = dist_sum;
        flow_potentials[i] = flow_sum;
    }

    // Create arrays of indices: [0, 1, 2, ..., size-1] to sort them
    let mut dist_indices: [usize; utils::MAX_SIZE] = [0; utils::MAX_SIZE];
    let mut flow_indices: [usize; utils::MAX_SIZE] = [0; utils::MAX_SIZE];
    for i in 0..size {
        dist_indices[i] = i;
        flow_indices[i] = i;
    }

    // Sort the indices based on the potential values - DIST ascending, FLOW descending
    dist_indices[0..size].sort_unstable_by(|&a, &b| dist_potentials[a].cmp(&dist_potentials[b]));
    flow_indices[0..size].sort_unstable_by(|&a, &b| flow_potentials[b].cmp(&flow_potentials[a]));

    //Randomness
    let mut rng = rand::thread_rng();
    for i in 0..size - 1 {
        if rng.gen_bool(0.20) { 
            flow_indices.swap(i, i + 1);
        }
    }

    // Highest flow is assigned to the location with the lowest distance.
    for i in 0..size {
        solution[dist_indices[i]] = (flow_indices[i]+1) as i32;
    }
}

pub fn local_search(size: usize, 
    current_solution: &mut [i32],
    mut current_cost: i64,
    distances: &[[i32; utils::MAX_SIZE]; utils::MAX_SIZE],
    flows: &[[i32; utils::MAX_SIZE]; utils::MAX_SIZE],
    is_greedy: bool) -> RunResult{
        let start_time = std::time::Instant::now();
        let mut steps: u64 = 0;
        let mut evaluations: u64 = 0;
        let mut improvement = true;
        let mut rng = rand::thread_rng();

        while improvement {
            improvement = false;

            let mut best_delta: i64 = 0;
            let mut best_pair: Option<(usize, usize)> = None;
            
            //greedy randomnization
            let start_i = if is_greedy { rng.gen_range(0..size - 1) } else { 0 };

            'outer: for step_i in 0..size-1 {
                let i = (start_i + step_i) % (size - 1);
                let num_j = size - 1 - i;
                let start_j_offset = if is_greedy && num_j > 1 { rng.gen_range(0..num_j) } else { 0 };

                for step_j in 0..num_j {
                    let j_offset = (start_j_offset + step_j) % num_j;
                    let j = (i + 1) + j_offset;

                    let delta = utils::calculate_delta(size, &current_solution, distances, flows, i, j);
                    evaluations += 1;
                    
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
        let mut best_solution_copy = [0i32; crate::utils::MAX_SIZE];
        best_solution_copy[..size].copy_from_slice(&current_solution[..size]);

        RunResult {
            best_cost: current_cost,
            best_solution: best_solution_copy,
            steps,
            evaluations,
            time_micros: start_time.elapsed().as_micros(),
        }
}

pub fn random_search(
    size: usize,
    current_solution: &mut [i32],
    current_cost: i64,
    time_limit_micros: u128,
    distances: &[[i32; utils::MAX_SIZE]; utils::MAX_SIZE],
    flows: &[[i32; utils::MAX_SIZE]; utils::MAX_SIZE],
) -> RunResult {
    let start_time = std::time::Instant::now();
    let mut best_cost = current_cost;
    let mut evaluations: u64 = 0;

    let mut best_solution = [0i32; crate::utils::MAX_SIZE];
    best_solution[..size].copy_from_slice(&current_solution[..size]);

    const TIME_CHECK_INTERVAL: u64 = 350; // Check time periodically
    let mut iter_count: u64 = 0;

    //Time limit
    loop {
        utils::generate_permutations(current_solution);
        let cost = utils::evaluate(size, current_solution, distances, flows);
        evaluations += 1;

        if cost < best_cost {
            best_cost = cost;
            best_solution[..size].copy_from_slice(&current_solution[..size]);
        }

        iter_count += 1;
        if (iter_count & (TIME_CHECK_INTERVAL - 1)) == 0 {
            if start_time.elapsed().as_micros() >= time_limit_micros {
                break;
            }
        }
    }

    RunResult {
        best_cost,
        best_solution,
        steps: evaluations,
        evaluations,
        time_micros: start_time.elapsed().as_micros(),
    }
}

pub fn random_walk(
    size: usize,
    current_solution: &mut [i32],
    mut current_cost: i64,
    time_limit_micros: u128,
    distances: &[[i32; utils::MAX_SIZE]; utils::MAX_SIZE],
    flows: &[[i32; utils::MAX_SIZE]; utils::MAX_SIZE],
) -> RunResult {
    let start_time = std::time::Instant::now();
    let mut best_cost = current_cost;
    let mut evaluations: u64 = 0;

    let mut best_solution = [0i32; crate::utils::MAX_SIZE];
    best_solution[..size].copy_from_slice(&current_solution[..size]);

    const TIME_CHECK_INTERVAL: u64 = 1300;
    let mut iter_count: u64 = 0;

    //Time limit
    loop {
        let (i, j) = utils::generate_unique_pairs(size);
        let delta = utils::calculate_delta(size, current_solution, distances, flows, i, j);
        evaluations += 1;

        current_solution.swap(i, j);
        current_cost += delta;
        

        if current_cost < best_cost {
            best_cost = current_cost;
            best_solution[..size].copy_from_slice(&current_solution[..size]);
        }

        iter_count += 1;
        if (iter_count & (TIME_CHECK_INTERVAL - 1)) == 0 {
            if start_time.elapsed().as_micros() >= time_limit_micros {
                break;
            }
        }
    }

    RunResult {
        best_cost,
        best_solution,
        steps: evaluations,
        evaluations,
        time_micros: start_time.elapsed().as_micros(),
    }
}