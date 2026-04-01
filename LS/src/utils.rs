#[allow(dead_code)]
use rand::Rng;
use std::time::Instant;
use std::fs;
use std::error::Error;

pub const MAX_SIZE: usize = 64;

//Generating random permutations
pub fn generate_permutations(arr: &mut [i32]) {
    let n = arr.len();
    let mut rng = rand::thread_rng();
    if n <= 1 {return;}

    let mut index = n - 1;
    while index > 0{
        let random_index = rng.gen_range(0..=index); //inclusive on both sides
        arr.swap(index, random_index);
        index -= 1;
    }
    
}

//Generating pairs of random but unique numbers 0..n–1 (for RW and SA)
pub fn generate_unique_pairs(n: usize) -> (usize, usize){
    let mut rng = rand::thread_rng();
    let x = rng.gen_range(0..n);
    let y = (x + rng.gen_range(0..n-1)+1) % n;
    (x, y)
}

//Measuring algorithm running time
pub fn measure_time<F>(mut f: F, end_time_micros: u128, min_iterations: u32)
where
    F: FnMut(),
{
    let start_time = Instant::now();
    let mut counter = 0;
    loop {
        f();
        counter+=1;

        if counter >= min_iterations {
            let elapsed = start_time.elapsed();
            if elapsed.as_micros() >= end_time_micros {
                let total_nanos = elapsed.as_nanos() as f64;
                let avg_ns = total_nanos / counter as f64;
                
                println!("--- Time Results ---");
                println!("Total Iterations: {}", counter);
                println!("Total Time:       {} µs", elapsed.as_micros());
                println!("Avg Runtime:      {:.2} µs", avg_ns / 1000.0);
                println!("Avg Runtime:      {:.1} ns", avg_ns);
                break;
            }
        }
        // if start_time.elapsed().as_millis() >= end_time && counter >= min_iterations{
        //     println!("Counter: {counter}, Time: {}", start_time.elapsed().as_millis());
        //     break;
        // }
    }
    //RT = delta t/C ??
}

//Loading instance data
pub fn load_data(
    file_path: &str,
    distances: &mut [[i32; MAX_SIZE]; MAX_SIZE],
    flows: &mut [[i32; MAX_SIZE]; MAX_SIZE],
) -> Result<usize, Box<dyn Error>> {
    let content: String = fs::read_to_string(file_path)?; //? means following if there is no error, returning it if there is
    let mut tokens = content.split_whitespace();
    let size: usize = tokens.next()
                        .ok_or("File is empty")? 
                        .parse::<usize>()?;
    if size > MAX_SIZE{
        return Err("Instance size exceeds MAX_SIZE buffer".into());
    }

    //update matrices
    for i in 0..size{
        for j in 0..size{
            let val_str = tokens.next().ok_or("Error while updating DISTANCES")?;
            distances[i][j] = val_str.parse::<i32>()?;
        }
    }

    for k in 0..size{
        for l in 0..size{
            let val_str = tokens.next().ok_or("Error while updating FLOWS")?;
            flows[k][l] = val_str.parse::<i32>()?;
        }
    }

    println!("Loading finished");
    Ok(size)
}

//Heuristic - average value for each row and column, match the highest average (from B) with lowest average (from A)
pub fn heuristic(
    size: usize,
    solution: &mut [i32],
    distances: &[[i32; MAX_SIZE]; MAX_SIZE],
    flows: &[[i32; MAX_SIZE]; MAX_SIZE],
) {
    let mut dist_potentials: [i32; MAX_SIZE] = [0; MAX_SIZE];
    let mut flow_potentials: [i32; MAX_SIZE] = [0; MAX_SIZE];

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
    let mut dist_indices: [usize; MAX_SIZE] = [0; MAX_SIZE];
    let mut flow_indices: [usize; MAX_SIZE] = [0; MAX_SIZE];
    for i in 0..size {
        dist_indices[i] = i;
        flow_indices[i] = i;
    }

    // Sort the indices based on the potential values - DIST ascending, FLOW descending
    dist_indices[0..size].sort_unstable_by(|&a, &b| dist_potentials[a].cmp(&dist_potentials[b]));
    flow_indices[0..size].sort_unstable_by(|&a, &b| flow_potentials[b].cmp(&flow_potentials[a]));

    // Highest flow is assigned to the location with the lowest distance.
    for i in 0..size {
        solution[dist_indices[i]] = flow_indices[i] as i32;
    }
}

//2-OPT neighborhood
// pub fn neighborhood() {}

pub fn test_symmetry(
    size: usize,
    distances: &[[i32; MAX_SIZE]; MAX_SIZE],
    flows: &[[i32; MAX_SIZE]; MAX_SIZE],
) -> bool {
    // We only need to iterate through the upper triangle of the matrix.
    for i in 0..size {
        for j in (i + 1)..size {
            if distances[i][j] != distances[j][i] || flows[i][j] != flows[j][i] {
                return false;
            }
        }
    }
    // If we've checked all pairs without returning, they are symmetric.
    true
}

// pub fn evaluate(){}
// pub fn random_search() {}
// pub fn random_walk() {}
// pub fn local_search() {}
// pub fn steepest() {}
// pub fn greedy(){}