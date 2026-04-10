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
            flows[i][j] = val_str.parse::<i32>()?;
        }
    }

    for k in 0..size{
        for l in 0..size{
            let val_str = tokens.next().ok_or("Error while updating FLOWS")?;
            distances[k][l] = val_str.parse::<i32>()?;
        }
    }

    println!("Loading finished");
    Ok(size)
}

//Calculate delta
pub fn calculate_delta(size: usize, solution: &[i32], distances: &[[i32; MAX_SIZE]; MAX_SIZE], flows: &[[i32; MAX_SIZE]; MAX_SIZE], a: usize, b: usize) -> i64 {
    let mut delta: i64 = 0;
    let loc_a = (solution[a]-1) as usize;
    let loc_b = (solution[b]-1) as usize;
    for k in 0..size {
        if k != a && k != b {
            let loc_k = (solution[k]-1) as usize;
            let flow_diff = flows[a][k] - flows[b][k];
            let dist_diff = distances[loc_b][loc_k] - distances[loc_a][loc_k];
            delta += (flow_diff as i64) * (dist_diff as i64);
        }
    }
    delta*2
}

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

pub fn evaluate(
    size: usize,
    solution: &[i32],
    distances: &[[i32; MAX_SIZE]; MAX_SIZE],
    flows: &[[i32; MAX_SIZE]; MAX_SIZE],
) -> i64 {
    let mut total_cost: i64 = 0;
    for i in 0..size {
        for j in i+1..size {
            let location_i = (solution[i]-1) as usize;
            let location_j = (solution[j]-1) as usize;
            total_cost += (distances[location_i][location_j] as i64) * (flows[i][j] as i64);

        }
    }
    total_cost*2 //to get the full cost
}

pub fn get_optimal_cost(file_path: &str) -> i64 {
    // Zamieniamy .dat na _solution.dat
    let sol_path = file_path.replace(".dat", "_solution.dat");
    if let Ok(content) = fs::read_to_string(&sol_path) {
        let mut tokens = content.split_whitespace();
        tokens.next(); // Pomijamy pierwszą liczbę (rozmiar N)
        if let Some(opt_str) = tokens.next() {
            return opt_str.parse::<i64>().unwrap_or(0);
        }
    }
    // Jeśli nie ma pliku lub jest błąd, zwracamy 0
    0 
}