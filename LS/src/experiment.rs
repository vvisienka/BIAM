mod utils;
mod search;
mod search_extended;

use std::fs;
use std::error::Error;
use std::io::Write;

fn main() -> Result<(), Box<dyn Error>> {
    println!("=== Parameters Experiments (SA & TS) ===");

    let target_instances = vec!["chr20c.dat", "esc32a.dat"];
    let mut instances = Vec::new();

    for entry in fs::read_dir("../data")? {
        let path = entry?.path();
        if let Some(filename) = path.file_name().and_then(|s| s.to_str()) {
            if target_instances.contains(&filename) {
                instances.push(path.to_str().unwrap().to_string());
            }
        }
    }

    let mut exp_file = fs::File::create("experiment_data.csv")?;
    writeln!(exp_file, "Instance,Algorithm,P1_Name,P1_Val,P2_Name,P2_Val,BestCost,Evaluations,TimeMicros")?;

    for file_path in &instances {
        let mut distances = [[0; utils::MAX_SIZE]; utils::MAX_SIZE];
        let mut flows = [[0; utils::MAX_SIZE]; utils::MAX_SIZE];
        
        let size = utils::load_data(file_path, &mut distances, &mut flows)?;
        let instance_name = file_path.split('/').last().unwrap_or("unknown");

        println!("\n--- Processing specific instance: {} ---", instance_name);

        // Adaptive temperature setup (95% initial acceptance)
        let mut t_arr = [0i32; utils::MAX_SIZE];
        for k in 0..size { t_arr[k] = (k + 1) as i32; }
        let (t_start, t_end) = utils::calculate_adaptive_temperatures(size, &t_arr[..size], &distances, &flows);

        for _ in 1..=20 {
            // Generate a unique initial solution for every run to ensure variety
            let mut init_solution = [0i32; utils::MAX_SIZE];
            for k in 0..size { init_solution[k] = (k + 1) as i32; }
            utils::generate_permutations(&mut init_solution[0..size]);
            let init_cost = utils::evaluate(size, &init_solution[0..size], &distances, &flows);
            
            // --- Simulated Annealing: Alpha vs L_Factor ---
            // Tuning alpha (cooling rate) and l_fact (Markov chain length multiplier)
            for alpha in [0.8, 0.85, 0.9, 0.95, 0.99] {
                for l_fact in [2, 5, 8] {
                    let mut sa_sol = init_solution;
                    let res = search_extended::simulated_annealing(
                        size, &mut sa_sol, init_cost, &distances, &flows, 
                        alpha, l_fact, 10, t_start, t_end
                    );
                    writeln!(exp_file, "{},SA,Alpha,{},L_Factor,{},{},{},{}", 
                        instance_name, alpha, l_fact, res.best_cost, res.evaluations, res.time_micros)?;
                }
            }

            // --- Tabu Search: P (Stagnation) vs Tenure_Factor ---
            // Tuning stagnation limit (P) and Tabu Tenure (size / factor)
            for p_test in [1, 2, 5, 10, 20] {
                for tenure_fact in [10, 4, 2] { 
                    let tenure = size / tenure_fact;
                    let mut ts_sol = init_solution;
                    let l = size * (size - 1) / 2;
                    // Uses Elite Candidate List (20% of N, top 20% of V)
                    let res = search_extended::tabu_search(
                        size, &mut ts_sol, init_cost, &distances, &flows, 
                        p_test, tenure, l/5, l/25);
                    writeln!(exp_file, "{},TS,P_Factor,{},Tenure_Factor,{},{},{},{}", 
                        instance_name, p_test, tenure_fact, res.best_cost, res.evaluations, res.time_micros)?;
                }
            }
        }
    }

    println!("✅ Experiments finished for chr20c and esc32a! Results saved in experiment_data.csv");
    Ok(())
}