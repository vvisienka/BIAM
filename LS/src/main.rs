mod utils;
mod search;

use std::fs;
use std::error::Error;
use std::io::Write;

fn main() -> Result<(), Box<dyn Error>> {
    println!("=== Multi Start Local Search for QAP");

    // === LOAD DATA ===
    let mut instances: [String; 10] = Default::default(); //intialized with empty strings
    let mut instance_count = 0;

    //1. Collect the file names
    for entry in fs::read_dir("../data")?{
        
        let path = entry?.path();
        if let Some(path_str) = path.to_str(){

        if path_str.contains(".dat") && !path_str.contains("_solution.dat"){
            instances[instance_count] = path_str.to_string();
            instance_count += 1;
        }
    }}

    // 2. Prepare CSV output files
    let mut csv_file = fs::File::create("results.csv")?;
    writeln!(csv_file, "Instance, OptCost, Algorithm, Run, BestCost, Steps, Evaluations, TimeMicros")?;

    // === RUN EXPERIMENTS ===
    for i in 0..instance_count{

        // 1. Load instance data
        let file_path = &instances[i];
        // An empty file path means we've reached the end of the found files.
        if file_path.is_empty() { break; }

        let mut distances: [[i32; utils::MAX_SIZE]; utils::MAX_SIZE] = [[0; utils::MAX_SIZE]; utils::MAX_SIZE];
        let mut flows: [[i32; utils::MAX_SIZE]; utils::MAX_SIZE] = [[0; utils::MAX_SIZE]; utils::MAX_SIZE];
        
        println!("\n--- Processing instance: {} ---", file_path);
        let size = utils::load_data(file_path, &mut distances, &mut flows)?;
        let instance_name = file_path.split('/').last().unwrap_or("unknown_instance");
        let opt_cost = utils::get_optimal_cost(file_path);

        // 2. Run the algorithms
        for run in 1..=10{

            // initial solution
            let mut init_solution = [0i32; utils::MAX_SIZE];
            for k in 0..size { init_solution[k] = (k + 1) as i32; }
            utils::generate_permutations(&mut init_solution[0..size]);
            // println!("Init {:?}", init_solution);
            let init_cost = utils::evaluate(size, &init_solution[0..size], &distances, &flows);
            
            //heuristic
            let mut h_sol = init_solution;
            let h_start = std::time::Instant::now();
            search::heuristic(size, &mut h_sol[0..size], &distances, &flows);
            let h_cost = utils::evaluate(size, &h_sol[0..size], &distances, &flows);
            let h_time = h_start.elapsed().as_micros();
            writeln!(csv_file, "{},{},Heuristic,{},{},0,0,{}", instance_name, opt_cost, run, h_cost, h_time)?;

            //greedy
            let mut g_sol = init_solution;
            let res_g = search::local_search(size, &mut g_sol[0..size], init_cost, &distances, &flows, true);
            writeln!(csv_file, "{},{},Greedy,{},{},{},{},{}", instance_name, opt_cost, run, res_g.best_cost, res_g.steps, res_g.evaluations, res_g.time_micros)?;

            //steepest
            let mut s_sol = init_solution;
            let res_s = search::local_search(size, &mut s_sol[0..size], init_cost, &distances, &flows, false);
            writeln!(csv_file, "{},{},Steepest,{},{},{},{},{}", instance_name, opt_cost, run, res_s.best_cost, res_s.steps, res_s.evaluations, res_s.time_micros)?;

            let avg_ls_time = (res_s.time_micros + res_g.time_micros) / 2;
            
            //random search
            let mut rs_sol = init_solution;
            let res_rs = search::random_search(size, &mut rs_sol[0..size], init_cost, avg_ls_time, &distances, &flows);
            writeln!(csv_file, "{},{},RandomSearch,{},{},{},{},{}", instance_name, opt_cost, run, res_rs.best_cost, res_rs.steps, res_rs.evaluations, res_rs.time_micros)?;
            
            //random walk
            let mut rw_sol = init_solution;
            let res_rw = search::random_walk(size, &mut rw_sol[0..size], init_cost, avg_ls_time, &distances, &flows);
            writeln!(csv_file, "{},{},RandomWalk,{},{},{},{},{}", instance_name, opt_cost, run, res_rw.best_cost, res_rw.steps, res_rw.evaluations, res_rw.time_micros)?;
        }

    }
    println!("All experiments completed. Results saved to results.csv");
    Ok(())
}