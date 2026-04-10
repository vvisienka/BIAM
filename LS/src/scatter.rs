mod utils;
mod search;

use std::fs;
use std::error::Error;
use std::io::Write;
use std::process::Command;

fn main() -> Result<(), Box<dyn Error>> {
    println!("=== Search Space Structure Analysis (450 Runs) ===");

    // The three interesting instances
    let target_instances = ["nug12.dat", "tai17a.dat", "els19.dat", "chr20a.dat", "chr20c.dat", "nug20.dat", "nug30.dat", "esc32a.dat", "tai50a.dat", "esc64a.dat"];
    let runs = 450;

    let mut csv_file = fs::File::create("../scatter_data.csv")?;
    writeln!(csv_file, "Instance, OptCost, Algorithm, Run, InitCost, FinalCost")?;

    for instance_file in target_instances.iter() {
        let file_path = format!("../../data/{}", instance_file);
        let instance_name = instance_file.replace(".dat", "");
        
        let mut distances: [[i32; utils::MAX_SIZE]; utils::MAX_SIZE] = [[0; utils::MAX_SIZE]; utils::MAX_SIZE];
        let mut flows: [[i32; utils::MAX_SIZE]; utils::MAX_SIZE] = [[0; utils::MAX_SIZE]; utils::MAX_SIZE];
        
        println!("\n--- Processing instance: {} ({} runs) ---", instance_name, runs);
        
        let size = match utils::load_data(&file_path, &mut distances, &mut flows) {
            Ok(s) => s,
            Err(_) => {
                println!("Could not load {}, skipping...", file_path);
                continue;
            }
        };
        
        let opt_cost = utils::get_optimal_cost(&file_path);

        for run in 1..=runs {
            if run % 50 == 0 {
                println!("  ... Run {}/{}", run, runs);
            }

            // Generate ONE shared starting point for both algorithms
            let mut init_solution = [0i32; utils::MAX_SIZE];
            for k in 0..size { init_solution[k] = (k + 1) as i32; }
            utils::generate_permutations(&mut init_solution[0..size]);
            let init_cost = utils::evaluate(size, &init_solution[0..size], &distances, &flows);

            // Run Greedy
            let mut sol_g = init_solution;
            let res_g = search::local_search(size, &mut sol_g[0..size], init_cost, &distances, &flows, true);
            writeln!(csv_file, "{},{},G,{},{},{}", instance_name, opt_cost, run, init_cost, res_g.best_cost)?;

            // Run Steepest
            let mut sol_s = init_solution;
            let res_s = search::local_search(size, &mut sol_s[0..size], init_cost, &distances, &flows, false);
            writeln!(csv_file, "{},{},S,{},{},{}", instance_name, opt_cost, run, init_cost, res_s.best_cost)?;
        }
    }

    println!("\nRust calculations complete. Launching Python plotter...");

    // Automatically call Python to draw the plots!
    let status = Command::new("python")
        .arg("plot_scatter.py")
        .status()?;

    if status.success() {
        println!("All scatter plots generated successfully!");
    } else {
        println!("Python script failed to run. Check if scipy and seaborn are installed.");
    }

    Ok(())
}