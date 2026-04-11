mod utils;
mod search;

use std::fs;
use std::error::Error;
use std::io::Write;

fn main() -> Result<(), Box<dyn Error>> {
    println!("=== Analiza struktury przestrzeni poszukiwań (450 powtórzeń x 5 algorytmów) ===");

    let target_instances = [
        "els19.dat", "chr20c.dat", "chr20a.dat", "esc32a.dat"];
    let runs = 450; 

    // Plik ląduje poziom wyżej, żeby Python miał do niego łatwy dostęp
    let mut csv_file = fs::File::create("scatter_data.csv")?;
    
    // Dodajemy kolumnę "Solution" na samym końcu, żeby móc liczyć Similarity!
    writeln!(csv_file, "Instance, OptCost, Algorithm, Run, InitCost, FinalCost, Solution")?;

    for instance_file in target_instances.iter() {
        let file_path = format!("../data/{}", instance_file);
        let instance_name = instance_file.replace(".dat", "");
        
        let mut distances: [[i32; utils::MAX_SIZE]; utils::MAX_SIZE] = [[0; utils::MAX_SIZE]; utils::MAX_SIZE];
        let mut flows: [[i32; utils::MAX_SIZE]; utils::MAX_SIZE] = [[0; utils::MAX_SIZE]; utils::MAX_SIZE];
        
        println!("\n--- Przetwarzanie instancji: {} ({} przebiegów) ---", instance_name, runs);
        
        let size = match utils::load_data(&file_path, &mut distances, &mut flows) {
            Ok(s) => s,
            Err(_) => {
                println!("⚠️ Nie można załadować pliku {}, pomijam...", file_path);
                continue;
            }
        };
        
        let opt_cost = utils::get_optimal_cost(&file_path);

        for run in 1..=runs {
            if run % 50 == 0 {
                println!("  ... Przebieg {}/{}", run, runs);
            }

            // Jeden wspólny, losowy punkt startowy dla wszystkich algorytmów w danym przebiegu
            let mut init_solution = [0i32; utils::MAX_SIZE];
            for k in 0..size { init_solution[k] = (k + 1) as i32; }
            utils::generate_permutations(&mut init_solution[0..size]);
            let init_cost = utils::evaluate(size, &init_solution[0..size], &distances, &flows);

            // Funkcja pomocnicza do zamiany tablicy na tekst oddzielony spacjami
            let format_sol = |sol: &[i32]| -> String {
                sol[0..size].iter().map(|val| val.to_string()).collect::<Vec<String>>().join(" ")
            };

            // 1. Heuristic (H)
            let mut h_sol = init_solution;
            search::heuristic(size, &mut h_sol[0..size], &distances, &flows);
            let h_cost = utils::evaluate(size, &h_sol[0..size], &distances, &flows);
            writeln!(csv_file, "{},{},H,{},{},{},{}", 
                instance_name, opt_cost, run, init_cost, h_cost, format_sol(&h_sol))?;

            // 2. Greedy (G)
            let mut g_sol = init_solution;
            let res_g = search::local_search(size, &mut g_sol[0..size], init_cost, &distances, &flows, true);
            writeln!(csv_file, "{},{},G,{},{},{},{}", 
                instance_name, opt_cost, run, init_cost, res_g.best_cost, format_sol(&res_g.best_solution))?;

            // 3. Steepest (S)
            let mut s_sol = init_solution;
            let res_s = search::local_search(size, &mut s_sol[0..size], init_cost, &distances, &flows, false);
            writeln!(csv_file, "{},{},S,{},{},{},{}", 
                instance_name, opt_cost, run, init_cost, res_s.best_cost, format_sol(&res_s.best_solution))?;

            // Limit czasu dla metod losowych
            let avg_ls_time = (res_s.time_micros + res_g.time_micros) / 2;

            // 4. Random Search (RS)
            let mut rs_sol = init_solution;
            let res_rs = search::random_search(size, &mut rs_sol[0..size], init_cost, avg_ls_time, &distances, &flows);
            writeln!(csv_file, "{},{},RS,{},{},{},{}", 
                instance_name, opt_cost, run, init_cost, res_rs.best_cost, format_sol(&res_rs.best_solution))?;

            // 5. Random Walk (RW)
            let mut rw_sol = init_solution;
            let res_rw = search::random_walk(size, &mut rw_sol[0..size], init_cost, avg_ls_time, &distances, &flows);
            writeln!(csv_file, "{},{},RW,{},{},{},{}", 
                instance_name, opt_cost, run, init_cost, res_rw.best_cost, format_sol(&res_rw.best_solution))?;
        }
    }

    drop(csv_file);
    println!("\n✅ Obliczenia Rusta zakończone. Możesz teraz odpalić skrypty w Pythonie!");

    Ok(())
}