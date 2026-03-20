#[allow(dead_code)]
use rand::Rng;
use std::time::Instant;

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
pub fn measure_time<F>(mut f: F, end_time: u128, min_iterations: u32)
where
    F: FnMut(),
{
    let start_time = Instant::now();
    let mut counter = 0;
    loop {
        f();
        counter+=1;

        if start_time.elapsed().as_millis() >= end_time && counter >= min_iterations{
            println!("Counter: {counter}, Time: {}", start_time.elapsed().as_millis());
            break;
        }
    }
}

//Loading instance data
pub fn load_data() {}

//Heuristic
pub fn heuristic() {}

//2-OPT neighborhood
pub fn neighborhood() {}