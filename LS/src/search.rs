use crate::utils;
// use std::time::Instant;

pub struct RunResult {
    pub best_cost: i64,
    pub best_solution: [i32; utils::MAX_SIZE],
    pub steps: u64,
    pub evaluations: u64,
    pub time_micros: u128,
}

pub fn random_search() {
    
}
// pub fn random_walk() {}
// pub fn local_search() {}
// pub fn steepest() {}
// pub fn greedy(){}