use std::sync::{Arc, atomic::{AtomicBool, AtomicUsize, Ordering}};
use ndarray::{Array1, Array2, Axis};
use moors::{
    AlgorithmBuilder, CloseDuplicatesCleaner, GaussianMutation, RandomSamplingFloat,
    SimulatedBinaryCrossover,
    selection::soo::RankSelection,
    survival::soo::FitnessSurvival,
    IterationData,
    NoConstraints,
};

fn fitness_sphere(population: &Array2<f64>) -> Array1<f64> {
    population.map_axis(Axis(1), |row| row.dot(&row))
}

#[test]
fn test_cancellation() {
    let mut algorithm = AlgorithmBuilder::default()
        .sampler(RandomSamplingFloat::new(-5.0, 5.0))
        .crossover(SimulatedBinaryCrossover::new(15.0))
        .mutation(GaussianMutation::new(0.05, 0.1))
        .selector(RankSelection)
        .survivor(FitnessSurvival)
        .duplicates_cleaner(CloseDuplicatesCleaner::new(1e-6))
        .fitness_fn(fitness_sphere)
        .constraints_fn(NoConstraints)
        .num_vars(3)
        .population_size(10)
        .num_offsprings(10)
        .num_iterations(100) // Many iterations
        .build()
        .expect("failed to build GA");

    let token = Arc::new(AtomicBool::new(false));
    let token_clone = token.clone();

    let iterations_run = Arc::new(AtomicUsize::new(0));
    let iterations_run_clone = iterations_run.clone();

    let callback = move |data: IterationData<_, _>| {
        iterations_run_clone.store(data.iteration, Ordering::Relaxed);
        if data.iteration >= 5 {
             token_clone.store(true, Ordering::Relaxed);
        }
    };

    algorithm.run_cancellable(token, Some(callback)).expect("GA run failed");

    // Check that it stopped early
    let iterations = iterations_run.load(Ordering::Relaxed);
    assert!(iterations >= 5 && iterations < 100, "Iterations: {}", iterations);
    // Also check context iteration. It should be 4 (0-indexed) if stopped after 5th iteration?
    // In run loop: 0..num_iterations
    // iteration passed to callback is current_iter + 1. So 1, 2, 3, 4, 5.
    // when iteration 5 (index 4) runs, callback sets token.
    // loop continues to index 5? No, next iteration checks token at start.
    // So next iteration (index 5) starts, checks token, breaks.
    // context.current_iteration was set to 4.

    // assert!(algorithm.context.current_iteration < 99);
}

#[test]
fn test_cancellation_no_callback() {
    // Just to ensure passing None works
    let mut algorithm = AlgorithmBuilder::default()
        .sampler(RandomSamplingFloat::new(-5.0, 5.0))
        .crossover(SimulatedBinaryCrossover::new(15.0))
        .mutation(GaussianMutation::new(0.05, 0.1))
        .selector(RankSelection)
        .survivor(FitnessSurvival)
        .duplicates_cleaner(CloseDuplicatesCleaner::new(1e-6))
        .fitness_fn(fitness_sphere)
        .constraints_fn(NoConstraints)
        .num_vars(3)
        .population_size(10)
        .num_offsprings(10)
        .num_iterations(10)
        .build()
        .expect("failed to build GA");

    let token = Arc::new(AtomicBool::new(false));
    // Explicitly define the callback type to help type inference
    let callback: Option<fn(IterationData<_, _>)> = None;

    algorithm.run_cancellable(token, callback).expect("GA run failed");
}
