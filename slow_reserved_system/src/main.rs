use std::time::{Instant, SystemTime};

struct Uniform {
    x: u64,
}

impl Uniform {
    const A: u64 = 134775813u64;
    const B: u64 = 1u64;
    // const M: u64 = 1 << 32u64;
    const M: u64 = (1 << 32u64) - 1;
    fn new(seed: u64) -> Uniform {
        Uniform {
            // x: seed % Self::M,
            x: seed & Self::M,
        }
    }

    fn next(&mut self) -> f64 {
        // self.x = (Self::A * self.x + Self::B) % Self::M;
        self.x = (Self::A * self.x + Self::B) & Self::M;

        self.x as f64 / Self::M as f64
    }
}

struct Exponential {
    lambda: f64,
    uniform: Uniform,
}

impl Exponential {
    fn new(lambda: f64, seed: u64) -> Exponential {
        Exponential {
            lambda,
            uniform: Uniform::new(seed),
        }
    }

    fn next(&mut self) -> f64 {
        -(1.0 - self.uniform.next()).ln() / self.lambda
    }
}

fn get_seed() -> u64 {
    let seed = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_nanos();

    ((seed >> 64) ^ (seed & ((1 << 64) - 1)))
        .try_into()
        .unwrap()
}

fn calc1(
    T: f64,
    main_gen: &mut Exponential,
    repair_gen: &mut Exponential,
    main_elements_cnt: usize,
    elements_in_reserv: usize,
    repair_elements_cnt: usize,
    min_main_elements_cnt_for_work: usize,
) -> f64 {
    let find_min_in_vec = |v: &Vec<f64>| {
        let mut min_val = f64::MAX;
        for &el in v {
            min_val = min_val.min(el);
        }
        min_val
    };

    let mut main_elements_time = Vec::with_capacity(main_elements_cnt);
    for _ in 0..main_elements_cnt {
        main_elements_time.push(main_gen.next())
    }

    let mut reserv = elements_in_reserv;

    let mut repair_elements_time = Vec::with_capacity(repair_elements_cnt);
    let mut repair_queue_len = 0usize;

    let mut t = 0.0;
    loop {
        let min_main_time = find_min_in_vec(&main_elements_time);
        let min_repair_time = find_min_in_vec(&repair_elements_time);

        let theta = min_main_time.min(min_repair_time);

        if t + theta > T {
            return 0.0;
        }

        for main_elements_time_i in &mut main_elements_time {
            *main_elements_time_i -= theta;
        }
        for repair_elements_time_i in &mut repair_elements_time {
            *repair_elements_time_i -= theta
        }

        if min_main_time < min_repair_time {
            // reject main

            if reserv > 0 {
                for main_elements_time_i in &mut main_elements_time {
                    if *main_elements_time_i == 0.0 {
                        *main_elements_time_i = main_gen.next();
                        break;
                    }
                }

                reserv -= 1;
            } else {
                if main_elements_time.len() == 1 || *main_elements_time.last().unwrap() == 0.0 {
                    main_elements_time.pop();
                } else {
                    for i in 0..main_elements_time.len() {
                        if main_elements_time[i] == 0.0 {
                            main_elements_time[i] = main_elements_time.pop().unwrap();
                            break;
                        }
                    }
                }

                if main_elements_time.len() < min_main_elements_cnt_for_work {
                    return 1.0;
                }
            }

            if repair_elements_time.len() == repair_elements_cnt {
                repair_queue_len += 1;
            } else {
                repair_elements_time.push(repair_gen.next());
            }
        } else {
            // finish repair

            if main_elements_time.len() == main_elements_cnt {
                reserv += 1;
            } else {
                main_elements_time.push(main_gen.next());
            }

            if repair_queue_len > 0 {
                for repair_elements_time_i in &mut repair_elements_time {
                    if *repair_elements_time_i == 0.0 {
                        *repair_elements_time_i = repair_gen.next();
                        break;
                    }
                }
                repair_queue_len -= 1;
            } else {
                if repair_elements_time.len() == 1 || *repair_elements_time.last().unwrap() == 0.0 {
                    repair_elements_time.pop();
                } else {
                    for i in 0..repair_elements_time.len() {
                        if repair_elements_time[i] == 0.0 {
                            repair_elements_time[i] = repair_elements_time.pop().unwrap();
                            break;
                        }
                    }
                }
            }
        }

        t += theta
    }
}

fn calc(
    N: usize,
    T: f64,
    lambda: f64,
    mu: f64,
    main_elements_cnt: usize,
    reserv_elements_cnt: usize,
    repair_elements_cnt: usize,
    min_main_elements_for_work: usize,
) -> f64 {
    let mut main_gen = Exponential::new(lambda, get_seed());
    let mut repair_gen = Exponential::new(mu, get_seed());

    let mut cnt = 0.0;

    for _ in 0..N {
        cnt += calc1(
            T,
            &mut main_gen,
            &mut repair_gen,
            main_elements_cnt,
            reserv_elements_cnt,
            repair_elements_cnt,
            min_main_elements_for_work,
        );
    }

    (cnt as f64) / (N as f64)
}

fn main() {
    let params = [
        (100_000, 1.0, 5.0, 0.0001, 1, 1, 1, 1),
        (100_000, 1.0, 5.0, 0.001, 1, 1, 1, 1),
        (100_000, 1.0, 5.0, 0.01, 1, 1, 1, 1),
        (100_000, 1.0, 5.0, 0.1, 1, 1, 1, 1),
        (100_000, 1.0, 5.0, 0.5, 1, 1, 1, 1),
        (100_000, 1.0, 5.0, 1.0, 1, 1, 1, 1),
        (100_000, 1.0, 5.0, 5.0, 1, 1, 1, 1),
        
        (100_000, 1.0, 20.0, 0.0001, 1, 1, 1, 1),
        (100_000, 1.0, 20.0, 0.001 , 1, 1, 1, 1),
        (100_000, 1.0, 20.0, 0.01  , 1, 1, 1, 1),
        (100_000, 1.0, 20.0, 0.1   , 1, 1, 1, 1),
        (100_000, 1.0, 20.0, 0.5   , 1, 1, 1, 1),
        (100_000, 1.0, 20.0, 1.0   , 1, 1, 1, 1),
        (100_000, 1.0, 20.0, 5.0   , 1, 1, 1, 1),
        
        (100_000, 6.0, 10.0, 0.0001, 1, 1, 1, 1),
        (100_000, 6.0, 10.0, 0.001 , 1, 1, 1, 1),
        (100_000, 6.0, 10.0, 0.01  , 1, 1, 1, 1),
        (100_000, 6.0, 10.0, 0.1   , 1, 1, 1, 1),
        (100_000, 6.0, 10.0, 0.5   , 1, 1, 1, 1),
        (100_000, 6.0, 10.0, 1.0   , 1, 1, 1, 1),
        (100_000, 6.0, 10.0, 5.0   , 1, 1, 1, 1),

        (100_000, 1.0, 20.0, 0.0001, 2, 0, 1, 1),
        (100_000, 1.0, 20.0, 0.001, 2, 0, 1, 1),
        (100_000, 1.0, 20.0, 0.01, 2, 0, 1, 1),
        (100_000, 1.0, 20.0, 0.1, 2, 0, 1, 1),
        (100_000, 1.0, 20.0, 0.5, 2, 0, 1, 1),
        (100_000, 1.0, 20.0, 1.0, 2, 0, 1, 1),
        (100_000, 1.0, 20.0, 5.0, 2, 0, 1, 1),

        (100_000_000, 1.0, 5.0, 0.0001, 1, 1, 1, 1),
        (100_000_000, 1.0, 5.0, 0.001, 1, 1, 1, 1),
        (100_000_000, 1.0, 5.0, 0.01, 1, 1, 1, 1),
        (100_000_000, 1.0, 5.0, 0.1, 1, 1, 1, 1),
        (100_000_000, 1.0, 5.0, 0.5, 1, 1, 1, 1),
        (100_000_000, 1.0, 5.0, 1.0, 1, 1, 1, 1),
        (100_000_000, 1.0, 5.0, 5.0, 1, 1, 1, 1),
        
        (100_000_000, 1.0, 20.0, 0.0001, 1, 1, 1, 1),
        (100_000_000, 1.0, 20.0, 0.001 , 1, 1, 1, 1),
        (100_000_000, 1.0, 20.0, 0.01  , 1, 1, 1, 1),
        (100_000_000, 1.0, 20.0, 0.1   , 1, 1, 1, 1),
        (100_000_000, 1.0, 20.0, 0.5   , 1, 1, 1, 1),
        (100_000_000, 1.0, 20.0, 1.0   , 1, 1, 1, 1),
        (100_000_000, 1.0, 20.0, 5.0   , 1, 1, 1, 1),
        
        (100_000_000, 6.0, 10.0, 0.0001, 1, 1, 1, 1),
        (100_000_000, 6.0, 10.0, 0.001 , 1, 1, 1, 1),
        (100_000_000, 6.0, 10.0, 0.01  , 1, 1, 1, 1),
        (100_000_000, 6.0, 10.0, 0.1   , 1, 1, 1, 1),
        (100_000_000, 6.0, 10.0, 0.5   , 1, 1, 1, 1),
        (100_000_000, 6.0, 10.0, 1.0   , 1, 1, 1, 1),
        (100_000_000, 6.0, 10.0, 5.0   , 1, 1, 1, 1),

        (100_000_000, 1.0, 20.0, 0.0001, 2, 0, 1, 1),
        (100_000_000, 1.0, 20.0, 0.001, 2, 0, 1, 1),
        (100_000_000, 1.0, 20.0, 0.01, 2, 0, 1, 1),
        (100_000_000, 1.0, 20.0, 0.1, 2, 0, 1, 1),
        (100_000_000, 1.0, 20.0, 0.5, 2, 0, 1, 1),
        (100_000_000, 1.0, 20.0, 1.0, 2, 0, 1, 1),
        (100_000_000, 1.0, 20.0, 5.0, 2, 0, 1, 1),
    ];

    let start = Instant::now();

    for (N, lambda, mu, t, main_elements_cnt, reserv, repair_elements_cnt, min_for_work) in params {
        let result = calc(
            N,
            t,
            lambda,
            mu,
            main_elements_cnt,
            reserv,
            repair_elements_cnt,
            min_for_work,
        );
        let output = format!("{N}\n{lambda}\n{mu}\n{t}\n{main_elements_cnt}\n{reserv}\n{repair_elements_cnt}\n{min_for_work}\n{result}\n").replace('.', ",");
        println!("{output}");
    }

    println!("\ntime {}s", start.elapsed().as_secs_f64());
}
