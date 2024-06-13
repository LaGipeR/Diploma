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
        let u = self.uniform.next();
        -(1.0 - u).ln() / self.lambda
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

fn calc1(T: f64, main_gen: &mut Exponential, repair_gen: &mut Exponential) -> bool {
    let mut main = main_gen.next();
    let mut repair = T + 1.0;

    let mut t = 0.0;
    loop {
        if t + main > T {
            return false;
        }

        if repair > T {
            t += main;

            main = main_gen.next(); 
            repair = repair_gen.next();
        } else if repair < main {
            t += repair;
            
            main -= repair;
            repair = T + 1.0;
        } else {
            return true;
        }
    }
}

fn calc(N: usize, T: f64, lambda: f64, mu: f64) -> f64 {
    let mut main_gen = Exponential::new(lambda, get_seed());
    let mut repair_gen = Exponential::new(mu, get_seed());

    let mut cnt = 0;
    
    for _ in 0..N {
        cnt += calc1(T, &mut main_gen, &mut repair_gen) as usize;
    }

    (cnt as f64) / (N as f64)
}

fn main() {
    let params = [
        (100_000, 1.0,  20.0, 0.0001),
        (100_000, 1.0,  20.0, 0.001),
        (100_000, 1.0,  20.0, 0.01),
        (100_000, 1.0,  20.0, 0.1),
        (100_000, 1.0,  20.0, 0.5),
        (100_000, 1.0,  20.0, 1.0),
        (100_000, 1.0,  20.0, 5.0),

        (100_000, 1.0,  5.0, 0.0001),
        (100_000, 1.0,  5.0, 0.001),
        (100_000, 1.0,  5.0, 0.01),
        (100_000, 1.0,  5.0, 0.1),
        (100_000, 1.0,  5.0, 0.5),
        (100_000, 1.0,  5.0, 1.0),
        (100_000, 1.0,  5.0, 5.0),

        (100_000, 6.0, 10.0, 0.0001),
        (100_000, 6.0, 10.0, 0.001),
        (100_000, 6.0, 10.0, 0.01),
        (100_000, 6.0, 10.0, 0.1),
        (100_000, 6.0, 10.0, 0.5),
        (100_000, 6.0, 10.0, 1.0),
        (100_000, 6.0, 10.0, 5.0),

        (100_000_000, 1.0,  20.0, 0.0001),
        (100_000_000, 1.0,  20.0, 0.001),
        (100_000_000, 1.0,  20.0, 0.01),
        (100_000_000, 1.0,  20.0, 0.1),
        (100_000_000, 1.0,  20.0, 0.5),
        (100_000_000, 1.0,  20.0, 1.0),
        (100_000_000, 1.0,  20.0, 5.0),

        (100_000_000, 1.0,  5.0, 0.0001),
        (100_000_000, 1.0,  5.0, 0.001),
        (100_000_000, 1.0,  5.0, 0.01),
        (100_000_000, 1.0,  5.0, 0.1),
        (100_000_000, 1.0,  5.0, 0.5),
        (100_000_000, 1.0,  5.0, 1.0),
        (100_000_000, 1.0,  5.0, 5.0),

        (100_000_000, 6.0, 10.0, 0.0001),
        (100_000_000, 6.0, 10.0, 0.001),
        (100_000_000, 6.0, 10.0, 0.01),
        (100_000_000, 6.0, 10.0, 0.1),
        (100_000_000, 6.0, 10.0, 0.5),
        (100_000_000, 6.0, 10.0, 1.0),
        (100_000_000, 6.0, 10.0, 5.0),
        ];

    let start = Instant::now();

    for (N, lambda, mu, t) in params {
        let result = calc(N, t, lambda, mu);
        let output = format!("{N}\n{lambda}\n{mu}\n{t}\n{result}\n").replace('.', ",");
        println!("{output}");
    }

    println!("\ntime {}s", start.elapsed().as_secs_f64());
}

