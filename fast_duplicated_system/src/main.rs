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
struct ExponentialK {
    lambda: f64,
    uniform: Uniform,
}

impl ExponentialK {
    fn new(lambda: f64, seed: u64) -> ExponentialK {
        ExponentialK {
            lambda,
            uniform: Uniform::new(seed),
        }
    }

    fn next(&mut self, delta_k: f64) -> f64 {
        let u = self.uniform.next();
        (1.0 - u + u * (-self.lambda * delta_k).exp()).ln() / -self.lambda
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

fn calc1<Dist>(t_z: f64, main_gen: &mut ExponentialK, repair_gen: &mut Exponential, F: Dist) -> f64 where Dist : FnOnce(f64) -> f64 + Copy, {
    // k = 0
    let delta_0 = t_z - 0.0;
    let p_0 = F(delta_0);
    let tau_0 = main_gen.next(delta_0);
    let T_0 = 0.0 + tau_0;

    // k = 1
    let delta_1 = t_z - T_0;
    let p_1 = F(delta_1);
    let tau_1 = main_gen.next(delta_1);
    let T_1 = T_0 + tau_1;

    if !(tau_1 > repair_gen.next()) {
        return p_0 * p_1;
    }

    let mut T = vec![T_0, T_1];
    let mut p = vec![p_0, p_1];
    loop {
        let delta_k = t_z - T.last().unwrap();
        p.push(F(delta_k));
        let tau_k1 = main_gen.next(delta_k);
        T.push(T.last().unwrap() + tau_k1);

        if !(tau_k1 > repair_gen.next()) {
            let mut Q = 1.0;
            for &p_i in &p[0..p.len()] {
                Q *= p_i;
            }

            return Q;
        }
    }

}

fn calc(N: usize, t_z: f64, lambda: f64, mu: f64) -> f64 {
    let mut Q = Vec::new();

    let mut main_gen = ExponentialK::new(lambda, get_seed());
    let mut repair_gen = Exponential::new(mu, get_seed());

    for _ in 0..N {
       Q.push(calc1(t_z, &mut main_gen, &mut repair_gen, move |x| 1.0 - (-lambda * x).exp()));
    }

    let mut Q_sum = 0.0;
    for Q_i in Q {
        Q_sum += Q_i;
    }

    Q_sum / (N as f64)
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
