use std::time::{Instant, SystemTime};

struct Uniform {
    x: u64,
}

impl Uniform {
    const A: u64 = 134775813u64;
    const B: u64 = 1u64;
    // const A: u64 = 1664525u64;
    // const B: u64 = 1013904223u64;

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

fn calc1<Dist>(
    t_z: f64,
    main_gen: &mut ExponentialK,
    repair_gen: &mut Exponential,
    F: Dist,
    mut main_elements_time: Vec<Option<f64>>,
    mut repair_elements_time: Vec<Option<f64>>,
    main_elements_cnt: usize,
    mut elements_in_reserv: usize,
    mut repair_queue_len: usize,
    repair_elements_cnt: usize,
    min_main_elements_for_work: usize,
) -> f64
where
    Dist: FnOnce(f64) -> f64 + Copy,
{
    {
        let mut working_main_elements_cnt = 0usize;
        for main_elements_time_i in &main_elements_time {
            if main_elements_time_i.is_some() {
                working_main_elements_cnt += 1;
            }
        }

        if working_main_elements_cnt < min_main_elements_for_work {
            return  1.0
        };
    }

    /*
       main stans
       1 - init -- not exist  ------------------ out of suport ------------------
       2 - work -- Some(t)
           2.1 - < t_z -- t < t_z
           2.2 - > t_z -- t > t_z
       3 - wait element from reserv -- None

       repair stans
       1 - repair  -- Some(t)
       2 - wait main element -- None
    */

    let find_min_value = |v: &Vec<Option<f64>>| {
        let mut min_val = f64::MAX;
        for &el in v {
            min_val = min_val.min(el.unwrap_or(f64::MAX));
        }
        min_val
    };

    let mut t = 0.0;

    let min_main_time = find_min_value(&main_elements_time);
    let min_repair_time = find_min_value(&repair_elements_time);

    let theta = min_main_time.min(min_repair_time);

    if t + theta > t_z {
        return 0.0;
    }

    t += theta;

    for main_elements_time_i in &mut main_elements_time {
        if main_elements_time_i.is_some() {
            main_elements_time_i.replace(main_elements_time_i.unwrap() - theta);
        }
    }
    for repair_elements_time_i in &mut repair_elements_time {
        if repair_elements_time_i.is_some() {
            repair_elements_time_i.replace(repair_elements_time_i.unwrap() - theta);
        }
    }

    if min_main_time < min_repair_time {
        for main_elements_time_i in &mut main_elements_time {
            if main_elements_time_i.is_some_and(|x| x == 0.0) {
                *main_elements_time_i = None;
                break;
            }
        }

        repair_queue_len += 1;
        for repair_elements_time_i in &mut repair_elements_time {
            if repair_elements_time_i.is_none() {
                *repair_elements_time_i = Some(repair_gen.next());
                repair_queue_len -= 1;
                break;
            }
        }

    } else {
        elements_in_reserv += 1;

        for repair_elements_time_i in &mut repair_elements_time {
            if repair_elements_time_i.is_some_and(|x| x == 0.0) {
                if repair_queue_len > 0 {
                    *repair_elements_time_i = Some(repair_gen.next());
                    repair_queue_len -= 1;
                } else {
                    *repair_elements_time_i = None;
                }

                break;
            }
        }
    }

    let mut not_working_main_element_pos = None;
    for i in 0..main_elements_time.len() {
        if main_elements_time[i].is_none() {
            not_working_main_element_pos = Some(i);
            break;
        }
    }

    if not_working_main_element_pos.is_some() && elements_in_reserv > 0 {
        elements_in_reserv -= 1;

        let mut main_elements_time_less_tz = main_elements_time.clone();
        let mut main_elements_time_greater_tz = main_elements_time;

        main_elements_time_less_tz[not_working_main_element_pos.unwrap()] =
            Some(main_gen.next(t_z - t));
        main_elements_time_greater_tz[not_working_main_element_pos.unwrap()] = Some(t_z + 1.0);

        let less_tz_probability = F(t_z - t);

        return less_tz_probability
            * calc1(
                t_z - t,
                main_gen,
                repair_gen,
                F,
                main_elements_time_less_tz,
                repair_elements_time.clone(),
                main_elements_cnt,
                elements_in_reserv,
                repair_queue_len,
                repair_elements_cnt,
                min_main_elements_for_work,
            )
            + (1.0 - less_tz_probability)
                * calc1(
                    t_z - t,
                    main_gen,
                    repair_gen,
                    F,
                    main_elements_time_greater_tz,
                    repair_elements_time,
                    main_elements_cnt,
                    elements_in_reserv,
                    repair_queue_len,
                    repair_elements_cnt,
                    min_main_elements_for_work,
                );
    } else {
        return calc1(
            t_z - t,
            main_gen,
            repair_gen,
            F,
            main_elements_time,
            repair_elements_time,
            main_elements_cnt,
            elements_in_reserv,
            repair_queue_len,
            repair_elements_cnt,
            min_main_elements_for_work,
        );
    }
}

fn calc(
    iteration_cnt: usize,
    t_z: f64,
    lambda: f64,
    mu: f64,
    main_cnt: usize,
    reserv_cnt: usize,
    repair_cnt: usize,
    min_work: usize,
) -> f64 {
    let mut main_gen = ExponentialK::new(lambda, get_seed());
    let mut repair_gen = Exponential::new(mu, get_seed());

    let less_tz_probability = 1.0 - (-lambda * t_z).exp(); // F(t_z), where F(x) - exponential distribution function for main elements time (with param lambda)

    let mut Q_sum = 0.0;
    for _j in 0..iteration_cnt {
        let mut Q_j = 0.0; // result iteration #j
        /*
        case params: how many main elements time is less than t_z = n, 0 <= n <= main_cnt
        main_elements_time = [Some(< t_z), Some(< t_z) ... Some(< t_z), Some(> t_z), Some(> t_z) ... Some(> t_z)]
                                                        n                                    main_cnt - n
        probability = F(t_z)^n * (1 - F(t_z))^(main_cnt - n)
        permutation_cnt = C(main_cnt, n), where C(n, k) = n! / (k! * (n - k)!)
        */

        // first case: n = 0
        let mut case_main_elements_time = vec![Some(t_z + 1.0); main_cnt];
        let mut case_probability = (1.0 - less_tz_probability).powi(
            main_cnt
                .try_into()
                .expect("Can not convert main_cnt: usize into i32"),
        );
        let mut case_permutation_cnt = 1.0;

        for main_elements_cnt_less_tz_in_cur_case in 0..=main_cnt {
            // case n = main_elemetns_less_tz_in_cur_case
            let case_result = calc1(
                t_z,
                &mut main_gen,
                &mut repair_gen,
                move |x| 1.0 - (-lambda * x).exp(),
                case_main_elements_time.clone(),
                vec![None; repair_cnt],
                main_cnt,
                reserv_cnt,
                0,
                repair_cnt,
                min_work,
            );

            Q_j += case_permutation_cnt
                * case_probability
                * case_result;
            
            // println!("{case_permutation_cnt}\n{case_probability}\n{case_result}\n");
            /*
            values on step n = k:
            main_elements_time_k = [Some(< t_z), Some(< t_z) ... Some(< t_z), Some(> t_z), Some(> t_z) ... Some(> t_z)]
                                                            k                                    main_cnt - k
            probability_k = F(t_z)^k * (1 - F(t_z))^(main_cnt - k)
            permutation_cnt_k = C(main_cnt, k) = main_cnt! / (k! * (main_cnt - k)!)

            values on step k+1
            main_elements_time_k+1 = [Some(< t_z), Some(< t_z) ... Some(< t_z), Some(> t_z), Some(> t_z) ... Some(> t_z)]
                                                              k + 1                               main_cnt - (k+1)
            main_elements_time_k+1[i] = main_elements_time_k[i], where 0 <= i < main_cnt and i != k
            main_elements_time_k+1[k] = Some(< t_z)

            probability_k+1 = F(t_z)^(k+1) * (1 - F(t_z))^(main_cnt - (k+1)) = F(t_z)^k * F(t_z) * (1 - F(t_z))^(main_cnt - k) / (1 - F(t_z)) =
                            = (F(t_z)^k * (1 - F(t_z))^(main_cnt - k)) * F(t_z) / (1 - F(t_z)) = probability_k * F(t_z) / (1 - F(t_z))

            permutation_cnt_k+1 = C(main_cnt, k+1) = main_cnt! / ((k+1)! * (main_cnt - (k+1))!) = main_cnt! / (k! * (k+1) * (main_cnt - k - 1)!) =
                                = (main_cnt! / (k! * (main_cnt - k)!)) * (main_cnt - k) / (k + 1) = C(main_cnt, k) * (main_cnt - k) / (k + 1) =
                                = permutation_cnt_k * (main_cnt - k) / (k + 1)

             */

            if main_elements_cnt_less_tz_in_cur_case < main_cnt {
                case_main_elements_time[main_elements_cnt_less_tz_in_cur_case] =
                    Some(main_gen.next(t_z));
                case_probability *= less_tz_probability / (1.0 - less_tz_probability);
                case_permutation_cnt *= ((main_cnt - main_elements_cnt_less_tz_in_cur_case) as f64)
                    / ((main_elements_cnt_less_tz_in_cur_case + 1) as f64)
            }
        }

        Q_sum += Q_j;
    }

    Q_sum / (iteration_cnt as f64)
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
