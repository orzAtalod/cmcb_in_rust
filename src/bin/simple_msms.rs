use rand::Rng;
use rand_distr::{Distribution, Uniform, Normal};
use argmin::core::ArgminFloat;

struct McmcResult<FLOAT:Copy> {
    value: Vec<FLOAT>,
    burnin: usize,
}

impl<FLOAT:Copy> Distribution<FLOAT> for McmcResult<FLOAT> {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> FLOAT {
        let index = Uniform::<usize>::new(self.burnin, self.value.len()).unwrap();
        let index  = index.sample(rng);
        self.value[index]
    }
}

fn simple_mcmc_core<T,R,F1,F2,FLOAT>(init: FLOAT, proposed_distr: T, unnormal_possibility: F1, stop_condition: F2, rng: &mut R) -> McmcResult<FLOAT>
    where T: Distribution<FLOAT>,
          F1: Fn(FLOAT) -> f64,
          F2: Fn(&McmcResult<FLOAT>) -> bool,
          R: Rng + ?Sized,
          FLOAT: ArgminFloat {
    let mut res= McmcResult {
        value: vec![init],
        burnin: 0,
    };
    let mut old_value = init;
    let mut old_possi= unnormal_possibility(init);
    while !stop_condition(&res) {
        let new_value = old_value + proposed_distr.sample(rng);
        let new_possi = unnormal_possibility(new_value);
        if (new_possi > old_possi) || (rng.random::<f64>() < new_possi / old_possi) {
            old_value = new_value;
            old_possi = new_possi;
        }
        res.value.push(old_value);
    }
    res
}

fn mcmc_stop_after<FLOAT:Copy>(step: usize) -> Box<dyn Fn(&McmcResult<FLOAT>) -> bool> {
    Box::new(move |chain: &McmcResult<FLOAT>| chain.value.len() > step)
}

fn simple_mcmc<T,F1>(init: f64, proposed_distr: T, unnormal_possibility: F1) -> McmcResult<f64> 
    where T: Distribution<f64>,
          F1: Fn(f64) -> f64 {
    let mut rng = rand::rng();
    let mut res = simple_mcmc_core(init, proposed_distr, unnormal_possibility, mcmc_stop_after(5000), &mut rng);
    res.burnin = 1000;
    res
}

fn likelihood(x: f64) -> f64 {
    Normal::new(x, 15.0).unwrap();
    10.0
}

fn main() {

}