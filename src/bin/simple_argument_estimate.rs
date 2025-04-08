use argmin::core::{CostFunction,Error,Executor};
use argmin::solver::simulatedannealing::{Anneal,SimulatedAnnealing};
use argmin_math::*;
use rand_distr::{Distribution,StandardNormal,Normal};
use anyhow::*;
use rand::Rng;

#[derive(Clone, Debug)]
struct LinerModel {
    gradient: f64,
    intercept: f64,
}

impl ArgminAdd<LinerModel, LinerModel> for LinerModel {
    fn add(&self, param: &LinerModel) -> LinerModel {
        LinerModel {
            gradient: self.gradient + param.gradient,
            intercept: self.intercept + param.intercept,
        }
    }
}

impl ArgminSub<LinerModel, LinerModel> for LinerModel {
    fn sub(&self, param: &LinerModel) -> LinerModel {
        LinerModel {
            gradient: self.gradient - param.gradient,
            intercept: self.intercept - param.intercept,
        }
    }
}

impl ArgminMul<f64, LinerModel> for LinerModel {
    fn mul(&self, param: &f64) -> LinerModel {
        LinerModel {
            gradient: self.gradient * param,
            intercept: self.intercept * param,
        }
    }
}

#[derive(Clone, Debug)]
struct LinerRegressionProblem {
    x: Vec<f64>,
    y: Vec<f64>,
}

impl CostFunction for LinerRegressionProblem {
    type Param = LinerModel;
    type Output = f64;
    
    fn cost(&self, param: &LinerModel) -> Result<f64, Error> {
        Ok(self.x.iter().zip(self.y.iter()).map(|(x,y)|{
            (param.gradient*x - y + param.intercept).powi(2)
        }).sum::<f64>() / self.x.len() as f64)
    }
}

impl Anneal for LinerRegressionProblem {
    type Param = LinerModel;
    type Output = LinerModel;
    type Float = f64;

    fn anneal(&self, param: &Self::Param, extent: Self::Float) -> Result<LinerModel, Error> {
        let mut rng = rand::rng();
        let normal = Normal::new(0.0, extent).unwrap();
        return Ok(LinerModel {
            gradient: param.gradient + extent*(rng.random::<f64>()-0.5),
            intercept: param.intercept + extent*(rng.random::<f64>()-0.5),
        })
    }
}

fn run_annealing(problem:LinerRegressionProblem) -> Result<LinerModel,Error> {
    let solver = SimulatedAnnealing::new(15.0)?
        .with_stall_best(10000);
    let res = Executor::new(problem, solver)
        .configure(|state| 
            state.param(LinerModel {
                gradient: -1.0,
                intercept: 0.2,
            }))
        .run()?.state.best_param.unwrap();
    Ok(res)
}

fn run_nealdermead(problem:LinerRegressionProblem) -> Result<LinerModel,Error> {
    let solver = argmin::solver::neldermead::NelderMead::new(vec![
        LinerModel {gradient: -1.0, intercept: 0.2},
        LinerModel {gradient: -0.95, intercept: 0.2},
        LinerModel {gradient: -1.0, intercept: 0.25},
    ]);
    let res = Executor::new(problem, solver)
        .configure(|state| state.max_iters(100)) 
        .run()?.state.best_param.unwrap();
    Ok(res)
}

fn stimulate_liner_regression_problem(rho:f64, intercept:f64, n_datapts:usize) -> Result<LinerRegressionProblem> {
    let mut rng = rand::rng();
    let normal = StandardNormal{};
    let mut result = LinerRegressionProblem {
        x: Vec::with_capacity(n_datapts),
        y: Vec::with_capacity(n_datapts),
    };

    for _ in 0..n_datapts {
        let ex :f64 = normal.sample(&mut rng);
        let ey :f64 = normal.sample(&mut rng);
        result.x.push(ex);
        result.y.push(ex*rho + intercept + ey*(1.0-rho.powi(2)).sqrt());
    };

    Ok(result)
}

fn classical_liner_regression(problem:LinerRegressionProblem) -> LinerModel {
    let n = problem.x.len() as f64;
    let x_sum = problem.x.iter().sum::<f64>();
    let y_sum = problem.y.iter().sum::<f64>();
    let xy_sum = problem.x.iter().zip(problem.y.iter()).map(|(x,y)| x*y).sum::<f64>();
    let xx_sum = problem.x.iter().map(|x| x*x).sum::<f64>();
    let gradient = (n*xy_sum - x_sum*y_sum) / (n*xx_sum - x_sum*x_sum);
    let intercept = (y_sum - gradient*x_sum) / n;
    LinerModel { gradient, intercept }
}

fn main() {
    let rho = 0.8;
    let intercept = 0.2;
    let n_datapts = 20;
    let problem = stimulate_liner_regression_problem(rho, intercept, n_datapts).unwrap();
    let result = run_annealing(problem.clone()).unwrap();
    let result2 = run_nealdermead(problem.clone()).unwrap();
    let result3 = classical_liner_regression(problem.clone());
    println!("rand:{}\nresult: {result:?}\nresult2:{result2:?}\nresult3:{result3:?}", rand::random::<f64>());
}