use rand_distr::{Distribution, Normal};
use plotters::prelude::*;
use anyhow::Result;

const NREPS       :usize = 10000;
const NSAMPLES    :usize = 2000;
const COLOR :&[RGBColor] = &[
    RGBColor(0, 0, 255),
    RGBColor(255, 0, 0),
    RGBColor(0, 255, 0),
    RGBColor(255, 255, 0),
    RGBColor(255, 0, 255),
    RGBColor(0, 255, 255),
];
const OUT_FILE_NAME_RANDOM_WALK :&str = "images\\random_walk\\random_walk.png";
const OUT_FILE_NAME_HISTO       :&str = "images\\random_walk\\histo.png";

type TimedValue = Vec::<f64>;
type WalkResult = (bool, TimedValue);

fn random_walk_model_stimulate((drift, sdrw, criterion):(f64, f64, f64)) -> Result<WalkResult> {
    let mut rng = rand::rng();
    let normal = Normal::new(drift, sdrw)?;
    let mut result = Vec::<f64>::with_capacity(NREPS/100);

    let mut choice = Option::<bool>::None;
    result.push(0f64);
    for _  in 1..NREPS {
        result.push(result.last().unwrap() + normal.sample(&mut rng));
        if (result.last().unwrap().abs() > criterion) && choice.is_none() {
            choice = Some(*result.last().unwrap() > 0f64);
            break;
        }
    }
    match choice {
        Some(c) => Ok((c, result)),
        None => Err(anyhow::anyhow!("No choice!")),
    }
}

fn draw_random_walk(walks:Vec<&TimedValue>, citerion:f64) {
    let max_latency = walks.iter().map(|i| i.len()).max().unwrap_or(0) as usize;

    let root_drawing_area = BitMapBackend::new(OUT_FILE_NAME_RANDOM_WALK, (1024, 768)).into_drawing_area();
    root_drawing_area.fill(&WHITE).unwrap();

    let mut chart = ChartBuilder::on(&root_drawing_area)
        .build_cartesian_2d(0..(max_latency+10), -citerion..citerion)
        .unwrap();

    for (walk,i) in walks.iter().zip(0..) {
        chart.draw_series(LineSeries::new(
            (0..).zip(walk.iter()).map(|(x, y)| (x, *y)),
            &COLOR[i % COLOR.len()],
        )).unwrap();
    }

    root_drawing_area.present().expect("msg: failed to present the plot");
    println!("Plot saved to {}", OUT_FILE_NAME_RANDOM_WALK);
}

fn draw_histo_gram(walks:Vec<WalkResult>) {
    let (top_latency, bottom_latency, top_num) = walks.iter().fold((0,0,0), |(t,b,c),(f,v)| {
        if *f {(t + v.len(), b, c+1)} else {(t, b + v.len(), c)}
    });
    let top_latency = top_latency as f64 / top_num as f64;
    let bottom_latency = bottom_latency as f64 / (walks.len()-top_num) as f64;

    let root_drawing_area = BitMapBackend::new(OUT_FILE_NAME_HISTO, (1024, 768)).into_drawing_area();
    root_drawing_area.fill(&WHITE).unwrap();
    let (upper, lower) = root_drawing_area.split_vertically(384);
    let root_drawing_area = [upper, lower];

    for i in 0..2 {
        let mut chart = ChartBuilder::on(&root_drawing_area[i])
            .caption(format!("{} reaction proposion {:.2}, average latency {:.2}", 
                ["top","bottom"][i], [top_num, walks.len()-top_num][i] as f64 / walks.len() as f64, [top_latency, bottom_latency][i]), 
                ("sans-serif", 20).into_font())
            .x_label_area_size(35)
            .y_label_area_size(40)
            .margin(5)
            .build_cartesian_2d(0..800, 0..20)
            .unwrap();

        chart
            .configure_mesh()
            .x_desc("Latency")
            .y_desc("Count")
            .draw()
            .unwrap();
        
        chart.draw_series(Histogram::vertical(&chart)
            .style(RED.mix(0.5).filled())
            .data(walks.iter()
                .filter(|(f,_)|{*f == [true,false][i]})
                .map(|(_,v)|{(v.len() as i32, 1)})
            )).unwrap();
    }

    root_drawing_area[0].present().expect("msg: failed to present the plot");
    root_drawing_area[1].present().expect("msg: failed to present the plot");
    println!("Plot saved to {}", OUT_FILE_NAME_HISTO);
}

fn main() {
    let drift = 0.00;
    let sdrw = 0.3;
    let citerion = 3.0;
    let mut tries = Vec::<WalkResult>::with_capacity(NSAMPLES);

    println!("INITIALIZE: drift: {}, sdrw: {}, citerion: {}", drift, sdrw, citerion);
    
    for _ in 0..NSAMPLES {
        if let Ok(r) = random_walk_model_stimulate((drift,sdrw,citerion)) {
            tries.push(r);
        }
    }

    println!("FINISHED: {} samples", tries.len());

    draw_random_walk(tries.iter().take(5).map(|(_,v)|{v}).collect(), citerion);
    draw_histo_gram(tries);
}