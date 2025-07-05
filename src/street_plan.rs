use core::f32;
use std::{collections::BinaryHeap, ops::Range};

use nalgebra::Vector2;
use rand::Rng;
use tokio::task::JoinError;

use crate::{
    aabb::BoundingBox,
    tensor_field::{EvalEigenvectors, GRID_SIZE, Point, TensorField},
};

pub fn distribute_points(point_count: u32) -> Vec<Point> {
    let mut rand = rand::rng();

    let mut points = Vec::new();

    while (points.len() as u32) < point_count {
        let candidates: Vec<Point> = (0..10)
            .map(|_| {
                Point::new(
                    rand.random_range(0..GRID_SIZE) as f32,
                    rand.random_range(0..GRID_SIZE) as f32,
                )
            })
            .collect();
        let distances: Vec<f32> = candidates
            .iter()
            .map(|point| closest_distance_to_points(*point, &points))
            .collect();
        let farthest_point_index = distances
            .iter()
            .enumerate()
            .fold((0, distances[0]), |acc, (i, dist)| {
                if *dist > acc.1 { (i, *dist) } else { acc }
            })
            .0;

        points.push(candidates[farthest_point_index]);
    }

    points
}

fn closest_distance_to_points(candidate: Point, points: &[Point]) -> f32 {
    points
        .iter()
        .fold(f32::MAX, |acc, p| {
            let dist_squared = (p - candidate).magnitude_squared();
            if dist_squared < acc {
                dist_squared
            } else {
                acc
            }
        })
        .sqrt()
}

#[derive(Debug, PartialEq, PartialOrd)]
pub struct SeedPoint {
    pub seed: Point,
    pub priority: f32,
    pub follow_major_eigenvector: bool,
}

impl Eq for SeedPoint {}

impl Ord for SeedPoint {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.priority.total_cmp(&other.priority)
    }
}

pub fn prioritize_points(
    points: &[Point],
    city_center: Point,
    tensor_field: &TensorField,
) -> BinaryHeap<SeedPoint> {
    points
        .iter()
        .flat_map(|point| {
            let tensor = tensor_field.evaluate_field_at_point(*point);
            if tensor.norm_squared() <= 0.0001 {
                None
            } else {
                let city_center_priority = (-(city_center - point).magnitude()).exp();
                let degenerate_point_priority =
                    (-closest_degenerate_point_distance(*point, tensor_field)).exp();

                Some(SeedPoint {
                    seed: *point,
                    priority: city_center_priority + degenerate_point_priority,
                    follow_major_eigenvector: true,
                })
            }
        })
        .collect()
}

/// Divide the full grid into rectangular sectors. Search each sector individually for a degenerate
/// point starting with the sectors closest to the queried point. The distance to the center of the closest sector with a
/// degenerate point is returned. Precision is not essential, so the centers should be good enough.
/// More sectors can help with precision, but decrease performance
fn closest_degenerate_point_distance(point: Point, tensor_field: &TensorField) -> f32 {
    let horizontal_sector_count = 16;
    let vertical_sector_count = horizontal_sector_count;

    let horizontal_sector_size = GRID_SIZE / horizontal_sector_count;
    let vertical_sector_size = GRID_SIZE / vertical_sector_count;

    let mut sector_order = (0..horizontal_sector_count)
        .flat_map(|x| {
            (0..vertical_sector_count)
                .map(|y| {
                    Point::new(
                        (x * horizontal_sector_size) as f32 + horizontal_sector_size as f32 / 2.0,
                        (y * vertical_sector_size) as f32 + vertical_sector_size as f32 / 2.0,
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    sector_order.sort_by(|a, b| {
        (a - point)
            .norm_squared()
            .total_cmp(&(b - point).norm_squared())
    });

    (sector_order
        .into_iter()
        .filter(|sector_center| {
            let x_sector = (sector_center.x - horizontal_sector_size as f32 / 2.0)
                / horizontal_sector_size as f32;
            let y_sector =
                (sector_center.y - vertical_sector_size as f32 / 2.0) / vertical_sector_size as f32;

            sector_has_degenerate_point(
                x_sector as u32,
                y_sector as u32,
                horizontal_sector_size,
                vertical_sector_size,
                tensor_field,
            )
        })
        .next()
        .unwrap_or(Point::new(f32::MAX, f32::MAX))
        - point)
        .norm_squared()
}

fn sector_has_degenerate_point(
    x_sector: u32,
    y_sector: u32,
    horizontal_sector_size: u32,
    vertical_sector_size: u32,
    tensor_field: &TensorField,
) -> bool {
    for x_in_sector in 0..horizontal_sector_size {
        for y_in_sector in 0..vertical_sector_size {
            let x = x_in_sector + x_sector * horizontal_sector_size;
            let y = y_in_sector + y_sector * vertical_sector_size;
            let tensor = tensor_field.evaluate_field_at_point(Point::new(x as f32, y as f32));
            if tensor.norm_squared() <= 0.0001 {
                return true;
            }
        }
    }

    return false;
}

pub async fn trace_street_plan(
    tensor_field: &TensorField,
    starting_seed_count: u32,
    city_center: Point,
    horizontal_sector_count: u32,
    vertical_sector_count: u32,
) -> (Vec<Vec<Point>>, usize) {
    let temp_points = [
        (391.0, 113.0),
        (10.0, 470.0),
        (382.0, 472.0),
        (61.0, 152.0),
        (413.0, 291.0),
        (191.0, 298.0),
        (0.0, 303.0),
        (147.0, 0.0),
        (304.0, 294.0),
        (298.0, 41.0),
        (230.0, 509.0),
        (502.0, 416.0),
        (127.0, 205.0),
        (285.0, 162.0),
        (459.0, 40.0),
        (299.0, 436.0),
        (121.0, 472.0),
        (508.0, 493.0),
        (470.0, 151.0),
        (214.0, 413.0),
        (364.0, 355.0),
        (171.0, 63.0),
        (355.0, 191.0),
        (274.0, 355.0),
        (66.0, 336.0),
        (230.0, 65.0),
        (30.0, 31.0),
        (223.0, 12.0),
        (193.0, 146.0),
        (447.0, 224.0),
    ]
    .map(|p| Point::new(p.0, p.1));
    let mut seed_points = prioritize_points(
        &temp_points,
        // &distribute_points(starting_seed_count),
        city_center,
        &tensor_field,
    );

    let h = 0.2;
    let d_sep = 20.0;

    let traces = trace_lanes(
        seed_points
            .into_iter()
            .enumerate()
            .map(|(i, seed)| (i, seed.seed))
            .collect(),
        tensor_field,
        h,
        d_sep,
        true,
        200.0,
        false,
    )
    .await;

    let curve_paths = smooth_lanes(traces, 0.03, 0.3, 20, h, 0.7)
        .await
        .unwrap();

    let clipped_paths = clip_pass(curve_paths, d_sep);

    (
        clipped_paths
            .iter()
            .map(|control_points| resample_curve(control_points, 20))
            .collect(),
        clipped_paths.len(),
    )

    /* let mut major_line_traces = Vec::new();
    let mut major_bounding_boxes: Vec<BoundingBox> = Vec::new();

    let mut minor_line_traces = Vec::new();
    let mut minor_bounding_boxes: Vec<BoundingBox> = Vec::new();

    // let mut full_trace = Vec::new();

    while let Some(SeedPoint {
        seed,
        follow_major_eigenvector,
        ..
    }) = seed_points.pop()
    {
        let d_sep = 20.0; // + 0.7 * (seed - city_center).norm();
        let trace_res = trace(
            tensor_field,
            seed,
            0.2,
            d_sep,
            follow_major_eigenvector,
            200.0,
            false,
        );

        if trace_res.raw_points.is_empty() {
            continue;
        }

        let trimmed_range = trim_trace(
            &trace_res.raw_points,
            if follow_major_eigenvector {
                &major_bounding_boxes
            } else {
                &minor_bounding_boxes
            },
            0,
            trace_res.raw_points.len() - 1,
            d_sep,
        );

        if trimmed_range.end - trimmed_range.start < 10 {
            continue;
        }

        let trace_path = trace_res.raw_points[trimmed_range].to_vec();

        if !is_path_long_enough(&trace_path, d_sep) {
            continue;
        }

        let mut bounding_box = vec![BoundingBox::new(&trace_path)];
        let frac_1_sqrt_2 = std::f32::consts::FRAC_1_SQRT_2;
        let bb = &bounding_box[0];
        let rectangularness = Vector2::new(bb.east() - bb.west(), bb.north() - bb.south())
            .normalize()
            .dot(&Vector2::new(frac_1_sqrt_2, frac_1_sqrt_2));
        if rectangularness > 0.5 || rectangularness < 3.0_f32.sqrt() / 2.0 {
            bounding_box = vec![
                BoundingBox::new(&trace_path[..trace_path.len() / 2]),
                BoundingBox::new(&trace_path[trace_path.len() / 2..]),
            ];
        }

        // full_trace.push(trace_path.clone());

        if follow_major_eigenvector {
            major_line_traces.push(trace_path);
            major_bounding_boxes.extend(bounding_box);
        } else {
            minor_line_traces.push(trace_path);
            minor_bounding_boxes.extend(bounding_box);
        }
        for new_seed in trace_res.new_seeds {
            seed_points.push(SeedPoint {
                seed: new_seed,
                priority: f32::MAX,
                follow_major_eigenvector: !follow_major_eigenvector,
            });
        }
    }

    let len = major_line_traces.len();
    let mut full_trace = major_line_traces;
    full_trace.extend(minor_line_traces);

    (
        futures::future::join_all(full_trace.into_iter().map(|trace| {
            tokio::spawn(async move {
                let first_pass = smooth_path(trace, 0.03, 0.3);
                resample_curve(&first_pass, 0.7, 10, 20, 0.2)
            })
        }))
        .await
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .expect("Failed to run smoothing"),
        len,
    ) */
}

fn is_path_long_enough(path: &[Point], d_sep: f32) -> bool {
    let mut last_point = path[0];
    let mut dist = 0.0;
    for point in path {
        dist += (point - last_point).norm();
        if dist > d_sep {
            return true;
        }
        last_point = *point;
    }

    false
}

async fn trace_lanes(
    seeds: Vec<(usize, Point)>,
    tensor_field: &TensorField,
    h: f32,
    d_sep: f32,
    follow_major_eigenvectors: bool,
    max_len: f32,
    reverse: bool,
) -> Vec<TraceOutput> {
    let (_, unordered_traces) = async_scoped::TokioScope::scope_and_block(|scope| {
        for (index, seed) in seeds {
            scope.spawn(async move {
                (
                    index,
                    trace(
                        tensor_field,
                        seed,
                        h,
                        d_sep,
                        follow_major_eigenvectors,
                        max_len,
                        reverse,
                    ),
                )
            });
        }
    });

    let mut ordered_traces = vec![
        TraceOutput {
            path: Vec::new(),
            new_seeds: Vec::new()
        };
        unordered_traces.len()
    ];

    for trace_res in unordered_traces {
        if let Ok((index, trace_output)) = trace_res {
            ordered_traces[index] = trace_output;
        }
    }

    ordered_traces
}

#[derive(Debug, Clone, Default)]
struct TraceOutput {
    path: Vec<Point>,
    new_seeds: Vec<Point>,
}

fn trace(
    tensor_field: &TensorField,
    seed: Point,
    h: f32,
    d_sep: f32,
    follow_major_eigenvectors: bool,
    max_len: f32,
    reverse: bool,
) -> TraceOutput {
    let origin = seed;
    let mut seed = seed;
    let mut path = vec![seed];
    let mut accumulated_distance = 0.0;
    let mut new_seeds = vec![];
    let mut steps = 0;

    let clamp_vel = |vel: Vector2<f32>| {
        clamp_vector_between_points(
            Vector2::new(-(GRID_SIZE as f32), -(GRID_SIZE as f32)),
            Vector2::new(GRID_SIZE as f32, GRID_SIZE as f32),
            vel,
        )
    };

    while !(seed.x < 0.0 || seed.y < 0.0 || seed.x > GRID_SIZE as f32 || seed.y > GRID_SIZE as f32)
    {
        let tensor = tensor_field.evaluate_smoothed_field_at_point(seed);

        if tensor.eigenvalues().is_none() {
            break;
        }

        if tensor.norm_squared() <= 0.00001 {
            break;
        }

        let rev_factor = -2.0 * (reverse as i32 as f32 - 0.5);

        let k_1_eigenvectors = tensor.eigenvectors();
        let k_1 = rev_factor
            * branchless_if(
                clamp_vel(k_1_eigenvectors.major),
                clamp_vel(k_1_eigenvectors.minor),
                follow_major_eigenvectors,
            )
            .normalize();
        let k_2_eigenvectors = tensor_field
            .evaluate_smoothed_field_at_point(clamp_vec_to_grid(seed + h / 2.0 * k_1))
            .eigenvectors();
        let k_2 = rev_factor
            * branchless_if(
                clamp_vel(k_2_eigenvectors.major),
                clamp_vel(k_2_eigenvectors.minor),
                follow_major_eigenvectors,
            )
            .normalize();
        let k_3_eigenvectors = tensor_field
            .evaluate_smoothed_field_at_point(clamp_vec_to_grid(seed + h / 2.0 * k_2))
            .eigenvectors();
        let k_3 = rev_factor
            * branchless_if(
                clamp_vel(k_3_eigenvectors.major),
                clamp_vel(k_3_eigenvectors.minor),
                follow_major_eigenvectors,
            )
            .normalize();
        let k_4_eigenvectors = tensor_field
            .evaluate_smoothed_field_at_point(clamp_vec_to_grid(seed + h * k_3))
            .eigenvectors();
        let k_4 = rev_factor
            * branchless_if(
                clamp_vel(k_4_eigenvectors.major),
                clamp_vel(k_4_eigenvectors.minor),
                follow_major_eigenvectors,
            )
            .normalize();

        let m = 1.0 / 6.0 * k_1 + 1.0 / 3.0 * k_2 + 1.0 / 3.0 * k_3 + 1.0 / 6.0 * k_4;

        let new_pos = seed + h * m;

        accumulated_distance += (new_pos - seed).norm();
        if accumulated_distance >= d_sep {
            accumulated_distance = 0.0;
            new_seeds.push(new_pos);
        }
        seed = new_pos;

        if h < 1.0 {
            if steps as f32 % (1.0 / h) <= 0.001 {
                path.push(new_pos);
            }
        }
        steps += 1;

        if (new_pos - origin).magnitude_squared() <= 0.0001 || accumulated_distance > max_len {
            break;
        }
    }

    TraceOutput { path, new_seeds }
}

fn branchless_if<T: std::ops::Mul<f32, Output = T> + std::ops::Add<T, Output = T>>(
    true_val: T,
    false_val: T,
    condition: bool,
) -> T {
    true_val * (condition as i32) as f32 + false_val * (1 - condition as i32) as f32
}

fn clamp_vector_between_points(p_0: Point, p_1: Point, subject: Point) -> Point {
    let max_point = Point::new(p_0.x.max(p_1.x), p_0.y.max(p_1.y));
    let min_point = Point::new(p_0.x.min(p_1.x), p_0.y.min(p_1.y));

    Point::new(
        subject.x.clamp(min_point.x, max_point.x),
        subject.y.clamp(min_point.y, max_point.y),
    )
}

fn clamp_vec_to_grid(vec: Vector2<f32>) -> Vector2<f32> {
    let clamped_x = vec.x.clamp(0.0, GRID_SIZE as f32);
    let clamped_y = vec.y.clamp(0.0, GRID_SIZE as f32);
    Vector2::new(
        if clamped_x.is_nan() { 1.0 } else { clamped_x },
        if clamped_y.is_nan() { 1.0 } else { clamped_y },
    )
}

#[derive(Debug, Clone, Copy)]
struct ControlPoint {
    position: Point,
    velocity: Vector2<f32>,
}

type HermiteCurve = Vec<ControlPoint>;

async fn smooth_lanes(
    traces: Vec<TraceOutput>,
    alpha: f32,
    beta: f32,
    point_side_padding: usize,
    h: f32,
    blend_factor: f32,
) -> Result<Vec<HermiteCurve>, JoinError> {
    let h_square = h * h;
    let (_, res) = async_scoped::TokioScope::scope_and_block(|scope| {
        for trace in traces{
            let raw_path = trace.path;
            scope.spawn(async move {
                let smoothed_path = smooth_path(raw_path.clone(), alpha, beta);

                let mut control_points_indices: Vec<usize> =
                    highest_curvature_points(&smoothed_path, point_side_padding);

                control_points_indices.sort();

                control_points_indices
                    .iter()
                    .map(|&index| {
                        let position = smoothed_path[index];
                        let velocity = if index == 0 {
                            (smoothed_path[1] - position) / h_square
                        } else if index == smoothed_path.len() - 1 {
                            (smoothed_path[smoothed_path.len() - 1]
                                - smoothed_path[smoothed_path.len() - 2])
                                / h_square
                        } else {
                            blend_factor
                                * ((smoothed_path[index + 1] - position) / h_square
                                    + (position - smoothed_path[index - 1]) / h_square)
                        };

                        ControlPoint { position, velocity }
                    })
                    .collect::<Vec<_>>()
            });
        }
    });

    res.into_iter().collect()
}

pub fn smooth_path(path: Vec<Point>, alpha: f32, beta: f32) -> Vec<Point> {
    let first_pass: Vec<Point> = path
        .iter()
        .enumerate()
        .map(|(i, point)| {
            if i == 0 || i == path.len() - 1 {
                *point
            } else {
                let current_vel = point_first_deriv(i, &path);
                let prev_vel = point_first_deriv(i - 1, &path);

                let current_vel_normed = point_first_deriv(i, &path).normalize();
                let prev_vel_normed = point_first_deriv(i - 1, &path).normalize();

                if ((1.0 - current_vel_normed.dot(&prev_vel_normed).min(1.0)).min(alpha) / alpha)
                    .floor()
                    == 0.0
                {
                    *point
                } else {
                    point
                        + (current_vel - prev_vel) / 2.0
                            * (current_vel.norm_squared() / (current_vel - prev_vel).norm_squared())
                                .sqrt()
                }
            }
        })
        .collect();

    first_pass
        .iter()
        .enumerate()
        .map(|(i, point)| {
            if i == 0 || i == first_pass.len() - 1 {
                *point
            } else if ((1.0 - beta)
                + (point_first_deriv(i, &first_pass).normalize()
                    - point_first_deriv(i - 1, &first_pass).normalize())
                .norm())
            .floor()
                > 0.0
            {
                (first_pass[i - 1] + point + first_pass[i + 1]) / 3.0
            } else {
                *point
            }
        })
        .collect()
}

fn point_first_deriv(point_index: usize, path: &[Point]) -> Vector2<f32> {
    path[point_index + 1] - path[point_index]
}

pub fn point_second_deriv(point_index: usize, path: &[Point]) -> Vector2<f32> {
    point_first_deriv(point_index, path).normalize()
        - point_first_deriv(point_index - 1, path).normalize()
}

fn resample_curve(control_points: &[ControlPoint], points_per_spline: i32) -> Vec<Point> {
    let sampler = |p_0: Point, p_1: Point, m_0: Vector2<f32>, m_1: Vector2<f32>| -> Vec<Point> {
        (0..=points_per_spline)
            .map(|i| {
                evaluate_hermite_curve(p_0, p_1, m_0, m_1, i as f32 / points_per_spline as f32)
            })
            .collect()
    };

    (0..control_points.len() - 1)
        .flat_map(|i| {
            let this_control_point = control_points[i];
            let next_control_point = control_points[i + 1];
            sampler(
                this_control_point.position,
                next_control_point.position,
                this_control_point.velocity,
                next_control_point.velocity,
            )
        })
        .collect()
}

pub fn highest_curvature_points(path: &[Point], point_padding_per_side: usize) -> Vec<usize> {
    let mut indices_sorted_by_curvature: Vec<usize> = (1..path.len() - 1).collect();

    indices_sorted_by_curvature.sort_by(|&a, &b| {
        point_second_deriv(a, path)
            .norm_squared()
            .partial_cmp(&point_second_deriv(b, path).norm_squared())
            .unwrap()
            .reverse()
    });
    let mut curvature_point_indices = vec![0, path.len() - 1];
    indices_sorted_by_curvature
        .into_iter()
        .filter(|&index| {
            for curvature_index in &curvature_point_indices {
                if curvature_index.abs_diff(index) < point_padding_per_side {
                    return false;
                }
            }
            curvature_point_indices.push(index);
            true
        })
        .chain(vec![0, path.len() - 1])
        .collect()
}

fn clip_pass(curve_paths: Vec<HermiteCurve>, d_sep: f32) -> Vec<HermiteCurve> {
    let d_sep_squared = d_sep * d_sep;
    (1..curve_paths.len())
        .flat_map(|curve_index| {
            let current_curve = &curve_paths[curve_index];
            let control_point_distances_squared: Vec<f32> = current_curve
                .iter()
                .map(|control_point| {
                    distance_squared_to_nearest_curve_approximation(
                        control_point.position,
                        &curve_paths[..curve_index],
                    )
                })
                .collect();
            let filtered: HermiteCurve = control_point_distances_squared
                .into_iter()
                .enumerate()
                .flat_map(|(i, dist_squared)| {
                    if dist_squared < d_sep_squared {
                        None
                    } else {
                        Some(current_curve[i])
                    }
                })
                .collect();

            if filtered.is_empty() {
                None
            } else {
                Some(filtered)
            }
        })
        .collect()
}

fn distance_squared_to_nearest_curve_approximation(
    point: Point,
    path_control_points: &[Vec<ControlPoint>],
) -> f32 {
    path_control_points
        .iter()
        .map(|path| {
            (0..path.len() - 1)
                .map(|start_index| {
                    let p_0 = path[start_index].position;
                    let p_1 = path[start_index + 1].position;
                    let segment_vector = p_1 - p_0;
                    let projection = project_vector(point - p_0, segment_vector);

                    (point - clamp_vector_between_points(p_0, p_1, p_0 + projection)).norm_squared()
                })
                .reduce(|acc, e| if e < acc { e } else { acc })
                .unwrap()
        })
        .reduce(|acc, e| if e < acc { e } else { acc })
        .unwrap()
}

fn project_vector(subject: Point, target: Vector2<f32>) -> Point {
    subject.dot(&target) / target.norm_squared() * target
}

fn evaluate_hermite_curve(
    p_0: Point,
    p_1: Point,
    m_0: Vector2<f32>,
    m_1: Vector2<f32>,
    t: f32,
) -> Point {
    t * (t * (t * (2.0 * (p_0 - p_1) + m_0 + m_1) - 2.0 * m_0 - m_1 - 3.0 * (p_0 - p_1)) + m_0)
        + p_0
}

fn spawn_new_points(
    control_point_curves: &[Vec<ControlPoint>],
    d_sep: f32,
    follow_major_eigenvectors: bool,
    min_curve_length: f32,
) -> Vec<Point> {
    control_point_curves
        .iter()
        .flat_map(|control_points| {
            let curve_length = approximated_hermite_length(control_points);

            if curve_length < min_curve_length {
                Vec::new()
            } else {
                vec![]
            }
        })
        .collect()
}

fn approximated_hermite_length(control_points: &[ControlPoint]) -> f32 {
    control_points[1..]
        .iter()
        .enumerate()
        .fold(0.0, |acc, (i, e)| {
            acc + (e.position - control_points[i - 1].position).norm()
        })
}
