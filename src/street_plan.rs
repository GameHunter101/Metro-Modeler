use core::f32;
use std::{
    collections::{BinaryHeap, HashSet},
    ops::Range,
};

use nalgebra::Vector2;
use rand::Rng;

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
) -> Vec<Vec<Point>> {
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

    let mut major_line_traces = Vec::new();
    let mut major_bounding_boxes: Vec<BoundingBox> = Vec::new();

    let mut minor_line_traces = Vec::new();
    let mut minor_bounding_boxes: Vec<BoundingBox> = Vec::new();

    let mut full_trace = Vec::new();

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
            horizontal_sector_count,
            vertical_sector_count,
            if follow_major_eigenvector {
                &major_bounding_boxes
            } else {
                &minor_bounding_boxes
            },
            false,
        );

        if trace_res.0.is_empty() {
            continue;
        }

        let trimmed_range = trim_trace(
            &trace_res.0,
            if follow_major_eigenvector {
                &major_bounding_boxes
            } else {
                &minor_bounding_boxes
            },
            0,
            trace_res.0.len() - 1,
            d_sep,
        );

        if trimmed_range.end - trimmed_range.start < 10 {
            continue;
        }

        let trace_path = trace_res.0[trimmed_range].to_vec();

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

        full_trace.push(trace_path.clone());

        if follow_major_eigenvector {
            major_line_traces.push(trace_path);
            major_bounding_boxes.extend(bounding_box);
        } else {
            minor_line_traces.push(trace_path);
            minor_bounding_boxes.extend(bounding_box);
        }
        for new_seed in trace_res.1 {
            seed_points.push(SeedPoint {
                seed: new_seed,
                priority: f32::MAX,
                follow_major_eigenvector: !follow_major_eigenvector,
            });
        }
    }

    futures::future::join_all(full_trace.into_iter().map(|trace| {
        tokio::spawn(async move {
            let first_pass = smooth_path(trace, 0.03, 0.3);
            resample_curve(&first_pass, 0.7, 10, 20, 0.2)
        })
    }))
    .await
    .into_iter()
    .collect::<Result<Vec<_>, _>>()
    .expect("Failed to run smoothing")
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

fn trace(
    tensor_field: &TensorField,
    seed: Point,
    h: f32,
    d_sep: f32,
    follow_major_eigenvectors: bool,
    max_len: f32,
    horizontal_sector_count: u32,
    vertical_sector_count: u32,
    bounding_boxes: &[BoundingBox],
    reverse: bool,
) -> (Vec<Point>, Vec<Point>, HashSet<(u32, u32)>) {
    for bounding_box in bounding_boxes {
        if bounding_box.check_point_collision_with_padding(seed, d_sep) {
            return (Vec::new(), Vec::new(), HashSet::new());
        }
    }
    let origin = seed;
    let mut seed = seed;
    let mut trace_res = vec![seed];
    let mut accumulated_distance = 0.0;
    let mut new_seeds: Vec<Point> = Vec::new();
    let mut prev_direction: Vector2<f32> = Vector2::zeros();
    let mut steps = 0;

    let mut sectors: HashSet<(u32, u32)> = HashSet::new();

    let horizontal_sector_size = GRID_SIZE / horizontal_sector_count;
    let vertical_sector_size = GRID_SIZE / vertical_sector_count;

    while !(seed.x < 0.0 || seed.y < 0.0 || seed.x > GRID_SIZE as f32 || seed.y > GRID_SIZE as f32)
    {
        for bounding_box in bounding_boxes {
            if bounding_box.check_point_collision_with_padding(seed, d_sep / 2.0) {
                break;
            }
        }
        sectors.insert((
            (seed.x / horizontal_sector_size as f32) as u32,
            (seed.y / vertical_sector_size as f32) as u32,
        ));

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
            * ((follow_major_eigenvectors as i32) as f32 * k_1_eigenvectors.major
                + (1 - follow_major_eigenvectors as i32) as f32 * k_1_eigenvectors.minor)
                .normalize();
        let k_2_eigenvectors = tensor_field
            .evaluate_smoothed_field_at_point(clamp_vec_to_grid(seed + h / 2.0 * k_1))
            .eigenvectors();
        let k_2 = rev_factor
            * ((follow_major_eigenvectors as i32) as f32 * k_2_eigenvectors.major
                + (1 - follow_major_eigenvectors as i32) as f32 * k_2_eigenvectors.minor)
                .normalize();
        let k_3_eigenvectors = tensor_field
            .evaluate_smoothed_field_at_point(clamp_vec_to_grid(seed + h / 2.0 * k_2))
            .eigenvectors();
        let k_3 = rev_factor
            * ((follow_major_eigenvectors as i32) as f32 * k_3_eigenvectors.major
                + (1 - follow_major_eigenvectors as i32) as f32 * k_3_eigenvectors.minor)
                .normalize();
        let k_4_eigenvectors = tensor_field
            .evaluate_smoothed_field_at_point(clamp_vec_to_grid(seed + h * k_3))
            .eigenvectors();
        let k_4 = rev_factor
            * ((follow_major_eigenvectors as i32) as f32 * k_4_eigenvectors.major
                + (1 - follow_major_eigenvectors as i32) as f32 * k_4_eigenvectors.minor)
                .normalize();

        let mut m = 1.0 / 6.0 * k_1 + 1.0 / 3.0 * k_2 + 1.0 / 3.0 * k_3 + 1.0 / 6.0 * k_4;

        // Trying to smooth out erratic paths
        if m.dot(&prev_direction).abs() < 0.5 {
            let x_axis = Vector2::x_axis();
            let angle_away = m.dot(&prev_direction).acos();
            let angle_diff = angle_away - std::f32::consts::FRAC_PI_3 * 4.0 / 3.0;
            let direction_flip = if prev_direction.y >= 0.0 { 1.0 } else { -1.0 };
            let signed_angle_diff = direction_flip
                * if m.dot(&x_axis) < prev_direction.dot(&x_axis) {
                    angle_diff
                } else {
                    -angle_diff
                };
            m = nalgebra::Rotation2::new(signed_angle_diff) * m;
        }

        let new_pos = seed + h * m;

        accumulated_distance += (new_pos - seed).norm();
        if accumulated_distance >= d_sep {
            new_seeds.push(new_pos);
            accumulated_distance = 0.0;
        }
        seed = new_pos;

        if h < 1.0 {
            if steps as f32 % (1.0 / h) <= 0.001 {
                trace_res.push(new_pos);
            }
        }
        steps += 1;

        if (new_pos - origin).magnitude_squared() <= 0.0001 || accumulated_distance > max_len {
            break;
        }

        prev_direction = m;
    }

    (trace_res, new_seeds, sectors)
}

fn clamp_vec_to_grid(vec: Vector2<f32>) -> Vector2<f32> {
    let clamped_x = vec.x.clamp(0.0, GRID_SIZE as f32);
    let clamped_y = vec.y.clamp(0.0, GRID_SIZE as f32);
    Vector2::new(
        if clamped_x.is_nan() { 1.0 } else { clamped_x },
        if clamped_y.is_nan() { 1.0 } else { clamped_y },
    )
}

fn trim_trace(
    trace: &[Point],
    bounding_boxes: &[BoundingBox],
    low: usize,
    high: usize,
    d_sep: f32,
) -> Range<usize> {
    let low_check = check_point_in_padded_bounding_boxes(trace[low], bounding_boxes, d_sep);
    let high_check = check_point_in_padded_bounding_boxes(trace[high], bounding_boxes, d_sep);

    let mid = (high - low) / 2;

    if !low_check && !high_check {
        Range {
            start: low,
            end: high,
        }
    } else if low_check && high_check {
        Range { start: 0, end: 0 }
    } else if low_check && !high_check {
        trim_trace(trace, bounding_boxes, mid + 1, high, d_sep)
    } else {
        trim_trace(trace, bounding_boxes, low, mid, d_sep)
    }
}

fn check_point_in_padded_bounding_boxes(
    point: Point,
    bounding_boxes: &[BoundingBox],
    padding: f32,
) -> bool {
    bounding_boxes
        .iter()
        .map(|b| b.check_point_collision_with_padding(point, padding))
        .any(|x| x)
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

pub fn resample_curve(
    path: &[Point],
    blend_factor: f32,
    points_per_spline: i32,
    point_padding_per_side: usize,
    h: f32,
) -> Vec<Point> {
    let mut control_points: Vec<usize> = highest_curvature_points(&path, point_padding_per_side);

    control_points.sort();

    let sampler = |p_0: Point, p_1: Point, m_0: Vector2<f32>, m_1: Vector2<f32>| -> Vec<Point> {
        (0..=points_per_spline)
            .map(|i| {
                evaluate_hermite_curve(p_0, p_1, m_0, m_1, i as f32 / points_per_spline as f32)
            })
            .collect()
    };

    let h_square = h * h;

    control_points[..control_points.len() - 1]
        .iter()
        .enumerate()
        .flat_map(|(i, &control_point_index)| {
            let p_0 = path[control_point_index];
            if i == 0 {
                let p_1 = path[control_points[1]];
                let m_0 = (path[1] - p_0) / h_square;
                let m_1 = if control_points.len() == 2 {
                    (path[path.len() - 1] - path[path.len() - 2]) / h_square
                } else {
                    blend_factor
                        * ((path[control_points[1] + 1] - p_1) / h_square
                            + (p_1 - path[control_points[1] - 1]) / h_square)
                };
                let res = sampler(p_0, p_1, m_0, m_1);
                res[..res.len() - 1].to_vec()
            } else if i == control_points.len() - 2 {
                let p_1 = path[path.len() - 1];
                let m_0 = blend_factor
                    * ((path[control_point_index + 1] - p_0) / h_square
                        + (p_0 - path[control_point_index - 1]) / h_square);
                let m_1 = (path[path.len() - 1] - path[path.len() - 2]) / h_square;
                sampler(p_0, p_1, m_0, m_1)[1..].to_vec()
            } else {
                let p_1 = path[control_points[i + 1]];
                let m_0 = blend_factor
                    * ((path[control_point_index + 1] - p_0) / h_square
                        + (p_0 - path[control_point_index - 1]) / h_square);
                let m_1 = blend_factor
                    * ((path[control_points[i + 1] + 1] - p_1) / h_square
                        + (p_1 - path[control_points[i + 1] - 1]) / h_square);
                let res = sampler(p_0, p_1, m_0, m_1);
                res[..res.len()].to_vec()
            }
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

pub fn evaluate_hermite_curve(
    p_0: Point,
    p_1: Point,
    m_0: Vector2<f32>,
    m_1: Vector2<f32>,
    t: f32,
) -> Point {
    t * (t * (t * (2.0 * (p_0 - p_1) + m_0 + m_1) - 2.0 * m_0 - m_1 - 3.0 * (p_0 - p_1)) + m_0)
        + p_0
}
