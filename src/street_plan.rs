use core::f32;
use std::collections::{BinaryHeap, HashMap, HashSet};

use nalgebra::Vector2;
use rand::Rng;

use crate::tensor_field::{EvalEigenvectors, GRID_SIZE, Point, TensorField};

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

pub fn trace_street_plan(
    tensor_field: &TensorField,
    starting_seed_count: u32,
    city_center: Point,
    horizontal_sector_count: u32,
    vertical_sector_count: u32,
) -> Vec<Vec<Point>> {
    let mut seed_points = prioritize_points(
        &distribute_points(starting_seed_count),
        city_center,
        &tensor_field,
    );

    let mut major_line_traces = Vec::new();
    let mut major_line_sectors: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
    (0..horizontal_sector_count).for_each(|x| {
        (0..vertical_sector_count).for_each(|y| {
            major_line_sectors.insert((x, y), Vec::new());
        });
    });

    let mut minor_line_traces = Vec::new();
    let mut minor_line_sectors: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
    (0..horizontal_sector_count).for_each(|x| {
        (0..vertical_sector_count).for_each(|y| {
            minor_line_sectors.insert((x, y), Vec::new());
        });
    });

    while let Some(SeedPoint {
        seed,
        follow_major_eigenvector,
        ..
    }) = seed_points.pop()
    {
        let trace_res = trace(
            tensor_field,
            seed,
            0.2,
            20.0, // + 0.7 * (seed - city_center).norm(),
            follow_major_eigenvector,
            200.0,
            horizontal_sector_count,
            vertical_sector_count,
            if follow_major_eigenvector {
                &major_line_sectors
            } else {
                &minor_line_sectors
            },
            if follow_major_eigenvector {
                &major_line_traces
            } else {
                &minor_line_traces
            },
            if follow_major_eigenvector {
                &minor_line_sectors
            } else {
                &major_line_sectors
            },
            if follow_major_eigenvector {
                &minor_line_traces
            } else {
                &major_line_traces
            },
            false,
        );

        if follow_major_eigenvector {
            let index = major_line_traces.len();
            major_line_traces.push(trace_res.0);
            for sector_coord in trace_res.2 {
                major_line_sectors
                    .get_mut(&sector_coord)
                    .unwrap()
                    .push(index);
            }
        } else {
            let index = minor_line_traces.len();
            minor_line_traces.push(trace_res.0);
            for sector_coord in trace_res.2 {
                minor_line_sectors
                    .get_mut(&sector_coord)
                    .unwrap()
                    .push(index);
            }
        }
        for new_seed in trace_res.1 {
            seed_points.push(SeedPoint {
                seed: new_seed,
                priority: f32::MAX,
                follow_major_eigenvector: !follow_major_eigenvector,
            });
        }
    }

    major_line_traces.append(&mut minor_line_traces);

    major_line_traces
}

pub fn trace(
    tensor_field: &TensorField,
    seed: Point,
    h: f32,
    d_sep: f32,
    follow_major_eigenvectors: bool,
    max_len: f32,
    horizontal_sector_count: u32,
    vertical_sector_count: u32,
    sector_map: &HashMap<(u32, u32), Vec<usize>>,
    all_traces: &[Vec<Point>],
    opposite_family_sector_map: &HashMap<(u32, u32), Vec<usize>>,
    opposite_family_traces: &[Vec<Point>],
    reverse: bool,
) -> (Vec<Point>, Vec<Point>, HashSet<(u32, u32)>) {
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
        if new_seeds.is_empty() && accumulated_distance >= d_sep {
            new_seeds.push(new_pos);
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
        if point_is_too_close_to_others(
            new_pos,
            horizontal_sector_count,
            vertical_sector_count,
            sector_map,
            all_traces,
            d_sep,
        )
        .is_some()
        {
            if new_seeds.is_empty() && (new_pos - origin).norm_squared() > 5.0 {
                new_seeds.push(new_pos);
            }
            if !reverse {
                let (mut trace_res_rev, new_seeds_rev, sectors_rev) = trace(
                    tensor_field,
                    origin,
                    h,
                    d_sep,
                    follow_major_eigenvectors,
                    max_len / 2.0,
                    horizontal_sector_count,
                    vertical_sector_count,
                    sector_map,
                    all_traces,
                    opposite_family_sector_map,
                    opposite_family_traces,
                    true,
                );

                trace_res_rev.reverse();
                trace_res_rev.extend(trace_res);
                trace_res = trace_res_rev;
                new_seeds.extend(new_seeds_rev);
                sectors.extend(sectors_rev);
            }
            break;
        }
        prev_direction = m;
    }

    if !reverse {
        if let Some(closest_point) = point_is_too_close_to_others(
            seed,
            horizontal_sector_count,
            vertical_sector_count,
            &opposite_family_sector_map
                .iter()
                .chain(sector_map.iter())
                .map(|(a, b)| (a.clone(), b.clone()))
                .collect(),
            &opposite_family_traces
                .into_iter()
                .chain(all_traces.into_iter())
                .cloned()
                .collect::<Vec<_>>(),
            d_sep / 2.0,
        ) {
            trace_res.push(closest_point);
        }
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

fn point_is_too_close_to_others(
    point: Point,
    horiztonal_sector_count: u32,
    vertical_sector_count: u32,
    sector_map: &HashMap<(u32, u32), Vec<usize>>,
    all_traces: &[Vec<Point>],
    d_sep: f32,
) -> Option<Point> {
    let horizontal_sector_size = GRID_SIZE / horiztonal_sector_count;
    let vertical_sector_size = GRID_SIZE / vertical_sector_count;
    let mut sorted_sectors: Vec<(u32, u32)> = (0..horiztonal_sector_count)
        .flat_map(|x| {
            (0..vertical_sector_count)
                .map(|y| (x, y))
                .collect::<Vec<_>>()
        })
        .collect();

    sorted_sectors.sort_by(|a, b| {
        let a = Point::new(a.0 as f32, a.1 as f32);
        let b = Point::new(b.0 as f32, b.1 as f32);

        let a_center = Point::new(
            (a.x + 0.5) * horizontal_sector_size as f32,
            (a.y + 0.5) * vertical_sector_size as f32,
        );

        let b_center = Point::new(
            (b.x + 0.5) * horizontal_sector_size as f32,
            (b.y + 0.5) * vertical_sector_size as f32,
        );

        (a_center - point)
            .norm_squared()
            .total_cmp(&(b_center - point).norm_squared())
    });

    for sector in sorted_sectors {
        if let Some(closest_point) =
            point_is_too_close_to_others_in_sector(point, &sector, sector_map, all_traces, d_sep)
        {
            return Some(closest_point);
        }
    }

    return None;
}

fn point_is_too_close_to_others_in_sector(
    point: Point,
    sector: &(u32, u32),
    sector_map: &HashMap<(u32, u32), Vec<usize>>,
    all_traces: &[Vec<Point>],
    d_sep: f32,
) -> Option<Point> {
    let d_sep_squared = d_sep * d_sep;
    let mut temp: Vec<Point> = sector_map[sector]
        .iter()
        .flat_map(|trace_index| {
            let trace = &all_traces[*trace_index];
            let eval_points = [
                trace.first().unwrap(),
                trace.last().unwrap(),
                &trace[trace.len() / 2],
            ];

            let min_dist_point = eval_points.into_iter().map(|p| p - point).fold(
                Point::new(f32::MAX, f32::MAX),
                |acc, e| {
                    if e.norm_squared() < acc.norm_squared() {
                        e
                    } else {
                        acc
                    }
                },
            );

            if min_dist_point.norm_squared() <= d_sep_squared {
                return Some(min_dist_point + point);
            }
            /* for node in &all_traces[*trace_index] {
                if (node - point).norm_squared() <= d_sep * d_sep {
                    return true;
                }
            } */
            return None;
        })
        .collect();

    temp.sort_by(|a, b| {
        (a - point)
            .norm_squared()
            .total_cmp(&(b - point).norm_squared())
    });

    temp.first().cloned()
}
