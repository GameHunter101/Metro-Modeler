use std::collections::HashSet;

use crate::{
    Point,
    street_graph::{IntersectionPoint, calc_intersection_point},
    street_plan::{HermiteCurve, resample_curve},
};

struct AABB {
    top_right: Point,
    bottom_left: Point,
    left: Option<Box<AABB>>,
    right: Option<Box<AABB>>,
    curve: (usize, HermiteCurve),
}

impl AABB {
    fn new(
        top_right: Point,
        bottom_left: Point,
        left: Option<Box<AABB>>,
        right: Option<Box<AABB>>,
        curve: (usize, HermiteCurve),
    ) -> AABB {
        AABB {
            top_right,
            bottom_left,
            left,
            right,
            curve,
        }
    }

    fn box_intersection(&self, other: &AABB) -> bool {
        let x_overlap = (self.top_right.x >= other.top_right.x
            && other.top_right.x >= self.bottom_left.x)
            || (other.top_right.x >= self.top_right.x && self.top_right.x >= other.bottom_left.x);
        let y_overlap = (self.top_right.y >= other.top_right.y
            && other.top_right.y >= self.bottom_left.y)
            || (other.top_right.y >= self.top_right.y && self.top_right.y >= other.bottom_left.y);

        x_overlap && y_overlap
    }

    fn path_intersection(&self, other: &AABB, resolution: i32) -> Vec<IntersectionPoint> {
        let self_samples = resample_curve(&self.curve.1, resolution);
        let other_samples = resample_curve(&other.curve.1, resolution);

        (0..self_samples.len() - 1)
            .flat_map(|segment_index| {
                let self_segment = [self_samples[segment_index], self_samples[segment_index + 1]];

                (0..other_samples.len() - 1)
                    .flat_map(|other_segment_index| {
                        let other_segment = [
                            other_samples[other_segment_index],
                            other_samples[other_segment_index + 1],
                        ];
                        calc_intersection_point(self_segment, other_segment).map(|point| {
                            IntersectionPoint {
                                position: point,
                                intersecting_segment_indices: vec![self.curve.0, other.curve.0],
                            }
                        })
                    })
                    .collect::<Vec<IntersectionPoint>>()
            })
            .collect()
    }

    fn is_leaf(&self) -> bool {
        !self.curve.1.is_empty()
    }
}

fn construct_bvh_tree(curves: &[HermiteCurve]) -> AABB {
    let dimensions: Vec<(usize, [Point; 2])> = curves
        .iter()
        .enumerate()
        .map(|(i, curve)| {
            let samples = resample_curve(curve, 5);

            let top_right = Point::new(
                samples.iter().max_by(|a, b| a.x.total_cmp(&b.x)).unwrap().x,
                samples.iter().max_by(|a, b| a.y.total_cmp(&b.y)).unwrap().y,
            );

            let bottom_left = Point::new(
                samples.iter().min_by(|a, b| a.x.total_cmp(&b.x)).unwrap().x,
                samples.iter().min_by(|a, b| a.y.total_cmp(&b.y)).unwrap().y,
            );

            (i, [top_right, bottom_left])
        })
        .collect();

    construct_bounding_boxes(curves, &dimensions)
}

fn construct_bounding_boxes<'a>(
    curves: &[HermiteCurve],
    dimensions: &[(usize, [Point; 2])],
) -> AABB {
    let top_right = Point::new(
        dimensions
            .iter()
            .max_by(|a, b| a.1[0].x.total_cmp(&b.1[0].x))
            .unwrap()
            .1[0]
            .x,
        dimensions
            .iter()
            .max_by(|a, b| a.1[0].y.total_cmp(&b.1[0].y))
            .unwrap()
            .1[0]
            .y,
    );

    let bottom_left = Point::new(
        dimensions
            .iter()
            .min_by(|a, b| a.1[1].x.total_cmp(&b.1[1].x))
            .unwrap()
            .1[1]
            .x,
        dimensions
            .iter()
            .min_by(|a, b| a.1[1].y.total_cmp(&b.1[1].y))
            .unwrap()
            .1[1]
            .y,
    );

    if dimensions.len() == 1 {
        AABB::new(
            top_right,
            bottom_left,
            None,
            None,
            (dimensions[0].0, curves[dimensions[0].0].clone()),
        )
    } else {
        let is_x_length_larger = top_right.x - bottom_left.x > top_right.y - bottom_left.y;
        let mut sorted = dimensions.to_vec();
        sorted.sort_by(|a, b| {
            let a_center = (a.1[0] + a.1[1]) / 2.0;
            let b_center = (b.1[0] + b.1[1]) / 2.0;
            if is_x_length_larger {
                a_center.x.total_cmp(&b_center.x)
            } else {
                a_center.y.total_cmp(&b_center.y)
            }
        });

        let split = sorted.len() / 2;
        let left = construct_bounding_boxes(curves, &sorted[0..split]);
        let right = construct_bounding_boxes(curves, &sorted[split..]);
        AABB::new(
            top_right,
            bottom_left,
            Some(Box::new(left)),
            Some(Box::new(right)),
            (usize::MAX, Vec::new()),
        )
    }
}

fn find_bvh_intersections_helper(
    left: &AABB,
    right: &AABB,
    checked_pairs: &mut HashSet<(usize, usize)>,
) -> Vec<IntersectionPoint> {
    if !left.box_intersection(right) {
        return Vec::new();
    }

    if left.is_leaf() && right.is_leaf() {
        let current_pair = (
            left.curve.0.min(right.curve.0),
            left.curve.0.max(right.curve.0),
        );
        if !checked_pairs.contains(&current_pair) {
            checked_pairs.insert(current_pair);
            left.path_intersection(right, 10)
        } else {
            Vec::new()
        }
    } else if left.is_leaf() && !right.is_leaf() {
        find_bvh_intersections_helper(left, right.left.as_ref().unwrap(), checked_pairs)
            .into_iter()
            .chain(find_bvh_intersections_helper(
                left,
                right.right.as_ref().unwrap(),
                checked_pairs,
            ))
            .collect()
    } else if !left.is_leaf() && right.is_leaf() {
        find_bvh_intersections_helper(left.left.as_ref().unwrap(), right, checked_pairs)
            .into_iter()
            .chain(find_bvh_intersections_helper(
                left.right.as_ref().unwrap(),
                right,
                checked_pairs,
            ))
            .collect()
    } else {
        find_bvh_intersections_helper(
            left.left.as_ref().unwrap(),
            right.left.as_ref().unwrap(),
            checked_pairs,
        )
        .into_iter()
        .chain(find_bvh_intersections_helper(
            left.left.as_ref().unwrap(),
            right.right.as_ref().unwrap(),
            checked_pairs,
        ))
        .chain(find_bvh_intersections_helper(
            left.right.as_ref().unwrap(),
            right.left.as_ref().unwrap(),
            checked_pairs,
        ))
        .chain(find_bvh_intersections_helper(
            left.right.as_ref().unwrap(),
            right.right.as_ref().unwrap(),
            checked_pairs,
        ))
        .collect()
    }
}

fn find_intersections(curves: &[HermiteCurve]) -> Vec<IntersectionPoint> {
    let bvh = construct_bvh_tree(curves);
    let mut checked_pairs: HashSet<(usize, usize)> = (0..curves.len()).map(|i| (i, i)).collect();
    find_bvh_intersections_helper(&bvh, &bvh, &mut checked_pairs)
}

#[cfg(test)]
mod test {
    use std::collections::HashSet;

    use crate::{
        intersections::find_intersections, street_graph::points_are_close,
        street_plan::ControlPoint, tensor_field::Point,
    };

    #[test]
    fn simple_bvh_intersection() {
        let curves = vec![
            vec![
                ControlPoint {
                    position: Point::new(2.0, 0.0),
                    velocity: Point::new(1.15, 2.23),
                },
                ControlPoint {
                    position: Point::new(0.466, 3.273),
                    velocity: Point::new(1.066, 4.073),
                },
            ],
            vec![
                ControlPoint {
                    position: Point::new(0.0, 1.0),
                    velocity: Point::new(0.41, -1.165),
                },
                ControlPoint {
                    position: Point::new(1.817, 1.44),
                    velocity: Point::new(0.927, -0.61),
                },
            ],
        ];

        let intersections = find_intersections(&curves);
        assert_eq!(intersections.len(), 1);
        assert!(points_are_close(
            intersections[0].position(),
            Point::new(1.29047, 1.3516)
        ));
        assert_eq!(intersections[0].intersecting_segment_indices.len(), 2);
        assert_eq!(
            HashSet::from_iter(intersections[0].intersecting_segment_indices.clone())
                .difference(&HashSet::<usize>::from_iter(vec![0, 1]))
                .count(),
            0
        );
    }

    #[test]
    fn bvh_triple_intersection() {
        let curves = vec![
            vec![
                ControlPoint {
                    position: Point::new(2.0, 0.0),
                    velocity: Point::new(1.15, 2.23),
                },
                ControlPoint {
                    position: Point::new(0.466, 3.273),
                    velocity: Point::new(1.066, 4.073),
                },
            ],
            vec![
                ControlPoint {
                    position: Point::new(0.0, 1.0),
                    velocity: Point::new(0.41, -1.165),
                },
                ControlPoint {
                    position: Point::new(1.817, 1.44),
                    velocity: Point::new(0.927, -0.61),
                },
            ],
            vec![
                ControlPoint {
                    position: Point::new(1.29047, 2.0),
                    velocity: Point::new(0.0, -1.0),
                },
                ControlPoint {
                    position: Point::new(1.29047, 0.0),
                    velocity: Point::new(0.0, -1.0),
                },
            ]
        ];

        let intersections = find_intersections(&curves);
        assert_eq!(intersections.len(), 1);
        assert!(points_are_close(
            intersections[0].position(),
            Point::new(1.29047, 1.3516)
        ));
        assert_eq!(intersections[0].intersecting_segment_indices.len(), 3);
        assert_eq!(
            HashSet::from_iter(intersections[0].intersecting_segment_indices.clone())
                .difference(&HashSet::<usize>::from_iter(vec![0, 1, 2]))
                .count(),
            0
        );
    }
}
