use std::collections::{HashMap, HashSet};

use cool_utils::data_structures::quadtree::Quadtree;
use nalgebra::Vector2;
use ordered_float::OrderedFloat;

use crate::{
    Point,
    street_graph::{IntersectionPoint, Segment, calc_intersection_point_semi_bounded},
};

struct AABB {
    top_right: Point,
    bottom_left: Point,
    left: Option<Box<AABB>>,
    right: Option<Box<AABB>>,
    segment: Option<(usize, Segment)>,
}

impl AABB {
    fn new(
        top_right: Point,
        bottom_left: Point,
        left: Option<Box<AABB>>,
        right: Option<Box<AABB>>,
        curve: Option<(usize, Segment)>,
    ) -> AABB {
        AABB {
            top_right,
            bottom_left,
            left,
            right,
            segment: curve,
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

    fn path_intersection(&self, other: &AABB) -> Option<Point> {
        calc_intersection_point_semi_bounded(self.segment.unwrap().1, other.segment.unwrap().1)
    }

    fn is_leaf(&self) -> bool {
        self.segment.is_some()
    }
}

fn construct_bvh_tree(segments: &[Segment]) -> AABB {
    let dimensions: Vec<(usize, [Point; 2])> = segments
        .iter()
        .enumerate()
        .map(|(i, segment)| {
            let top_right = Point::new(
                segment[0].x.max(segment[1].x),
                segment[0].y.max(segment[1].y),
            );

            let bottom_left = Point::new(
                segment[0].x.min(segment[1].x),
                segment[0].y.min(segment[1].y),
            );

            (i, [top_right, bottom_left])
        })
        .collect();

    construct_bounding_boxes(segments, &dimensions)
}

fn construct_bounding_boxes<'a>(curves: &[Segment], dimensions: &[(usize, [Point; 2])]) -> AABB {
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
            Some((dimensions[0].0, curves[dimensions[0].0].clone())),
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
            None,
        )
    }
}

fn get_nearest_intersection_point<'a, const N: usize>(
    target: Point,
    all_intersection_points: &mut Quadtree<N>,
    intersections: &'a mut Intersections,
    threshold: f32,
) -> &'a mut Vec<usize> {
    if let Some(merging_point) =
        all_intersection_points.get_point_within_distance(target, threshold.sqrt())
    {
        intersections
            .get_mut(&Vector2::new(
                OrderedFloat(merging_point.x),
                OrderedFloat(merging_point.y),
            ))
            .unwrap()
    } else {
        let ordered_target = Vector2::new(OrderedFloat(target.x), OrderedFloat(target.y));
        all_intersection_points.insert(target);
        intersections.insert(ordered_target, Vec::new());
        intersections.get_mut(&ordered_target).unwrap()
    }
}

pub type Intersections = HashMap<Vector2<OrderedFloat<f32>>, Vec<usize>>;

fn find_bvh_intersections_helper<const N: usize>(
    left: &AABB,
    right: &AABB,
    checked_pairs: &mut HashSet<(usize, usize)>,
    all_intersection_points: &mut Quadtree<N>,
    intersections: &mut Intersections,
    merging_threshold: f32,
) {
    if !left.box_intersection(right) {
        return;
    }

    if let Some(left_segment) = left.segment
        && let Some(right_segment) = right.segment
    {
        let current_pair = (
            left_segment.0.min(right_segment.0),
            left_segment.0.max(right_segment.0),
        );
        if !checked_pairs.contains(&current_pair) {
            checked_pairs.insert(current_pair);
            if let Some(intersection_point) = left.path_intersection(right) {
                let intersecting_segments = get_nearest_intersection_point(
                    intersection_point,
                    all_intersection_points,
                    intersections,
                    merging_threshold,
                );
                let intersecting_segments_set: HashSet<usize> =
                    HashSet::from_iter(intersecting_segments.clone());
                if !intersecting_segments_set.contains(&left_segment.0) {
                    intersecting_segments.push(left_segment.0);
                }
                if !intersecting_segments_set.contains(&right_segment.0) {
                    intersecting_segments.push(right_segment.0);
                }
            }
        }
    } else {
        let intersection_test_pairs: Vec<[&AABB; 2]> = if left.is_leaf() {
            vec![
                [left, right.left.as_ref().unwrap()],
                [left, right.right.as_ref().unwrap()],
            ]
        } else if right.is_leaf() {
            vec![
                [left.left.as_ref().unwrap(), right],
                [left.right.as_ref().unwrap(), right],
            ]
        } else {
            vec![
                [left.left.as_ref().unwrap(), right.left.as_ref().unwrap()],
                [left.left.as_ref().unwrap(), right.right.as_ref().unwrap()],
                [left.right.as_ref().unwrap(), right.left.as_ref().unwrap()],
                [left.right.as_ref().unwrap(), right.right.as_ref().unwrap()],
            ]
        };

        for pair in intersection_test_pairs {
            find_bvh_intersections_helper(
                pair[0],
                pair[1],
                checked_pairs,
                all_intersection_points,
                intersections,
                merging_threshold,
            );
        }
    }
}

pub fn find_segments_intersections(
    segments: &[Segment],
    merging_threshold: f32,
    include_all_endpoints: bool,
) -> Vec<IntersectionPoint> {
    let bvh = construct_bvh_tree(segments);
    let mut checked_pairs: HashSet<(usize, usize)> = (0..segments.len()).map(|i| (i, i)).collect();
    let mut all_intersection_points: Quadtree<4> = Quadtree::new(
        (bvh.bottom_left + bvh.top_right) / 2.0,
        (bvh.top_right - bvh.bottom_left).max(),
    );
    let mut intersections: HashMap<Vector2<OrderedFloat<f32>>, Vec<usize>> = HashMap::new();

    if include_all_endpoints {
        for (i, segment) in segments.iter().enumerate() {
            let ordered_point_0 =
                Vector2::new(OrderedFloat(segment[0].x), OrderedFloat(segment[0].y));
            let ordered_point_1 =
                Vector2::new(OrderedFloat(segment[1].x), OrderedFloat(segment[1].y));
            all_intersection_points.insert(segment[0]);
            all_intersection_points.insert(segment[1]);

            if let Some(point_0_intersections) = intersections.get_mut(&ordered_point_0) {
                point_0_intersections.push(i);
            } else {
                intersections.insert(ordered_point_0, vec![i]);
            }
            if let Some(point_1_intersections) = intersections.get_mut(&ordered_point_1) {
                point_1_intersections.push(i);
            } else {
                intersections.insert(ordered_point_1, vec![i]);
            }
        }
    }

    find_bvh_intersections_helper(
        &bvh,
        &bvh,
        &mut checked_pairs,
        &mut all_intersection_points,
        &mut intersections,
        merging_threshold,
    );

    intersections
        .into_iter()
        .map(
            |(position, intersecting_segment_indices)| IntersectionPoint {
                position: Point::new(*position.x, *position.y),
                intersecting_segment_indices,
            },
        )
        .collect()
}

#[allow(unused)]
pub fn assert_intersections_eq(test: &[IntersectionPoint], expected: &[IntersectionPoint]) {
    assert_eq!(test.len(), expected.len());
    for test_intersection in test {
        assert!(expected.iter().any(|expected_intersection| {
            crate::street_graph::points_are_close(
                test_intersection.position(),
                expected_intersection.position(),
            ) && HashSet::<usize>::from_iter(test_intersection.intersecting_segment_indices.clone())
                .difference(&HashSet::from_iter(
                    expected_intersection.intersecting_segment_indices.clone(),
                ))
                .count()
                == 0
        }));
    }
}

#[cfg(test)]
mod test {

    use crate::intersections::assert_intersections_eq;
    use crate::{
        intersections::{AABB, find_segments_intersections},
        street_graph::IntersectionPoint,
        tensor_field::Point,
    };

    #[test]
    fn box_tr_bl_intersection() {
        let a = AABB::new(Point::new(2.0, 2.0), Point::new(1.0, 1.0), None, None, None);
        let b = AABB::new(Point::new(3.0, 3.0), Point::new(1.5, 1.5), None, None, None);
        assert!(a.box_intersection(&b));
        assert!(b.box_intersection(&a));
    }

    #[test]
    fn box_tl_br_intersection() {
        let a = AABB::new(Point::new(2.0, 2.0), Point::new(1.0, 1.0), None, None, None);
        let b = AABB::new(Point::new(1.5, 2.5), Point::new(0.5, 1.5), None, None, None);
        assert!(a.box_intersection(&b));
        assert!(b.box_intersection(&a));
    }

    #[test]
    fn box_enveloped_intersection() {
        let a = AABB::new(Point::new(2.0, 2.0), Point::new(1.0, 1.0), None, None, None);
        let b = AABB::new(Point::new(1.5, 1.5), Point::new(1.1, 1.1), None, None, None);
        assert!(a.box_intersection(&b));
        assert!(b.box_intersection(&a));
    }

    #[test]
    fn box_side_intersection() {
        let a = AABB::new(Point::new(2.0, 2.0), Point::new(1.0, 1.0), None, None, None);
        let b = AABB::new(Point::new(3.0, 3.0), Point::new(1.5, 0.0), None, None, None);
        assert!(a.box_intersection(&b));
        assert!(b.box_intersection(&a));
    }

    #[test]
    fn box_no_thickness_intersection() {
        let a = AABB::new(Point::new(2.0, 2.0), Point::new(1.0, 2.0), None, None, None);
        let b = AABB::new(Point::new(1.5, 3.0), Point::new(1.5, 0.0), None, None, None);
        assert!(a.box_intersection(&b));
        assert!(b.box_intersection(&a));
    }

    #[test]
    fn simple_bvh_intersection() {
        let segments = vec![
            [Point::new(2.0, 0.0), Point::new(0.466, 3.273)],
            [Point::new(0.0, 1.0), Point::new(1.817, 1.44)],
        ];

        let intersections = find_segments_intersections(&segments, 0.0001, false);
        let expected_intersections = vec![IntersectionPoint {
            position: Point::new(1.3752345, 1.3330231),
            intersecting_segment_indices: vec![0, 1],
        }];
        assert_intersections_eq(&intersections, &expected_intersections);
    }

    #[test]
    fn bvh_triple_intersection() {
        let segments = vec![
            [Point::new(2.0, 0.0), Point::new(0.466, 3.273)],
            [Point::new(0.0, 1.0), Point::new(1.817, 1.44)],
            [Point::new(1.3752345, 2.0), Point::new(1.3752345, 0.0)],
        ];

        let intersections = find_segments_intersections(&segments, 0.0001, false);
        let expected_intersections = vec![IntersectionPoint {
            position: Point::new(1.3752345, 1.3330231),
            intersecting_segment_indices: vec![0, 1, 2],
        }];
        assert_intersections_eq(&intersections, &expected_intersections);
    }

    #[test]
    fn chained_segment_bvh_intersections() {
        let segments = vec![
            [Point::new(0.97, 0.58), Point::new(2.52, 1.66)],
            [Point::new(2.52, 1.66), Point::new(3.89, 0.38)],
            [Point::new(1.07, 1.28), Point::new(4.94, 0.56)],
        ];

        let intersections = find_segments_intersections(&segments, 0.0001, true);
        let expected_intersections = vec![
            IntersectionPoint {
                position: Point::new(1.78399, 1.14717),
                intersecting_segment_indices: vec![0, 2],
            },
            IntersectionPoint {
                position: Point::new(3.38837, 0.84868),
                intersecting_segment_indices: vec![1, 2],
            },
            IntersectionPoint {
                position: Point::new(2.52, 1.66),
                intersecting_segment_indices: vec![0, 1],
            },
            IntersectionPoint {
                position: Point::new(0.97, 0.58),
                intersecting_segment_indices: vec![0],
            },
            IntersectionPoint {
                position: Point::new(3.89, 0.38),
                intersecting_segment_indices: vec![1],
            },
            IntersectionPoint {
                position: Point::new(1.07, 1.28),
                intersecting_segment_indices: vec![2],
            },
            IntersectionPoint {
                position: Point::new(4.94, 0.56),
                intersecting_segment_indices: vec![2],
            },
        ];

        assert_intersections_eq(&intersections, &expected_intersections);
    }

    #[test]
    fn bvh_intersection_tolerance() {
        let segments = vec![
            [
                Point::new(84.59501, 327.1706),
                Point::new(69.01391, 312.52502),
            ],
            [
                Point::new(106.17567, 303.3888),
                Point::new(80.89961, 331.2429),
            ],
        ];

        let intersections = find_segments_intersections(&segments, 0.0001, true);
        let expected_intersections = vec![
            IntersectionPoint {
                position: Point::new(69.01391, 312.52502),
                intersecting_segment_indices: vec![0],
            },
            IntersectionPoint {
                position: Point::new(106.17567, 303.3888),
                intersecting_segment_indices: vec![1],
            },
            IntersectionPoint {
                position: Point::new(80.89961, 331.2429),
                intersecting_segment_indices: vec![1],
            },
            IntersectionPoint {
                position: Point::new(84.59501, 327.1706),
                intersecting_segment_indices: vec![0, 1],
            },
        ];
        assert_intersections_eq(&intersections, &expected_intersections);
    }

    #[test]
    fn bvh_intersection_tolerance_2() {
        let segments = vec![
            [
                Point::new(62.39522, 313.1381),
                Point::new(91.93122, 288.34058),
            ],
            [
                Point::new(84.14339, 278.44148),
                Point::new(97.12109, 294.93295),
            ],
        ];

        let intersections = find_segments_intersections(&segments, 0.0001, true);
        let expected_intersections = vec![
            IntersectionPoint {
                position: Point::new(62.39522, 313.1381),
                intersecting_segment_indices: vec![0],
            },
            IntersectionPoint {
                position: Point::new(84.14339, 278.44148),
                intersecting_segment_indices: vec![1],
            },
            IntersectionPoint {
                position: Point::new(97.12109, 294.93295),
                intersecting_segment_indices: vec![1],
            },
            IntersectionPoint {
                position: Point::new(91.93122, 288.34058),
                intersecting_segment_indices: vec![0, 1],
            },
        ];
        assert_intersections_eq(&intersections, &expected_intersections);
    }
}
