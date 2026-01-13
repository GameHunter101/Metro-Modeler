use std::collections::{HashMap, HashSet};

use cool_utils::data_structures::rbtree::RBTree;
use nalgebra::Vector2;
use ordered_float::OrderedFloat;

use crate::{
    Point,
    street_graph::{Segment, calc_intersection_point_with_tolerance},
    street_plan::{HermiteCurve, resample_curve},
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
        println!(
            "Self: {:?}, other: {:?}",
            self.segment.unwrap(),
            other.segment.unwrap()
        );
        let temp = calc_intersection_point_with_tolerance(
            self.segment.unwrap().1,
            other.segment.unwrap().1,
            0.001,
        );
        println!("Intersection: {temp:?}");
        temp
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

fn get_nearest_intersection_point<'a>(
    target: Point,
    x_search: &'a mut RBTree<OrderedFloat<f32>>,
    y_search: &'a mut RBTree<OrderedFloat<f32>>,
    intersections: &'a mut HashMap<Vector2<OrderedFloat<f32>>, Vec<usize>>,
    threshold: f32,
) -> &'a mut Vec<usize> {
    let ordered_target = Vector2::new(OrderedFloat(target.x), OrderedFloat(target.y));
    if let Some(x_res) = x_search.get_nearest(&ordered_target.x)
        && let Some(y_res) = y_search.get_nearest(&ordered_target.y)
        && intersections.contains_key(&Vector2::new(*x_res, *y_res))
        && (Point::new(**x_res, **y_res) - &target).norm_squared() <= threshold
    {
        intersections
            .get_mut(&Vector2::new(*x_res, *y_res))
            .unwrap()
    } else {
        intersections.insert(ordered_target, Vec::new());
        x_search.insert(ordered_target.x);
        y_search.insert(ordered_target.y);
        intersections.get_mut(&ordered_target).unwrap()
    }
}

pub type Intersections = HashMap<Vector2<OrderedFloat<f32>>, Vec<usize>>;

fn find_bvh_intersections_helper(
    left: &AABB,
    right: &AABB,
    checked_pairs: &mut HashSet<(usize, usize)>,
    x_search: &mut RBTree<OrderedFloat<f32>>,
    y_search: &mut RBTree<OrderedFloat<f32>>,
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
                    x_search,
                    y_search,
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
                x_search,
                y_search,
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
) -> Intersections {
    let bvh = construct_bvh_tree(segments);
    let mut checked_pairs: HashSet<(usize, usize)> = (0..segments.len()).map(|i| (i, i)).collect();
    let mut x_search: RBTree<OrderedFloat<f32>> = RBTree::new();
    let mut y_search: RBTree<OrderedFloat<f32>> = RBTree::new();
    let mut intersections: HashMap<Vector2<OrderedFloat<f32>>, Vec<usize>> = HashMap::new();

    if include_all_endpoints {
        for (i, segment) in segments.iter().enumerate() {
            let ordered_point_0 =
                Vector2::new(OrderedFloat(segment[0].x), OrderedFloat(segment[0].y));
            let ordered_point_1 =
                Vector2::new(OrderedFloat(segment[1].x), OrderedFloat(segment[1].y));
            x_search.insert(ordered_point_0.x);
            y_search.insert(ordered_point_0.y);
            x_search.insert(ordered_point_1.x);
            y_search.insert(ordered_point_1.y);
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
        &mut x_search,
        &mut y_search,
        &mut intersections,
        merging_threshold,
    );

    intersections
}

pub fn find_curves_intersections(
    curves: &[HermiteCurve],
    resample_resolution: u32,
    merging_threshold: f32,
    include_all_endpoints: bool,
) -> Intersections {
    let samples: Vec<Vec<Point>> = curves
        .iter()
        .map(|curve| resample_curve(&curve, resample_resolution))
        .collect();
    let segmented_curves: Vec<Segment> = samples
        .into_iter()
        .flat_map(|curve| {
            (0..curve.len() - 1)
                .map(|i| [curve[i], curve[i + 1]])
                .collect::<Vec<_>>()
        })
        .collect();

    find_segments_intersections(&segmented_curves, merging_threshold, include_all_endpoints)
}

#[cfg(test)]
mod test {
    use std::collections::HashSet;

    use nalgebra::Vector2;
    use ordered_float::OrderedFloat;

    use crate::{
        intersections::{AABB, Intersections, find_segments_intersections},
        street_graph::points_are_close,
        tensor_field::Point,
    };

    fn assert_intersection_maps_eq(test: &Intersections, expected: &Intersections) {
        assert_eq!(test.len(), expected.len());
        let expected_key_value_pairs: Vec<(Point, HashSet<usize>)> = expected
            .iter()
            .map(|(key, val)| (Point::new(*key.x, *key.y), HashSet::from_iter(val.clone())))
            .collect();
        for key in test.keys() {
            println!(
                "{key}, value: {:?}, expected: {expected_key_value_pairs:?}",
                &test[key]
            );
            assert!(
                expected_key_value_pairs
                    .iter()
                    .any(|(expected_key, expected_value)| {
                        println!("Close: {:?}", (key.map(|x| *x), *expected_key));
                        points_are_close(key.map(|x| *x), *expected_key)
                            && expected_value.len() == test[key].len()
                            && expected_value
                                .difference(&HashSet::from_iter(test[key].clone()))
                                .count()
                                == 0
                    })
            );
        }
    }

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
        let mut expected_intersections = Intersections::new();
        expected_intersections.insert(
            Vector2::new(OrderedFloat(1.3752345), OrderedFloat(1.3330231)),
            vec![0, 1],
        );
        assert_intersection_maps_eq(&intersections, &expected_intersections);
    }

    #[test]
    fn bvh_triple_intersection() {
        let segments = vec![
            [Point::new(2.0, 0.0), Point::new(0.466, 3.273)],
            [Point::new(0.0, 1.0), Point::new(1.817, 1.44)],
            [Point::new(1.3752345, 2.0), Point::new(1.3752345, 0.0)],
        ];

        let intersections = find_segments_intersections(&segments, 0.0001, false);
        let mut expected_intersections = Intersections::new();
        expected_intersections.insert(
            Vector2::new(OrderedFloat(1.3752345), OrderedFloat(1.3330231)),
            vec![0, 1, 2],
        );
        assert_intersection_maps_eq(&intersections, &expected_intersections);
    }

    #[test]
    fn chained_segment_bvh_intersections() {
        let segments = vec![
            [Point::new(0.97, 0.58), Point::new(2.52, 1.66)],
            [Point::new(2.52, 1.66), Point::new(3.89, 0.38)],
            [Point::new(1.07, 1.28), Point::new(4.94, 0.56)],
        ];

        let intersections = find_segments_intersections(&segments, 0.0001, true);
        let mut expected_intersections = Intersections::new();
        expected_intersections.insert(
            Vector2::new(OrderedFloat(1.78399), OrderedFloat(1.14717)),
            vec![0, 2],
        );
        expected_intersections.insert(
            Vector2::new(OrderedFloat(3.38837), OrderedFloat(0.84868)),
            vec![1, 2],
        );
        expected_intersections.insert(
            Vector2::new(OrderedFloat(2.52), OrderedFloat(1.66)),
            vec![0, 1],
        );

        expected_intersections.insert(
            Vector2::new(OrderedFloat(0.97), OrderedFloat(0.58)),
            vec![0],
        );

        expected_intersections.insert(
            Vector2::new(OrderedFloat(3.89), OrderedFloat(0.38)),
            vec![1],
        );

        expected_intersections.insert(
            Vector2::new(OrderedFloat(1.07), OrderedFloat(1.28)),
            vec![2],
        );

        expected_intersections.insert(
            Vector2::new(OrderedFloat(4.94), OrderedFloat(0.56)),
            vec![2],
        );

        assert_intersection_maps_eq(&intersections, &expected_intersections);
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
        let mut expected_intersections = Intersections::new();
        expected_intersections.insert(
            Vector2::new(OrderedFloat(69.01391), OrderedFloat(312.52502)),
            vec![0],
        );
        expected_intersections.insert(
            Vector2::new(OrderedFloat(106.17567), OrderedFloat(303.3888)),
            vec![1],
        );
        expected_intersections.insert(
            Vector2::new(OrderedFloat(80.89961), OrderedFloat(331.2429)),
            vec![1],
        );
        expected_intersections.insert(
            Vector2::new(OrderedFloat(84.59501), OrderedFloat(327.1706)),
            vec![0, 1],
        );
        // println!("Intersections: {:?}", intersections.keys().map(|point| (*point.x, *point.y)).collect::<Vec<_>>());
        assert_intersection_maps_eq(&intersections, &expected_intersections);
    }
}
