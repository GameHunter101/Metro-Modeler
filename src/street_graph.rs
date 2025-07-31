use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

use crate::status::SkipList;
use crate::tensor_field::Point;

#[derive(Debug, PartialEq, PartialOrd)]
struct IntersectionPoint {
    position: Point,
    intersecting_segment_indices: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct EventPoint {
    position: Point,
    segment_indices: HashSet<usize>,
    event_type: EventPointType,
}

impl EventPoint {
    pub fn new(
        position: Point,
        segment_indices: HashSet<usize>,
        event_type: EventPointType,
    ) -> Self {
        Self {
            position,
            segment_indices,
            event_type,
        }
    }

    pub fn position(&self) -> Point {
        self.position
    }

    pub fn segment_indices(&self) -> &HashSet<usize> {
        &self.segment_indices
    }

    pub fn event_type(&self) -> EventPointType {
        self.event_type
    }

    pub fn add_segments(event: std::ptr::NonNull<cool_utils::data_structures::rbtree::Node<Self>>, segment_indices: &HashSet<usize>) {
        unsafe {
            (*event.as_ptr()).value.segment_indices = (*event.as_ptr()).value
                .segment_indices
                .union(segment_indices)
                .copied()
                .collect();
        }
    }

    pub fn test(&mut self) {

    }
}

#[derive(PartialEq, PartialOrd, Eq, Debug, Clone, Copy)]
pub enum EventPointType {
    StartPoint,
    Intersection,
    EndPoint,
}

impl Eq for EventPoint {}

impl PartialEq for EventPoint {
    fn eq(&self, other: &Self) -> bool {
        self.event_type == other.event_type
            && (other.position - self.position).norm_squared() < 0.01
    }
}

impl PartialOrd for EventPoint {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match self.position.y.partial_cmp(&other.position.y) {
            Some(y_comp) => {
                if y_comp == std::cmp::Ordering::Equal {
                    std::cmp::Reverse(self.position.x)
                        .partial_cmp(&std::cmp::Reverse(other.position.x))
                } else {
                    Some(y_comp)
                }
            }
            None => None,
        }
    }
}

impl Ord for EventPoint {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Greater)
    }
}

pub type Segment = [Point; 2];

fn segments_to_event_queue(segments: &[Segment]) -> BinaryHeap<EventPoint> {
    BinaryHeap::from_iter(segments.iter().enumerate().flat_map(|(i, &[start, end])| {
        let (start, end) = if start.y > end.y {
            (start, end)
        } else {
            (end, start)
        };
        [
            EventPoint {
                position: start,
                segment_indices: HashSet::from_iter(std::iter::once(i)),
                event_type: EventPointType::StartPoint,
            },
            EventPoint {
                position: end,
                segment_indices: HashSet::from_iter(std::iter::once(i)),
                event_type: EventPointType::EndPoint,
            },
        ]
    }))
}

fn calc_intersection_point(segment_0: Segment, segment_1: Segment) -> Option<Point> {
    let cross = |p_0: Point, p_1: Point| p_0.x * p_1.y - p_0.y * p_1.x;

    let p = segment_0[0];
    let q = segment_1[0];
    let r = segment_0[1];
    let s = segment_1[1];

    let denominator = cross(s - q, r - p);

    let u = cross(p - q, s - q) / denominator;
    let t = cross(p - q, r - p) / denominator;

    if (u >= 0.0 && u <= 1.0) && (t >= 0.0 && t <= 1.0) {
        Some(p + u * (r - p))
    } else {
        None
    }
}

fn update_status(
    status: &mut SkipList,
    event: EventPoint,
    segments: &[Segment],
) -> (Vec<EventPoint>, Option<IntersectionPoint>) {
    match event.event_type {
        EventPointType::StartPoint => {
            let (all_left_neighbors, all_right_neighbors): (
                Vec<Option<usize>>,
                Vec<Option<usize>>,
            ) = event
                .segment_indices
                .iter()
                .map(|segment_index| status.insert(*segment_index, &segments, event.position.y))
                .unzip();

            let non_event_left_neighbors: Vec<usize> = all_left_neighbors
                .iter()
                .flat_map(|segment| {
                    if let Some(real_segment) = segment
                        && !event.segment_indices.contains(real_segment)
                    {
                        Some(*real_segment)
                    } else {
                        None
                    }
                })
                .collect();
            let left_neighbor = non_event_left_neighbors.get(0).copied();

            let non_event_right_neighbors: Vec<usize> = all_right_neighbors
                .iter()
                .flat_map(|segment| {
                    if let Some(real_segment) = segment
                        && !event.segment_indices.contains(real_segment)
                    {
                        Some(*real_segment)
                    } else {
                        None
                    }
                })
                .collect();
            let right_neighbor = non_event_right_neighbors.get(0).copied();

            let start_point_intersecting_segments = if event.segment_indices.len() == 1 {
                Vec::new()
            } else {
                event.segment_indices.iter().copied().collect()
            };

            let mut lower_intersection_points = Vec::new();

            for index in &event.segment_indices {
                if let Some(left_neighbor) = left_neighbor {
                    if let Some(intersection_point) =
                        calc_intersection_point(segments[left_neighbor], segments[*index])
                        && intersection_point.y < event.position.y
                    {
                        lower_intersection_points.push((
                            intersection_point,
                            HashSet::from_iter(vec![*index, left_neighbor]),
                        ));
                    }
                }

                if let Some(right_neighbor) = right_neighbor {
                    if let Some(intersection_point) =
                        calc_intersection_point(segments[right_neighbor], segments[*index])
                        && intersection_point.y < event.position.y
                    {
                        lower_intersection_points.push((
                            intersection_point,
                            HashSet::from_iter(vec![*index, right_neighbor]),
                        ));
                    }
                }
            }

            let event_points: Vec<EventPoint> = lower_intersection_points
                .into_iter()
                .map(|(position, segment_indices)| EventPoint {
                    position,
                    segment_indices,
                    event_type: EventPointType::Intersection,
                })
                .collect();

            (
                event_points,
                if !start_point_intersecting_segments.is_empty() {
                    Some(IntersectionPoint {
                        position: event.position,
                        intersecting_segment_indices: start_point_intersecting_segments,
                    })
                } else {
                    None
                },
            )
        }
        EventPointType::Intersection => todo!(),
        EventPointType::EndPoint => todo!(),
    }
}

fn find_interesctions(segments: &[Segment]) -> Vec<IntersectionPoint> {
    let mut event_queue = segments_to_event_queue(segments);
    let mut status = SkipList::new();

    let mut intersections = Vec::new();

    while let Some(event) = event_queue.pop() {
        let (new_events, possible_intersection) = update_status(&mut status, event, segments);
        if let Some(intersection) = possible_intersection {
            intersections.push(intersection);
        }

        event_queue.extend(new_events.into_iter());
    }

    intersections
}

pub fn test_status() {
    let segments = vec![
        [Point::new(1.0, 1.0), Point::new(2.0, -4.0)],
        [Point::new(-5.0, 12.0), Point::new(13.0, -0.2)],
        [Point::new(2.0, 2.0), Point::new(0.0, -10.0)],
        [Point::new(-10.0, -5.0), Point::new(2.0, 5.0)],
    ];

    let mut list = SkipList::new();
    for i in 0..segments.len() {
        list.insert(i, &segments, 1.0);
    }

    println!("{:?}", list.iter(0).collect::<Vec<_>>());
    println!("{:?}", list.iter(1).collect::<Vec<_>>());
    println!("{:?}", list.iter(2).collect::<Vec<_>>());
    panic!();
}

#[cfg(test)]
mod test {
    use std::{collections::HashSet, iter};

    use crate::{street_graph::IntersectionPoint, street_plan::heap_to_vec, tensor_field::Point};

    use super::{
        EventPoint, EventPointType, SkipList, calc_intersection_point, segments_to_event_queue,
        update_status,
    };

    #[test]
    fn segments_to_event_queue_correctly_prioritizes_single_segment() {
        let segments = [[Point::new(1.0, 1.0), Point::new(2.0, -4.0)]];

        let event_queue = segments_to_event_queue(&segments);

        let event_queue_vec = event_queue.into_vec();

        let expected_event_queue_vec = vec![
            EventPoint {
                position: Point::new(1.0, 1.0),
                segment_indices: HashSet::from_iter(std::iter::once(0)),
                event_type: EventPointType::StartPoint,
            },
            EventPoint {
                position: Point::new(2.0, -4.0),
                segment_indices: HashSet::from_iter(std::iter::once(0)),
                event_type: EventPointType::EndPoint,
            },
        ];

        assert_eq!(event_queue_vec, expected_event_queue_vec);
    }

    #[test]
    fn segments_to_event_queue_correctly_prioritizes_segments() {
        let segments = [
            [Point::new(1.0, 1.0), Point::new(2.0, -4.0)],
            [Point::new(-5.0, 12.0), Point::new(13.0, -0.2)],
            [Point::new(2.0, 2.0), Point::new(0.0, -10.0)],
            [Point::new(-10.0, -5.0), Point::new(2.0, 5.0)],
        ];

        let event_queue = segments_to_event_queue(&segments);
        let event_queue_as_vec = heap_to_vec(event_queue);

        let proper_events = [
            EventPoint {
                position: Point::new(-5.0, 12.0),
                segment_indices: HashSet::from_iter(iter::once(1)),
                event_type: EventPointType::StartPoint,
            },
            EventPoint {
                position: Point::new(2.0, 5.0),
                segment_indices: HashSet::from_iter(iter::once(3)),
                event_type: EventPointType::StartPoint,
            },
            EventPoint {
                position: Point::new(2.0, 2.0),
                segment_indices: HashSet::from_iter(iter::once(2)),
                event_type: EventPointType::StartPoint,
            },
            EventPoint {
                position: Point::new(1.0, 1.0),
                segment_indices: HashSet::from_iter(iter::once(0)),
                event_type: EventPointType::StartPoint,
            },
            EventPoint {
                position: Point::new(13.0, -0.2),
                segment_indices: HashSet::from_iter(iter::once(1)),
                event_type: EventPointType::EndPoint,
            },
            EventPoint {
                position: Point::new(2.0, -4.0),
                segment_indices: HashSet::from_iter(iter::once(0)),
                event_type: EventPointType::EndPoint,
            },
            EventPoint {
                position: Point::new(-10.0, -5.0),
                segment_indices: HashSet::from_iter(iter::once(3)),
                event_type: EventPointType::EndPoint,
            },
            EventPoint {
                position: Point::new(0.0, -10.0),
                segment_indices: HashSet::from_iter(iter::once(2)),
                event_type: EventPointType::EndPoint,
            },
        ];

        assert_eq!(event_queue_as_vec, proper_events);
    }

    #[test]
    fn intersection_calc_correctly_determines_intersection_point() {
        let segment_0 = [Point::new(1.0, 1.0), Point::new(2.0, -4.0)];
        let segment_1 = [Point::new(4.0, 2.0), Point::new(1.5, -3.0)];

        let intersection_res = calc_intersection_point(segment_0, segment_1);

        assert!((intersection_res.unwrap() - Point::new(1.71428571, -2.57142857)).norm() < 0.00001);
    }

    #[test]
    fn intersection_calc_correctly_detects_when_no_intersection() {
        let segment_0 = [Point::new(2.0, 2.0), Point::new(0.0, -10.0)];
        let segment_1 = [Point::new(4.0, 2.0), Point::new(1.5, -3.0)];

        let intersection_res = calc_intersection_point(segment_0, segment_1);

        assert!(intersection_res.is_none());
    }

    #[test]
    fn intersection_calc_correctly_ignores_parallel_lines() {
        let segment_0 = [Point::new(2.0, 2.0), Point::new(0.0, -10.0)];
        let segment_1 = [Point::new(2.0, 0.0), Point::new(0.0, -12.0)];

        let intersection_res = calc_intersection_point(segment_0, segment_1);

        assert!(intersection_res.is_none());
    }

    #[test]
    fn start_points_are_correctly_inserted_into_status() {
        let segments = [
            [Point::new(-5.0, 12.0), Point::new(13.0, -0.2)],
            [Point::new(2.0, 5.0), Point::new(-10.0, -5.0)],
            [Point::new(2.0, 2.0), Point::new(0.0, -10.0)],
        ];

        let mut status = SkipList::new();

        let mut new_events: Vec<EventPoint> = Vec::new();
        let mut new_intersections = Vec::new();

        for (i, segment) in segments.iter().enumerate() {
            let event = EventPoint {
                position: segment[0],
                segment_indices: HashSet::from_iter(iter::once(i)),
                event_type: EventPointType::StartPoint,
            };

            let (result_events, result_intersection) = update_status(&mut status, event, &segments);

            new_events.extend(result_events.into_iter());
            if let Some(intersection) = result_intersection {
                new_intersections.push(intersection);
            }
        }

        assert!(new_events.is_empty());
        assert!(new_intersections.is_empty());

        let status_as_vec = status.to_vec();

        let expected_status_vec = vec![1, 2, 0];

        assert_eq!(status_as_vec, expected_status_vec);
    }

    fn points_are_close(p_1: Point, p_2: Point) {
        println!("--- {p_1:?} | {p_2:?}");
        assert!((p_1 - p_2).norm() < 0.00001)
    }

    #[test]
    fn start_points_correctly_calculate_single_future_intersection() {
        let segments = [
            [Point::new(-5.0, 12.0), Point::new(13.0, -0.2)],
            [Point::new(2.0, 5.0), Point::new(-10.0, -5.0)],
            [Point::new(-0.7, 1.5), Point::new(1.4, 0.0)],
            [Point::new(1.0, 1.0), Point::new(2.0, -4.0)],
        ];

        let mut status = SkipList::new();

        let mut new_events: Vec<EventPoint> = Vec::new();
        let mut new_intersections = Vec::new();

        for (i, segment) in segments.iter().enumerate() {
            let event = EventPoint {
                position: segment[0],
                segment_indices: HashSet::from_iter(iter::once(i)),
                event_type: EventPointType::StartPoint,
            };

            let (result_events, result_intersection) = update_status(&mut status, event, &segments);

            new_events.extend(result_events.into_iter());
            if let Some(intersection) = result_intersection {
                new_intersections.push(intersection);
            }
        }

        let expected_new_events = vec![EventPoint {
            position: Point::new(1.16666667, 0.16666667),
            segment_indices: HashSet::from_iter(iter::once(3)),
            event_type: EventPointType::Intersection,
        }];

        assert_eq!(new_events.len(), expected_new_events.len());
        for (i, expected_event) in expected_new_events.iter().enumerate() {
            let real_event = &new_events[i];
            points_are_close(real_event.position, expected_event.position);
        }
        assert!(new_intersections.is_empty());

        let status_as_vec = status.to_vec();

        let expected_status_vec = vec![1, 2, 3, 0];

        assert_eq!(status_as_vec, expected_status_vec);
    }

    #[test]
    fn start_points_correctly_calculate_double_future_intersection() {
        let segments = [
            [Point::new(-5.0, 12.0), Point::new(13.0, -0.2)],
            [Point::new(2.0, 5.0), Point::new(-10.0, -5.0)],
            [Point::new(2.0, 2.0), Point::new(0.0, -10.0)],
            [Point::new(-0.7, 1.5), Point::new(1.4, 0.0)],
            [Point::new(1.0, 1.0), Point::new(2.0, -4.0)],
        ];

        let mut status = SkipList::new();

        let mut new_events: Vec<EventPoint> = Vec::new();
        let mut new_intersections = Vec::new();

        for (i, segment) in segments.iter().enumerate() {
            let event = EventPoint {
                position: segment[0],
                segment_indices: HashSet::from_iter(iter::once(i)),
                event_type: EventPointType::StartPoint,
            };

            let (result_events, result_intersection) = update_status(&mut status, event, &segments);

            new_events.extend(result_events.into_iter());
            if let Some(intersection) = result_intersection {
                new_intersections.push(intersection);
            }
        }

        let expected_new_events = vec![
            EventPoint {
                position: Point::new(1.16666667, 0.16666667),
                segment_indices: HashSet::from_iter(iter::once(3)),
                event_type: EventPointType::Intersection,
            },
            EventPoint {
                position: Point::new(1.45454545, -1.27272727),
                segment_indices: HashSet::from_iter(iter::once(3)),
                event_type: EventPointType::Intersection,
            },
        ];

        assert_eq!(new_events.len(), expected_new_events.len());
        for (i, expected_event) in expected_new_events.iter().enumerate() {
            let real_event = &new_events[i];
            points_are_close(real_event.position, expected_event.position);
        }

        assert!(new_intersections.is_empty());

        let status_as_vec = status.to_vec();

        let expected_status_vec = vec![1, 3, 4, 2, 0];

        assert_eq!(status_as_vec, expected_status_vec);
    }

    #[test]
    fn start_points_correctly_identify_single_start_intersection() {
        let segments = [
            [Point::new(-5.0, 12.0), Point::new(13.0, -0.2)],
            [Point::new(2.0, 5.0), Point::new(-10.0, -5.0)],
            [Point::new(1.0, 1.0), Point::new(1.4, 0.0)],
            [Point::new(1.0, 1.0), Point::new(2.0, -4.0)],
        ];

        let mut status = SkipList::new();

        let mut new_events: Vec<EventPoint> = Vec::new();
        let mut new_intersections = Vec::new();

        for (i, segment) in segments[..3].iter().enumerate() {
            let event = EventPoint {
                position: segment[0],
                segment_indices: if i == 2 {
                    HashSet::from_iter(vec![i, i + 1])
                } else {
                    HashSet::from_iter(iter::once(i))
                },
                event_type: EventPointType::StartPoint,
            };

            let (result_events, result_intersection) = update_status(&mut status, event, &segments);

            new_events.extend(result_events.into_iter());
            if let Some(intersection) = result_intersection {
                new_intersections.push(intersection);
            }
        }

        let expected_intersections = vec![IntersectionPoint {
            position: Point::new(1.0, 1.0),
            intersecting_segment_indices: vec![3, 2],
        }];

        // Uncertainty in the order of elements is caused by the use of hashsets, either order is
        // valid and should not affect algorithm performance

        assert_eq!(new_intersections.len(), 1);
        assert_eq!(
            new_intersections[0].position,
            expected_intersections[0].position
        );
        assert_eq!(
            HashSet::<&usize>::from_iter(new_intersections[0].intersecting_segment_indices.iter()),
            HashSet::from_iter(
                expected_intersections[0]
                    .intersecting_segment_indices
                    .iter()
            )
        );

        assert!(new_events.is_empty());

        let status_as_vec = status.to_vec();

        let expected_status_vec_1 = vec![1, 2, 3, 0];
        let expected_status_vec_2 = vec![1, 3, 2, 0];

        assert!(status_as_vec == expected_status_vec_1 || status_as_vec == expected_status_vec_2);
    }

    /* #[test]
    fn start_points_correctly_identify_double_start_intersection() {
        let segments = [
            [Point::new(-5.0, 12.0), Point::new(13.0, -0.2)],
            [Point::new(2.0, 5.0), Point::new(-10.0, -5.0)],
            [Point::new(1.0, 1.0), Point::new(1.4, 0.0)],
            [Point::new(1.0, 1.0), Point::new(2.0, -4.0)],
        ];

        let mut status = SkipList::new();

        let mut new_events: Vec<EventPoint> = Vec::new();
        let mut new_intersections = Vec::new();

        for (i, segment) in segments.iter().enumerate() {
            let event = EventPoint {
                position: segment[0],
                segment_indices: HashSet::from_iter(iter::once(i)),
                event_type: EventPointType::StartPoint,
            };

            let (result_events, result_intersection) = update_status(&mut status, event, &segments);

            new_events.extend(result_events.into_iter());
            if let Some(intersection) = result_intersection {
                new_intersections.push(intersection);
            }
        }

        let expected_intersections = vec![IntersectionPoint {
            position: Point::new(1.0, 1.0),
            intersecting_segment_indices: vec![3, 2],
        }];

        assert_eq!(new_intersections, expected_intersections);
        assert!(new_events.is_empty());

        let status_as_vec = status_to_vec(&status);

        let expected_status_vec = vec![1, 2, 3, 0];

        assert_eq!(status_as_vec, expected_status_vec);
    } */
}
