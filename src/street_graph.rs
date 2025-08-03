use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

use crate::status::{SkipList, get_x_val_of_segment_at_height};
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

    pub fn add_segments(&mut self, segment_indices: &HashSet<usize>) {
        unsafe {
            self.segment_indices = self
                .segment_indices
                .union(segment_indices)
                .copied()
                .collect();
        }
    }

    pub fn test(&mut self) {}
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
        EventPointType::StartPoint => update_status_with_start_point(event, status, segments),
        EventPointType::Intersection => {
            update_status_with_intersection_point(event, status, segments)
        }
        EventPointType::EndPoint => todo!(),
    }
}

fn update_status_with_start_point(
    event: EventPoint,
    status: &mut SkipList,
    segments: &[Segment],
) -> (Vec<EventPoint>, Option<IntersectionPoint>) {
    let event_segment_indices_vec: Vec<usize> = event.segment_indices.iter().copied().collect();

    let earliest_segment_end = segments.iter().map(|[p_0, p_1]| p_0.y.min(p_1.y)).fold(
        segments[event_segment_indices_vec[0]][0]
            .y
            .min(segments[event_segment_indices_vec[0]][1].y),
        |acc, y| acc.min(y),
    );

    let mut sorted_event_segment_indices = event_segment_indices_vec.clone();
    sorted_event_segment_indices.sort_by(|a, b| {
        get_x_val_of_segment_at_height(&segments[*a], earliest_segment_end).total_cmp(
            &get_x_val_of_segment_at_height(&segments[*b], earliest_segment_end),
        )
    });

    let (left_neighbor, right_neighbor) =
        if let Some(segment_index) = sorted_event_segment_indices.get(0) {
            status.insert(*segment_index, segments, event.position().y)
        } else {
            (None, None)
        };

    if sorted_event_segment_indices.len() > 1 {
        for segment_index in &sorted_event_segment_indices[1..] {
            let _ = status.insert(*segment_index, segments, event.position().y);
        }
    }

    let start_point_intersecting_segments = if event.segment_indices.len() == 1 {
        Vec::new()
    } else {
        event.segment_indices.iter().copied().collect()
    };

    let mut lower_intersection_points = Vec::new();

    for segment_index in &event.segment_indices {
        if let Some(left_neighbor) = left_neighbor {
            if let Some(intersection_point) =
                calc_intersection_point(segments[left_neighbor], segments[*segment_index])
                && intersection_point.y <= event.position.y
            {
                lower_intersection_points.push((
                    intersection_point,
                    HashSet::from_iter(vec![*segment_index, left_neighbor]),
                ));
            }
        }

        if let Some(right_neighbor) = right_neighbor {
            if let Some(intersection_point) =
                calc_intersection_point(segments[right_neighbor], segments[*segment_index])
                && intersection_point.y <= event.position.y
            {
                lower_intersection_points.push((
                    intersection_point,
                    HashSet::from_iter(vec![*segment_index, right_neighbor]),
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

fn update_status_with_intersection_point(
    event: EventPoint,
    status: &mut SkipList,
    segments: &[Segment],
) -> (Vec<EventPoint>, Option<IntersectionPoint>) {
    let event_segment_indices_vec: Vec<usize> = event.segment_indices().iter().copied().collect();
    let calc_lowest_segment_start = |segment_indices: &[usize]| {
        segment_indices
            .iter()
            .map(|index| {
                let segment = segments[*index];

                segment[0].y.max(segment[1].y)
            })
            .fold(
                segments[segment_indices[0]][0]
                    .y
                    .max(segments[segment_indices[0]][1].y),
                |acc, height| acc.min(height),
            )
    };
    let lowest_segment_start = calc_lowest_segment_start(&event_segment_indices_vec);

    let sorted_segment_indices = if let Some(horizontal_segment_index) =
        event_segment_indices_vec.iter().find(|&&segment_index| {
            segments[segment_index][0]
                .y
                .max(segments[segment_index][1].y)
                == lowest_segment_start
        }) {
        let mut filtered_indices: Vec<usize> = event_segment_indices_vec
            .iter()
            .filter(|&index| index != horizontal_segment_index)
            .copied()
            .collect();
        let new_lowest_segment_start = calc_lowest_segment_start(&filtered_indices);

        filtered_indices.sort_by(|a, b| {
            get_x_val_of_segment_at_height(&segments[*a], new_lowest_segment_start).total_cmp(
                &get_x_val_of_segment_at_height(&segments[*b], new_lowest_segment_start),
            )
        });

        std::iter::once(*horizontal_segment_index)
            .chain(filtered_indices)
            .collect()
    } else {
        let mut sorted_indices = event_segment_indices_vec.clone();
        sorted_indices.sort_by(|a, b| {
            get_x_val_of_segment_at_height(&segments[*a], lowest_segment_start).total_cmp(
                &get_x_val_of_segment_at_height(&segments[*b], lowest_segment_start),
            )
        });

        sorted_indices
    };

    let (potential_left_neighbor, potential_right_neighbor) = status.reverse(
        sorted_segment_indices[0],
        *sorted_segment_indices.last().unwrap(),
        segments,
        event.position().y,
    );

    let mut new_events = Vec::new();
    if let Some(left) = potential_left_neighbor
        && let Some(intersection_point) =
            calc_intersection_point(segments[left], segments[sorted_segment_indices[0]])
    {
        new_events.push(EventPoint {
            position: intersection_point,
            segment_indices: HashSet::from_iter(vec![left, sorted_segment_indices[0]]),
            event_type: EventPointType::Intersection,
        });
    }

    if let Some(right) = potential_right_neighbor
        && let Some(intersection_point) = calc_intersection_point(
            segments[right],
            segments[*sorted_segment_indices.last().unwrap()],
        )
    {
        new_events.push(EventPoint {
            position: intersection_point,
            segment_indices: HashSet::from_iter(vec![
                right,
                *sorted_segment_indices.last().unwrap(),
            ]),
            event_type: EventPointType::Intersection,
        });
    }

    if let Some(left) = potential_left_neighbor
        && let Some(intersection_point) =
            calc_intersection_point(segments[left], segments[sorted_segment_indices[0]])
    {
        new_events.push(EventPoint {
            position: intersection_point,
            segment_indices: HashSet::from_iter(vec![
                left,
                *sorted_segment_indices.last().unwrap(),
            ]),
            event_type: EventPointType::Intersection,
        });
    }

    (
        new_events,
        Some(IntersectionPoint {
            position: event.position(),
            intersecting_segment_indices: event_segment_indices_vec,
        }),
    )
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

    use crate::{
        event_queue::{self, EventQueue},
        street_graph::IntersectionPoint,
        street_plan::heap_to_vec,
        tensor_field::Point,
    };

    use super::{
        EventPoint, EventPointType, Segment, SkipList, calc_intersection_point,
        segments_to_event_queue, update_status,
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

    fn load_segment_start_event_to_event_queue(segments: &[Segment]) -> EventQueue {
        let mut event_queue = EventQueue::new();
        for (i, segment) in segments.iter().enumerate() {
            event_queue.push(EventPoint {
                position: if segment[0].y > segment[1].y {
                    segment[0]
                } else {
                    segment[1]
                },
                segment_indices: HashSet::from_iter(iter::once(i)),
                event_type: EventPointType::StartPoint,
            });
        }

        event_queue
    }

    #[test]
    fn start_points_are_correctly_inserted_into_status() {
        let segments = [
            [Point::new(-5.0, 12.0), Point::new(13.0, -0.2)],
            [Point::new(2.0, 5.0), Point::new(-10.0, -5.0)],
            [Point::new(2.0, 2.0), Point::new(0.0, -10.0)],
        ];

        let mut event_queue = load_segment_start_event_to_event_queue(&segments);

        let mut status = SkipList::new();

        let mut new_events: Vec<EventPoint> = Vec::new();
        let mut new_intersections = Vec::new();

        while let Some(event) = event_queue.pop() {
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

    fn points_are_close(p_1: Point, p_2: Point) -> bool {
        (p_1 - p_2).norm() < 0.00001
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
            assert!(points_are_close(
                real_event.position,
                expected_event.position
            ));
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

        let mut event_queue = load_segment_start_event_to_event_queue(&segments);
        let mut status = SkipList::new();

        let mut new_events: Vec<EventPoint> = Vec::new();
        let mut new_intersections = Vec::new();

        while let Some(event) = event_queue.pop() {
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
            assert!(points_are_close(
                real_event.position,
                expected_event.position
            ));
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

        let mut event_queue = load_segment_start_event_to_event_queue(&segments);
        let mut status = SkipList::new();

        let mut new_events: Vec<EventPoint> = Vec::new();
        let mut new_intersections = Vec::new();

        while let Some(event) = event_queue.pop() {
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

        let expected_vals_1 = vec![1, 2, 3, 0];
        let expected_vals_2 = vec![1, 3, 2, 0];
        assert!(status_as_vec == expected_vals_1 || status_as_vec == expected_vals_2);
    }

    #[test]
    fn start_points_correctly_identify_single_intersection_in_joint_start_point() {
        let segments = [
            [Point::new(-5.0, 12.0), Point::new(13.0, -0.2)],
            [Point::new(2.0, 5.0), Point::new(-10.0, -5.0)],
            [Point::new(1.0, 1.0), Point::new(1.4, 0.0)],
            [Point::new(1.0, 1.0), Point::new(2.0, -4.0)],
            [Point::new(0.4, 0.4), Point::new(1.3, 0.0)],
        ];

        let mut event_queue = load_segment_start_event_to_event_queue(&segments);
        let mut status = SkipList::new();

        let mut new_events: Vec<EventPoint> = Vec::new();
        let mut new_intersections = Vec::new();

        while let Some(event) = event_queue.pop() {
            let (result_events, result_intersection) = update_status(&mut status, event, &segments);

            new_events.extend(result_events.into_iter());
            if let Some(intersection) = result_intersection {
                new_intersections.push(intersection);
            }
        }

        assert_eq!(new_intersections.len(), 1);
        assert_eq!(new_intersections[0].position, Point::new(1.0, 1.0));
        assert_eq!(new_intersections[0].intersecting_segment_indices.len(), 2);
        assert!(
            new_intersections[0]
                .intersecting_segment_indices
                .contains(&3)
        );
        assert!(
            new_intersections[0]
                .intersecting_segment_indices
                .contains(&2)
        );

        assert_eq!(new_events.len(), 1);
        assert_eq!(new_events[0].event_type(), EventPointType::Intersection);
        assert!(points_are_close(
            new_events[0].position(),
            Point::new(1.1902439, 0.04878049)
        ));
        let intersection_segments: Vec<usize> =
            new_events[0].segment_indices.iter().copied().collect();
        assert_eq!(intersection_segments.len(), 2);
        assert!(intersection_segments.contains(&3));
        assert!(intersection_segments.contains(&4));

        let status_as_vec = status.to_vec();

        let expected_vals_1 = vec![1, 4, 2, 3, 0];
        let expected_vals_2 = vec![1, 4, 3, 2, 0];
        assert!(status_as_vec == expected_vals_1 || status_as_vec == expected_vals_2);
    }

    #[test]
    fn start_points_correctly_identify_segment_start_on_other_segment() {
        let segments = vec![
            [Point::new(0.0, 5.0), Point::new(5.0, 5.0)],
            [Point::new(2.0, 5.0), Point::new(2.0, 0.0)],
        ];

        let mut event_queue = load_segment_start_event_to_event_queue(&segments);
        let mut status = SkipList::new();

        let mut new_events: Vec<EventPoint> = Vec::new();
        let mut new_intersections = Vec::new();

        while let Some(event) = event_queue.pop() {
            let (result_events, result_intersection) = update_status(&mut status, event, &segments);

            new_events.extend(result_events.into_iter());
            if let Some(intersection) = result_intersection {
                new_intersections.push(intersection);
            }
        }

        assert!(new_intersections.is_empty());
        assert_eq!(new_events.len(), 1);
        assert_eq!(new_events[0].position, Point::new(2.0, 5.0));
        assert_eq!(
            new_events[0]
                .segment_indices
                .difference(&HashSet::from_iter(vec![0, 1]))
                .count(),
            0
        );
    }

    #[test]
    fn simple_intersection_point_correctly_correctly_flips_status() {
        let segments = vec![
            [Point::new(0.0, 5.0), Point::new(5.0, 5.0)],
            [Point::new(2.0, 7.0), Point::new(2.0, 0.0)],
        ];

        let mut event_queue = load_segment_start_event_to_event_queue(&segments);
        let mut status = SkipList::new();

        let mut new_events = EventQueue::new();
        let mut intersections = Vec::new();

        while let Some(event) = event_queue.pop() {
            let (result_events, result_intersection) = update_status(&mut status, event, &segments);

            for event in result_events {
                new_events.push(event);
            }

            if let Some(intersection) = result_intersection {
                intersections.push(intersection)
            }
        }

        assert_eq!(new_events.len(), 1);
        assert_eq!(status.to_vec(), vec![0, 1]);
        assert!(intersections.is_empty());

        let (result_events, result_intersection) =
            update_status(&mut status, new_events.pop().unwrap().clone(), &segments);
        assert!(result_events.is_empty());

        assert!(result_intersection.is_some(),);
        assert_eq!(
            result_intersection.as_ref().unwrap().position,
            Point::new(2.0, 5.0)
        );
        let intersecting_indices_set: HashSet<usize> =
            HashSet::from_iter(result_intersection.unwrap().intersecting_segment_indices);
        assert_eq!(
            intersecting_indices_set
                .difference(&HashSet::from_iter(vec![0, 1]))
                .count(),
            0
        );

        assert_eq!(status.to_vec(), vec![1, 0]);
    }

    #[test]
    fn complex_intersection_point_corectly_flips_status() {
        let segments = vec![
            [Point::new(0.0, 1.0), Point::new(5.0, 1.0)],
            [Point::new(1.0, 2.0), Point::new(1.0, 0.0)],
            [Point::new(0.0, 0.0), Point::new(2.0, 2.0)],
            [Point::new(0.0, 2.0), Point::new(2.0, 0.0)],
        ];

        let mut event_queue = load_segment_start_event_to_event_queue(&segments);
        let mut status = SkipList::new();

        let mut new_events = EventQueue::new();
        let mut intersections = Vec::new();

        while let Some(event) = event_queue.pop() {
            let (result_events, result_intersection) = update_status(&mut status, event, &segments);

            for event in result_events {
                new_events.push(event);
            }

            if let Some(intersection) = result_intersection {
                intersections.push(intersection)
            }
        }

        assert_eq!(new_events.len(), 1);
        assert_eq!(status.to_vec(), vec![0, 3, 1, 2]);
        let (result_events, result_intersection) =
            update_status(&mut status, new_events.pop().unwrap().clone(), &segments);
        dbg!(&result_events);
        assert!(result_events.is_empty());

        assert!(result_intersection.is_some(),);
        let intersecting_indices_set: HashSet<usize> =
            HashSet::from_iter(result_intersection.unwrap().intersecting_segment_indices);
        assert_eq!(
            intersecting_indices_set
                .difference(&HashSet::from_iter(vec![0, 1, 2, 3]))
                .count(),
            0
        );

        assert_eq!(status.to_vec(), vec![2, 1, 3, 0]);
    }
}
