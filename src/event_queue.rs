use std::{collections::HashSet, ptr::NonNull};

use cool_utils::data_structures::rbtree::{Node, RBTree};

use crate::{
    street_graph::{EventPoint, EventPointType, Segment},
    tensor_field::Point,
};

pub struct EventQueue {
    tree: RBTree<EventPoint>,
    arr: Vec<NonNull<Node<EventPoint>>>,
    len: usize,
}

impl EventQueue {
    pub fn new() -> Self {
        Self {
            tree: RBTree::new(),
            arr: Vec::new(),
            len: 0,
        }
    }

    pub fn from_segments(segments: &[Segment]) -> Self {
        let mut event_queue = EventQueue::new();
        for (i, segment) in segments.iter().enumerate() {
            let (start, end) = if segment[0].y > segment[1].y {
                (segment[0], segment[1])
            } else {
                if segment[0].y == segment[1].y && segment[0].x < segment[1].x {
                    (segment[0], segment[1])
                } else {
                    (segment[1], segment[0])
                }
            };

            let start_event = EventPoint {
                position: start,
                segment_indices: HashSet::from_iter(std::iter::once(i)),
                event_type: EventPointType::StartPoint,
            };

            let end_event = EventPoint {
                position: end,
                segment_indices: HashSet::from_iter(std::iter::once(i)),
                event_type: EventPointType::EndPoint,
            };

            event_queue.push(start_event);
            event_queue.push(end_event);
        }

        event_queue
    }

    pub fn print(&self) {
        unsafe {
            println!(
                "{:?}",
                self.arr
                    .iter()
                    .map(|ptr| &(*ptr.as_ptr()).value)
                    .collect::<Vec<_>>()
            );
        }
    }

    fn swap(&mut self, a: usize, b: usize) {
        assert!(a != b);
        let (a_ref, b_ref) = if a < b {
            let (a_split, b_split) = self.arr.split_at_mut(b);
            (&mut a_split[a], &mut b_split[0])
        } else {
            let (b_split, a_split) = self.arr.split_at_mut(a);
            (&mut a_split[0], &mut b_split[b])
        };

        std::mem::swap(a_ref, b_ref);
    }

    fn sift_up(&mut self, start_index: usize) {
        unsafe {
            if start_index == 0 {
                return;
            }

            let parent_index = (start_index - 1) / 2;

            let current_node = &self.arr[start_index];
            let parent_node = &self.arr[parent_index];

            if (*current_node.as_ptr()).value > (*parent_node.as_ptr()).value {
                self.swap(start_index, parent_index);
                self.sift_up(parent_index);
            }
        }
    }

    pub fn push(&mut self, event_point: EventPoint) {
        unsafe {
            if let Some(node) = self.tree.unsafe_search(&event_point) {
                (&mut (*node.as_ptr()).value).add_segments(event_point.segment_indices());
            } else {
                let ptr = self.tree.unsafe_insert(event_point);
                self.arr.push(ptr);
                self.len += 1;
                self.sift_up(self.len - 1);
            }
        }
    }

    fn sift_down(&mut self, start_index: usize) {
        unsafe {
            if start_index == self.len - 1 {
                return;
            }

            let left_child_index = 2 * start_index + 1;
            let right_child_index = left_child_index + 1;

            if left_child_index > self.len - 1 {
                return;
            }

            let (max_index, max_val) = if right_child_index > self.len - 1
                || (*self.arr[left_child_index].as_ptr()).value
                    > (*self.arr[right_child_index].as_ptr()).value
            {
                (left_child_index, &self.arr[left_child_index])
            } else {
                (right_child_index, &self.arr[right_child_index])
            };

            if (*self.arr[start_index].as_ptr()).value < (*max_val.as_ptr()).value {
                self.swap(start_index, max_index);
                self.sift_down(max_index);
            }
        }
    }

    pub fn pop(&mut self) -> Option<EventPoint> {
        if self.len == 0 {
            return None;
        }
        let node_ptr = if self.len == 1 {
            self.len -= 1;
            self.arr.pop().unwrap()
        } else {
            self.swap(0, self.len - 1);
            let val = self.arr.pop().unwrap();
            self.len -= 1;
            self.sift_down(0);
            val
        };
        unsafe {
            let value = (*node_ptr.as_ptr()).value.clone();
            if self.tree.delete(&value) {
                Some(value)
            } else {
                None
            }
        }
    }

    pub fn to_vec(&self) -> Vec<EventPoint> {
        self.arr
            .iter()
            .map(|ptr| unsafe { (*ptr.as_ptr()).value.clone() })
            .collect()
    }

    pub fn len(&self) -> usize {
        self.len
    }
}

#[cfg(test)]
mod test {
    use std::{collections::HashSet, iter};

    use crate::{
        street_graph::{EventPoint, EventPointType},
        tensor_field::Point,
    };

    use super::EventQueue;

    #[test]
    fn inserting_one_element_works_correctly() {
        let mut queue = EventQueue::new();
        queue.push(EventPoint::new(
            Point::new(0.0, 0.0),
            HashSet::new(),
            EventPointType::StartPoint,
        ));

        let expected_body = vec![EventPoint::new(
            Point::new(0.0, 0.0),
            HashSet::new(),
            EventPointType::StartPoint,
        )];

        assert_eq!(queue.to_vec(), expected_body);
    }

    #[test]
    fn inserting_multiple_elements_works_correctly() {
        let mut queue = EventQueue::new();
        queue.push(EventPoint::new(
            Point::new(0.0, 0.0),
            HashSet::new(),
            EventPointType::StartPoint,
        ));
        queue.push(EventPoint::new(
            Point::new(0.0, 1.0),
            HashSet::new(),
            EventPointType::StartPoint,
        ));
        queue.push(EventPoint::new(
            Point::new(1.0, 1.0),
            HashSet::new(),
            EventPointType::StartPoint,
        ));

        let expected_body = vec![
            EventPoint::new(
                Point::new(0.0, 1.0),
                HashSet::new(),
                EventPointType::StartPoint,
            ),
            EventPoint::new(
                Point::new(0.0, 0.0),
                HashSet::new(),
                EventPointType::StartPoint,
            ),
            EventPoint::new(
                Point::new(1.0, 1.0),
                HashSet::new(),
                EventPointType::StartPoint,
            ),
        ];

        assert_eq!(queue.to_vec(), expected_body);
    }

    #[test]
    fn popping_single_element_works_correctly() {
        let mut queue = EventQueue::new();
        queue.push(EventPoint::new(
            Point::new(0.0, 0.0),
            HashSet::new(),
            EventPointType::StartPoint,
        ));
        queue.push(EventPoint::new(
            Point::new(0.0, 1.0),
            HashSet::new(),
            EventPointType::StartPoint,
        ));
        queue.push(EventPoint::new(
            Point::new(1.0, 1.0),
            HashSet::new(),
            EventPointType::StartPoint,
        ));

        queue.pop();

        let expected_body = vec![
            EventPoint::new(
                Point::new(1.0, 1.0),
                HashSet::new(),
                EventPointType::StartPoint,
            ),
            EventPoint::new(
                Point::new(0.0, 0.0),
                HashSet::new(),
                EventPointType::StartPoint,
            ),
        ];

        assert_eq!(queue.to_vec(), expected_body);
    }

    #[test]
    fn popping_multiple_elements_works_correctly() {
        let mut queue = EventQueue::new();
        queue.push(EventPoint::new(
            Point::new(0.0, 0.0),
            HashSet::new(),
            EventPointType::StartPoint,
        ));
        queue.push(EventPoint::new(
            Point::new(0.0, 1.0),
            HashSet::new(),
            EventPointType::StartPoint,
        ));
        queue.push(EventPoint::new(
            Point::new(1.0, 1.0),
            HashSet::new(),
            EventPointType::StartPoint,
        ));
        queue.push(EventPoint::new(
            Point::new(12.0, 0.7),
            HashSet::new(),
            EventPointType::StartPoint,
        ));

        assert_eq!(queue.len(), 4);

        queue.pop();
        queue.pop();

        assert_eq!(queue.len(), 2);

        let expected_body = vec![
            EventPoint::new(
                Point::new(12.0, 0.7),
                HashSet::new(),
                EventPointType::StartPoint,
            ),
            EventPoint::new(
                Point::new(0.0, 0.0),
                HashSet::new(),
                EventPointType::StartPoint,
            ),
        ];

        assert_eq!(queue.to_vec(), expected_body);
    }

    #[test]
    fn popping_to_empty_queue_works_correctly() {
        let mut queue = EventQueue::new();
        queue.push(EventPoint::new(
            Point::new(0.0, 0.0),
            HashSet::new(),
            EventPointType::StartPoint,
        ));
        queue.push(EventPoint::new(
            Point::new(0.0, 1.0),
            HashSet::new(),
            EventPointType::StartPoint,
        ));
        queue.push(EventPoint::new(
            Point::new(1.0, 1.0),
            HashSet::new(),
            EventPointType::StartPoint,
        ));
        queue.push(EventPoint::new(
            Point::new(12.0, 0.7),
            HashSet::new(),
            EventPointType::StartPoint,
        ));

        queue.pop();
        queue.pop();
        queue.pop();
        queue.pop();

        let expected_body = Vec::new();

        assert_eq!(queue.to_vec(), expected_body);
    }

    #[test]
    fn inserting_joint_start_segments_gets_merged() {
        let mut queue = EventQueue::new();

        queue.push(EventPoint::new(
            Point::new(0.0, 0.0),
            HashSet::from_iter(iter::once(0)),
            EventPointType::StartPoint,
        ));
        queue.push(EventPoint::new(
            Point::new(0.0, 0.0),
            HashSet::from_iter(iter::once(1)),
            EventPointType::StartPoint,
        ));

        assert_eq!(queue.len(), 1);
        unsafe {
            let val = &(*queue.arr[0].as_ptr()).value;
            assert_eq!(val.position(), Point::new(0.0, 0.0));
            let segments = val
                .segment_indices()
                .into_iter()
                .copied()
                .collect::<Vec<_>>();

            assert_eq!(segments.len(), 2);
            assert!(segments.contains(&0));
            assert!(segments.contains(&1));
            assert_eq!(val.event_type(), EventPointType::StartPoint);
        }
    }

    #[test]
    fn inserting_multiple_joint_start_segments_gets_merged() {
        let mut queue = EventQueue::new();

        queue.push(EventPoint::new(
            Point::new(0.0, 0.0),
            HashSet::from_iter(iter::once(0)),
            EventPointType::StartPoint,
        ));
        queue.push(EventPoint::new(
            Point::new(0.0, 0.0),
            HashSet::from_iter(iter::once(1)),
            EventPointType::StartPoint,
        ));
        queue.push(EventPoint::new(
            Point::new(12.0, -5.0),
            HashSet::from_iter(iter::once(5)),
            EventPointType::EndPoint,
        ));
        queue.push(EventPoint::new(
            Point::new(0.0, 0.0),
            HashSet::from_iter(iter::once(2)),
            EventPointType::StartPoint,
        ));

        assert_eq!(queue.len(), 2);
        unsafe {
            let val = &(*queue.arr[0].as_ptr()).value;
            assert_eq!(val.position(), Point::new(0.0, 0.0));
            let segments = val
                .segment_indices()
                .into_iter()
                .copied()
                .collect::<Vec<_>>();
            assert_eq!(segments.len(), 3);
            assert!(segments.contains(&0));
            assert!(segments.contains(&1));
            assert!(segments.contains(&2));
            assert_eq!(val.event_type(), EventPointType::StartPoint);
        }
    }

    #[test]
    fn inserting_multiple_different_types_of_intersecting_segments_dont_get_merged() {
        let mut queue = EventQueue::new();

        queue.push(EventPoint::new(
            Point::new(0.0, 0.0),
            HashSet::from_iter(iter::once(0)),
            EventPointType::StartPoint,
        ));
        queue.push(EventPoint::new(
            Point::new(0.0, 0.0),
            HashSet::from_iter(iter::once(1)),
            EventPointType::Intersection,
        ));
        queue.push(EventPoint::new(
            Point::new(12.0, -5.0),
            HashSet::from_iter(iter::once(5)),
            EventPointType::EndPoint,
        ));
        queue.push(EventPoint::new(
            Point::new(0.0, 0.0),
            HashSet::from_iter(iter::once(2)),
            EventPointType::EndPoint,
        ));

        assert_eq!(queue.len(), 4);
    }
}
