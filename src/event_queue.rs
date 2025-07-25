use std::ptr::NonNull;

use futures::task::UnsafeFutureObj;
use rayon::slice;

use crate::street_graph::EventPoint;

pub struct EventQueue {
    arr: Vec<EventPoint>,
    len: usize,
}

impl EventQueue {
    fn new() -> Self {
        Self {
            arr: Vec::new(),
            len: 0,
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
        if start_index == 0 {
            return;
        }

        let parent_index = (start_index - 1) / 2;

        let current_val = &self.arr[start_index];
        let parent_val = &self.arr[parent_index];

        if current_val > parent_val {
            self.swap(start_index, parent_index);
            self.sift_up(parent_index);
        }
    }

    fn push(&mut self, event_point: EventPoint) {
        self.arr.push(event_point);
        self.len += 1;
        self.sift_up(self.len - 1);
    }

    fn sift_down(&mut self, start_index: usize) {
        if start_index == self.len - 1 {
            return;
        }

        let left_child_index = 2 * start_index + 1;
        let right_child_index = left_child_index + 1;

        if left_child_index > self.len - 1 {
            return;
        }

        let (max_index, max_val) = if right_child_index > self.len - 1
            || self.arr[left_child_index] > self.arr[right_child_index]
        {
            (left_child_index, &self.arr[left_child_index])
        } else {
            (right_child_index, &self.arr[right_child_index])
        };

        if &self.arr[start_index] < max_val {
            self.swap(start_index, max_index);
            self.sift_down(start_index);
        }
    }

    fn pop(&mut self) {
        if self.len == 1 {
            self.arr.pop();
        } else {
            self.swap(0, self.len - 1);
            self.arr.pop();
            self.len -= 1;
            self.sift_down(0);
        }
    }

    fn to_vec(&self) -> Vec<EventPoint> {
        self.arr.clone()
    }

    fn to_slice(&self) -> &[EventPoint] {
        &self.arr
    }

    fn len(&self) -> usize {
        self.len
    }
}

type Link<T> = Option<NonNull<Node<T>>>;

struct RBTree<T: Ord + std::fmt::Debug> {
    root: Link<T>,
}

impl<T: Ord + std::fmt::Debug> RBTree<T> {
    fn new() -> Self {
        Self { root: None }
    }

    fn insert(&mut self, element: T) {
        unsafe {
            if let Some(root) = self.root {
                Node::insert(root, element);
            } else {
                let boxed_node = Node::new(element, None);

                self.root = Some(boxed_node);
            }
        }
    }
}

#[derive(Debug, PartialEq, PartialOrd, Clone, Copy)]
enum Color {
    Red,
    Black,
}

struct Node<T: Ord + std::fmt::Debug> {
    value: T,
    color: Color,
    left: Link<T>,
    right: Link<T>,
    parent: Link<T>,
}

impl<T: Ord + std::fmt::Debug> Node<T> {
    fn new(element: T, parent: Link<T>) -> NonNull<Node<T>> {
        unsafe {
            NonNull::new_unchecked(Box::into_raw(Box::new(Node {
                value: element,
                color: Color::Red,
                left: None,
                right: None,
                parent,
            })))
        }
    }

    fn insert(node: NonNull<Node<T>>, element: T) {
        unsafe {
            if element < (*node.as_ptr()).value {
                if let Some(left) = (*node.as_ptr()).left {
                    Self::insert(left, element);
                } else {
                    let boxed_node = Node::new(element, Some(node));
                    (*node.as_ptr()).left = Some(boxed_node);
                }
            } else {
                if let Some(right) = (*node.as_ptr()).right {
                    Self::insert(right, element);
                } else {
                    let boxed_node = Node::new(element, Some(node));
                    (*node.as_ptr()).right = Some(boxed_node);
                }
            }
        }
    }

    fn rebalance(node: NonNull<Node<T>>) {
        unsafe {
            let potential_parent = (*node.as_ptr()).parent;
            if let Some(parent) = potential_parent {
                let parent_color = (*parent.as_ptr()).color;
                if parent_color == Color::Black {
                    // Case 1
                    return;
                } else {
                    let potential_grandparent = (*parent.as_ptr()).parent;
                    if let Some(grandparent) = potential_grandparent {
                        let (potential_uncle, parent_is_left_of_grandparent) =
                            if (*grandparent.as_ptr()).left == (*node.as_ptr()).parent {
                                ((*grandparent.as_ptr()).right, true)
                            } else {
                                ((*grandparent.as_ptr()).left, false)
                            };

                        let uncle_color = if let Some(uncle) = potential_uncle {
                            (*uncle.as_ptr()).color
                        } else {
                            Color::Black
                        };

                        if parent_color == Color::Red && uncle_color == Color::Red {
                            // Case 2
                            (*parent.as_ptr()).color = Color::Black;
                            (*potential_uncle.unwrap().as_ptr()).color = Color::Black;
                            (*grandparent.as_ptr()).color = Color::Red;
                            Self::rebalance(grandparent);
                        } else if parent_color == Color::Red && uncle_color == Color::Black {
                            let node_is_left_of_parent = (*parent.as_ptr()).left == Some(node);
                            if node_is_left_of_parent != parent_is_left_of_grandparent {
                                // Case 5
                                if node_is_left_of_parent {
                                    Self::rotate_right(node, parent_is_left_of_grandparent);
                                } else {
                                    Self::rotate_left(node, parent_is_left_of_grandparent);
                                }
                            }
                            // case 6
                            if node_is_left_of_parent {
                                Self::rotate_right(node, parent_is_left_of_grandparent);
                            } else {
                                Self::rotate_left(node, parent_is_left_of_grandparent);
                            }
                            (*node.as_ptr()).color = Color::Black;
                            if parent_is_left_of_grandparent {
                                (*(*node.as_ptr()).right.unwrap().as_ptr()).color = Color::Red
                            } else {
                                (*(*node.as_ptr()).left.unwrap().as_ptr()).color = Color::Red
                            }
                        }
                    } else {
                        if parent_color == Color::Red {
                            // Case 4
                            (*parent.as_ptr()).color = Color::Black;
                            return;
                        }
                    }
                }
            } else {
                // Case 3
                return;
            }
        }
    }

    fn rotate_left(node: NonNull<Node<T>>, on_left_side_of_grandparent: bool) {
        unsafe {
            let new_top = node;
            let new_left = (*node.as_ptr()).parent.unwrap();
            (*new_left.as_ptr()).right = (*new_top.as_ptr()).left;
            (*new_top.as_ptr()).left = Some(new_left);
            (*new_top.as_ptr()).parent = (*new_left.as_ptr()).parent;
            (*new_left.as_ptr()).parent = Some(new_top);
            if let Some(new_parent) = (*new_top.as_ptr()).parent {
                if on_left_side_of_grandparent {
                    (*new_parent.as_ptr()).left = Some(new_top);
                } else {
                    (*new_parent.as_ptr()).right = Some(new_top);
                }
            }
        }
    }

    fn rotate_right(node: NonNull<Node<T>>, on_left_side_of_grandparent: bool) {
        unsafe {
            let new_top = node;
            let new_right = (*node.as_ptr()).parent.unwrap();
            (*new_right.as_ptr()).left = (*new_top.as_ptr()).right;
            (*new_top.as_ptr()).right = Some(new_right);
            (*new_top.as_ptr()).parent = (*new_right.as_ptr()).parent;
            (*new_right.as_ptr()).parent = Some(new_top);
            if let Some(new_parent) = (*new_top.as_ptr()).parent {
                if on_left_side_of_grandparent {
                    (*new_parent.as_ptr()).left = Some(new_top);
                } else {
                    (*new_parent.as_ptr()).right = Some(new_top);
                }
            }
        }
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

        assert_eq!(queue.to_slice(), &expected_body);
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

        assert_eq!(queue.to_slice(), &expected_body);
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

        assert_eq!(queue.to_slice(), &expected_body);
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

        queue.pop();
        queue.pop();

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

        assert_eq!(queue.to_slice(), &expected_body);
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

        assert_eq!(queue.to_slice(), &expected_body);
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

        let expected_body = vec![EventPoint::new(
            Point::new(0.0, 0.0),
            HashSet::from_iter(vec![0, 1]),
            EventPointType::StartPoint,
        )];

        assert_eq!(queue.len(), 1);
        assert_eq!(queue.to_slice(), &expected_body);
    }
}
