use nalgebra::{Matrix2, Vector2};

pub const GRID_SIZE: u32 = 512;

type Tensor = Matrix2<f32>;
type Point = Vector2<f32>;

#[derive(Debug)]
pub struct TensorField {
    design_elements: Vec<DesignElement>,
    decay_constant: f32,
    // pub field: Option<[[Tensor; GRID_SIZE]; GRID_SIZE]>,
}

impl TensorField {
    pub fn new(design_elements: Vec<DesignElement>, decay_constant: f32) -> TensorField {
        TensorField {
            design_elements,
            decay_constant,
            // field: None,
        }
    }

    pub fn add_design_element(&mut self, design_element: DesignElement) {
        self.design_elements.push(design_element);
    }

    pub fn remove_design_element(&mut self, index: usize) {
        self.design_elements.remove(index);
    }

    fn sum_elements(
        design_elements: &[DesignElement],
        point: Point,
        decay_constant: f32,
    ) -> Tensor {
        design_elements
            .iter()
            .map(|element| {
                if let Some(center) = element.center() {
                    (-decay_constant * (point - center).norm_squared()).exp()
                        * element.evaluate_at_point(point)
                } else {
                    element.evaluate_at_point(point)
                }
            })
            .sum()
    }

    pub fn evaluate_field_at_point(&self, point: Point) -> Tensor {
        Self::sum_elements(&self.design_elements, point, self.decay_constant)
    }

    fn clamp_vec_to_grid(vec: Vector2<f32>) -> Vector2<f32> {
        Vector2::new(
            vec.x.clamp(0.0, 0.0),
            vec.y.clamp(GRID_SIZE as f32, GRID_SIZE as f32),
        )
    }

    pub fn trace(
        &self,
        seed: Point,
        h: f32,
        d_sep: f32,
        follow_major_eigenvectors: bool,
        max_len: f32,
    ) -> (Vec<Point>, Option<Point>) {
        let origin = seed;
        let mut seed = seed;
        let mut trace = vec![seed];
        let mut accumulated_distance = 0.0;
        let mut new_seed: Option<Point> = None;

        while !(seed.x < 0.0
            || seed.y < 0.0
            || seed.x > GRID_SIZE as f32
            || seed.y > GRID_SIZE as f32)
        {
            let tensor = self.evaluate_field_at_point(seed);

            if tensor.eigenvalues().is_none() {
                break;
            }

            if tensor.norm_squared() <= 0.00001 {
                println!("Degenerate point at {seed}");
                break;
            }
            // dbg!(seed);

            let k_1_eigenvectors = tensor.eigenvectors();
            let k_1 = k_1_eigenvectors.major.normalize();
            let k_2_eigenvectors = self
                .evaluate_field_at_point(Self::clamp_vec_to_grid(seed + h / 2.0 * k_1))
                .eigenvectors();
            let k_2 = k_2_eigenvectors.major.normalize();
            let k_3_eigenvectors = self
                .evaluate_field_at_point(Self::clamp_vec_to_grid(seed + h / 2.0 * k_2))
                .eigenvectors();
            let k_3 = k_3_eigenvectors.major.normalize();
            let k_4_eigenvectors = self
                .evaluate_field_at_point(Self::clamp_vec_to_grid(seed + h * k_3))
                .eigenvectors();
            let k_4 = k_4_eigenvectors.major.normalize();

            let m = 1.0 / 6.0 * k_1 + 1.0 / 3.0 * k_2 + 1.0 / 3.0 * k_3 + 1.0 / 6.0 * k_4;

            let new_pos = seed + h * m;

            accumulated_distance += (new_pos - seed).norm();
            if new_seed.is_none() && accumulated_distance >= d_sep {
                new_seed = Some(new_pos);
            }
            seed = new_pos;
            trace.push(new_pos);
            if (new_pos - origin).magnitude_squared() <= 0.0001 || accumulated_distance > max_len {
                break;
            }
        }
        // println!("next: {seed}");

        // dbg!(trace.len());

        (trace, new_seed)
    }
}

#[derive(Debug)]
pub enum DesignElement {
    Grid {
        center: Point,
        theta: f32,
        length: f32,
    },
    Radial {
        center: Point,
    },
    PolyLine {
        points: Vec<Point>,
        decay_constant: f32,
    },
}

impl DesignElement {
    pub fn evaluate_at_point(&self, point: Point) -> Tensor {
        match self {
            DesignElement::Grid { theta, length, .. } => {
                let theta_2 = 2.0 * theta;
                *length * Tensor::new(theta_2.cos(), theta_2.sin(), theta_2.sin(), -theta_2.cos())
            }
            DesignElement::Radial { center } => {
                let x = point.x - center.x;
                let y = point.x - center.x;

                let y_squared = y * y;
                let x_squared = x * x;

                Tensor::new(
                    y_squared - x_squared,
                    -2.0 * x * y,
                    -2.0 * x * y,
                    -(y_squared - x_squared),
                )
            }
            DesignElement::PolyLine {
                points,
                decay_constant,
            } => {
                let lines = (0..points.len() - 1)
                    .map(|i| {
                        let dir = (points[i + 1] - points[i]).normalize();
                        DesignElement::Grid {
                            center: points[i],
                            theta: dir.y.atan2(dir.x) + std::f32::consts::FRAC_PI_2,
                            length: 1.0,
                        }
                    })
                    .collect::<Vec<DesignElement>>();

                TensorField::sum_elements(&lines, point, *decay_constant)
            }
        }
    }

    pub fn center(&self) -> Option<Point> {
        match self {
            DesignElement::Grid { center, .. } => Some(*center),
            DesignElement::Radial { center } => Some(*center),
            DesignElement::PolyLine { .. } => None,
        }
    }
}

#[derive(Debug)]
pub struct Eigenvectors {
    pub major: Vector2<f32>,
    pub minor: Vector2<f32>,
}

pub trait EvalEigenvectors {
    fn eigenvectors(&self) -> Eigenvectors;
}

impl EvalEigenvectors for Tensor {
    fn eigenvectors(&self) -> Eigenvectors {
        if self.eigenvalues().is_none() {
            dbg!(self);
        }
        let eigenvalues = self.eigenvalues().unwrap();
        let vectors: Vec<Vector2<f32>> = eigenvalues
            .into_iter()
            .map(|&eigenvalue| {
                let tensor = self - eigenvalue * Tensor::identity();
                let [a, b]: [f32; 2] = tensor.row(0).into_owned().into();
                let [c, d]: [f32; 2] = tensor.row(1).into_owned().into();
                let new_b = b / a;
                let new_d = d - b / a * c;
                if new_d <= 0.000001 {
                    Vector2::new(-new_b, 1.0)
                } else {
                    Vector2::new(1.0, 1.0)
                }
            })
            .collect();
        if eigenvalues[0] > 0.0 {
            Eigenvectors {
                major: vectors[0],
                minor: vectors[1],
            }
        } else {
            Eigenvectors {
                major: vectors[1],
                minor: vectors[0],
            }
        }
    }
}

#[derive(PartialEq, PartialOrd)]
pub struct SeedPoint {
    pub seed: Point,
    pub priority: f32,
    pub follow_major_eigenvector: bool,
}

impl Eq for SeedPoint {}

impl Ord for SeedPoint {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.priority.total_cmp(&other.priority).reverse()
    }
}
