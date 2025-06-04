use nalgebra::Vector2;
use tensor_field::{DesignElement, Eigenvectors};

mod tensor_field;

fn main() {
    let element = DesignElement::Grid {
        center: Vector2::new(0.0, 0.0),
        width: 2.0,
        height: 1.0,
    };

    let tensor = element.evaluate_at_point(Vector2::new(0.0, 0.0));
    let eigenvectors = tensor.eigenvectors();

    println!("{tensor}");
    println!("{}, {}", eigenvectors[0], eigenvectors[1]);
}
