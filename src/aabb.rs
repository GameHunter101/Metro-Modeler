use crate::tensor_field::Point;

#[derive(Debug, Clone, Copy)]
pub struct BoundingBox {
    north: f32,
    northmost_point: Point,
    south: f32,
    southmost_point: Point,
    east: f32,
    eastmost_point: Point,
    west: f32,
    westmost_point: Point,
}

impl BoundingBox {
    pub fn new(points: &[Point]) -> BoundingBox {
        let mut northmost_point = points[0];
        let mut southmost_point = points[0];
        let mut eastmost_point = points[0];
        let mut westmost_point = points[0];

        for point in points {
            if point.x > eastmost_point.x {
                eastmost_point = *point;
            }
            if point.x < westmost_point.x {
                westmost_point = *point;
            }
            if point.y > northmost_point.y {
                northmost_point = *point;
            }
            if point.y < southmost_point.y {
                southmost_point = *point;
            }
        }

        BoundingBox {
            north: northmost_point.y,
            northmost_point,
            south: southmost_point.y,
            southmost_point,
            east: eastmost_point.x,
            eastmost_point,
            west: westmost_point.x,
            westmost_point,
        }
    }

    pub fn check_point_collision_with_padding(&self, point: Point, padding: f32) -> bool {
        let vertical_main_axis = self.north - self.south > self.east - self.west;
        let horiz_coeff = ((vertical_main_axis as i32) as f32).clamp(0.6, 1.0);
        let vert_coeff  = 1.6 - horiz_coeff;
        point.x >= self.west - padding * horiz_coeff
            && point.x <= self.east + padding * horiz_coeff
            && point.y >= self.south - padding * vert_coeff
            && point.y <= self.north + padding * vert_coeff
    }

    pub fn north(&self) -> f32 {
        self.north
    }

    pub fn south(&self) -> f32 {
        self.south
    }

    pub fn east(&self) -> f32 {
        self.east
    }

    pub fn west(&self) -> f32 {
        self.west
    }
}
