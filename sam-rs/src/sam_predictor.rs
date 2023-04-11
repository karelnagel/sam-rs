pub struct SamPredictor {}

impl SamPredictor {
    pub fn new() -> SamPredictor {
        SamPredictor {}
    }

    pub fn set_image(&self, _input: &str) -> String {
        String::from("Hello, world!")
    }
    pub fn get_image_embeddings(&self, _input: &str) -> String {
        String::from("Hello, world!")
    }
    pub fn apply_coords(&self, _input: &str) -> String {
        String::from("Hello, world!")
    }
}
