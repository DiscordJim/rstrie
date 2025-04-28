use std::str::Chars;

pub(crate) struct Iter<V> {
    values: Vec<Option<V>>,
    position: usize,
}

impl<V> Iter<V> {
    pub fn new(values: Vec<Option<V>>) -> Self {
        Self {
            values,
            position: 0
        }
    }
}

impl<V> Iterator for Iter<V> {
    type Item = V;

    fn next(&mut self) -> Option<Self::Item> {
        if self.position >= self.values.len() {
            None
        } else {
            self.position += 1;
            self.values[self.position - 1].take()
        }
    }
}


