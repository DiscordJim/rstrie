use rstrie::Trie;

#[derive(Debug)]
pub enum Value {
    A,
    B,
    C,
    D
}

pub fn main() {
    let mut tree = Trie::<char, Value>::new();
    tree.insert("hello".chars(), Value::A);
    tree.insert("world".chars(), Value::B);
    tree.insert("cool".chars(), Value::C);
    tree.insert("hey".chars(), Value::D);

    println!("Hello: {:?}", tree.get("hello".chars()).unwrap());
}