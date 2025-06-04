use rstrie::Trie;

// [Insert([], 163), Insert([111, 111, 111, 111, 111, 111, 111, 111, 255], 133), Delete([111, 111, 111, 111]), Get([]), Get([255, 255, 17, 7, 7, 42, 255, 7, 255, 255, 255, 255, 2, 255, 255, 43, 43, 43, 43, 255, 255]), Get([201, 201, 201, 161, 255, 255]), Delete([111, 111, 111, 111, 111, 111, 111, 111, 255]), Delete([]), Delete([]), Delete([]), Insert([], 65), Delete([111, 111, 220, 255])]

fn main() {
    let mut wow = Trie::<u8, u8>::new();
    wow.insert(vec![], 163);

    println!("AYY: {:?}", wow.remove::<Vec<u8>,_>(vec![]));

    println!("WOW: {:?}", wow);

}