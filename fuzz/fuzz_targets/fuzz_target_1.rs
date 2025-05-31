#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use rstrie::Trie;

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let mut unstructured = Unstructured::new(data);

    let mut trie: Trie<char, u64> = Trie::arbitrary(&mut unstructured).unwrap();
    trie.insert(Vec::<char>::arbitrary(&mut unstructured).unwrap(), u64::arbitrary(&mut unstructured).unwrap());
    // fuzzed code goes here
});
