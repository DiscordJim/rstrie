#![no_main]

use libfuzzer_sys::fuzz_target;
use arbitrary::Arbitrary;
use rstrie::Trie;
use std::collections::HashSet;
use std::collections::HashMap;

#[derive(Arbitrary, Debug)]
pub enum Ops {
    Insert(Vec<u8>, u8),
    Delete(Vec<u8>),
    Get(Vec<u8>),
    Defragment,
    RemoveEntry(Vec<u8>)
}

fuzz_target!(|data: Vec<Ops>| {
    let mut trie = Trie::<u8, u8>::new();

    let mut twin = HashMap::<Vec<u8>, u8>::new();

    for datum in data {
        match datum {
            Ops::Insert(key, ops) => {
                assert_eq!(trie.insert(key.clone(), ops), twin.insert(key, ops));
            }
            Ops::Delete(key) => {
                assert_eq!(trie.remove(&key), twin.remove(&key));
            }
            Ops::RemoveEntry(key) => {
                let twin_entry = twin.remove_entry(&key);
                let trie_entry = trie.remove_entry::<_, _, Vec<_>>(key);
                assert_eq!(twin_entry, trie_entry);
            }
            Ops::Get(key) => {
                assert_eq!(trie.get(&key), twin.get(&key));
            }
            Ops::Defragment => {
                trie.shrink_to_fit();
            }
        }
        
        assert_eq!(trie.len(), twin.len());
    }

});
