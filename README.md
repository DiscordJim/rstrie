# RsTrie
A generalized Trie implementation for Rust. The implementation supports generic tries, the only requirement is that the key 'fragments,' or the pieces that compose the key, be able to iterated from a key and collected into a key. The library has no external dependencies when not in development mode. In development mode there are several dependencies for benchmarking and the like.

```rust
use rstrie::Trie;

let mut trie: Trie<char, usize> = Trie::new();
trie.insert("hello".chars(), 4);
trie.insert("hey".chars(), 5);

assert_eq!(trie.get("hello".chars()), Some(&4));
assert_eq!(trie.get("hey".chars(), Some(&5)));
```

# Features
- A `Trie` that works with any data type.
- Supports the same interfaces as `HashMap`
- Longest prefix search
- Completions search
- Specialized methods for working with strings.
- No dependencies
- Full serialization/deseralization support (`serde`, `rkyv`)


## Fuzzing
```bash
$ cargo install cargo-fuzz
```

## TODO
- ARBITRARY
- SERDE
- RKYV
- Add size tests where we add duplicate keys, etc.

## Examples
There are several examples contained within the examples folder.
- `examples/basic.rs` has a basic String trie.
- `examples/word_trie.rs` features a word trie that composes into sentences.
- `examples/bitrouting.rs` features an IP routing table.



## Benchmarks
To run the benchmarks, you simply run the following command:
```bash
$ cargo bench
```