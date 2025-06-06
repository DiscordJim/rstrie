# RsTrie
[![GitHub Actions Status](https://github.com/DiscordJim/rstrie/actions/workflows/rust.yml/badge.svg)](https://github.com/DiscordJim/rstrie/actions)
[![Crates.io Version](https://img.shields.io/crates/v/rstrie.svg)](https://crates.io/crates/rstrie)
[![Crates.io Downloads](https://img.shields.io/crates/d/rstrie.svg)](https://crates.io/crates/rstrie)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/DiscordJim/rstrie/blob/main/LICENSE)

A generalized Trie implementation for Rust. The implementation supports generic tries, the only requirement is that the key 'fragments,' or the pieces that compose the key, be able to iterated from a key and collected into a key. The library has no external dependencies when not in development mode. In development mode there are several dependencies for benchmarking and the like.

## Quickstart
Add the following line to your dependencies,
```toml
[dependencies]
rstrie = "0.5.0"
```
The following code demonstrates a simple `Trie`:
```rust
use rstrie::Trie;

let mut trie: Trie<char, usize> = Trie::new();
trie.insert("hello".chars(), 4);
trie.insert("hey".chars(), 5);

assert_eq!(trie.get("hello".chars()), Some(&4));
assert_eq!(trie.get("hey".chars()), Some(&5));
```

## Features
- A `Trie` that works with any data type.
- Supports the same interfaces as `HashMap`
- Longest prefix search
- Completions search
- Specialized methods for working with strings.
- No dependencies
- Full serialization/deseralization support (`serde`, `rkyv`)

cargo fuzz run ops fuzz/artifacts/ops/crash-c7261ee3e185eb226e890252921a51f5f4ebaaf3
## Fuzzing
```bash
$ cargo install cargo-fuzz
$ rustup default nightly
$ cargo fuzz run fuzz_target_1
```

## Examples
There are several examples contained within the examples folder.
- `examples/basic.rs` has a basic String trie.
- `examples/word_trie.rs` features a word trie that composes into sentences.
- `examples/bitrouting.rs` features an IP routing table.
- `examples/1984_trie.rs` features a trie that ingests the contents of George Orwell's 1984.

## Benchmarks
To run the benchmarks, you simply run the following command:
```bash
$ cargo bench
```