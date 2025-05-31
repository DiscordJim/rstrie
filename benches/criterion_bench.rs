//! This code is mostly identical to the Cloudflare triehard implementation which
//! can be found here:
//! https://github.com/cloudflare/trie-hard/blob/main/benches/criterion_bench.rs
//!
//! The code there is also, by their own admittance, a rip off of the benchmark
//! suite for [`radix_trie`](https://github.com/michaelsproul/rust_radix_trie/blob/master/Cargo.toml)
//!
//! Very slight modifications were made to the file in order to make it compatible
//! with this library.

use std::collections::HashSet;

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use rstrie::Trie;

const OW_1984: &str = include_str!("../data/1984.txt");
const SUN_RISING: &str = include_str!("../data/sun-rising.txt");
const RANDOM: &str = include_str!("../data/random.txt");

fn get_big_text() -> Vec<&'static str> {
    OW_1984
        .split(|c: char| c.is_whitespace())
        .collect::<HashSet<_>>()
        .into_iter()
        .collect()
}

fn get_small_text() -> Vec<&'static str> {
    SUN_RISING
        .split(|c: char| c.is_whitespace())
        .collect::<HashSet<_>>()
        .into_iter()
        .collect()
}

fn get_random_text() -> Vec<&'static str> {
    RANDOM
        .split(|c: char| c.is_whitespace())
        .collect::<HashSet<_>>()
        .into_iter()
        .collect()
}

fn make_trie<'a>(words: &[&'a str]) -> Trie<char, ()> {
    words.iter().copied().map(|i| (i.chars(), ())).collect()
}

fn trie_insert_big(b: &mut Criterion) {
    let words = get_big_text();
    b.bench_function("rstrie insert - big", |b| {
        b.iter(|| make_trie(black_box(&words)))
    });
}

fn trie_insert_small(b: &mut Criterion) {
    let words = get_small_text();
    b.bench_function("rstrie insert - small", |b| {
        b.iter(|| make_trie(black_box(&words)))
    });
}

fn generate_samples<'a>(hits: &[&'a str], hit_percent: i32) -> Vec<&'a str> {
    let roulette_inc = hit_percent as f64 / 100.;
    let mut roulette = 0.;

    let mut result = get_random_text().to_owned();
    let mut hit_iter = hits.iter().cycle().copied();

    for w in result.iter_mut() {
        roulette += roulette_inc;
        if roulette >= 1. {
            roulette -= 1.;
            *w = hit_iter.next().unwrap();
        }
    }

    result
}

macro_rules! bench_percents_impl {
    ( [ $( ($size:expr, $percent:expr ), )+ ] ) => {$(
        paste::paste! {
            // Trie Hard
            fn [< trie_get_ $size _ $percent >] (b: &mut Criterion) {
                let words = [< get_ $size _text >]();
                let trie = make_trie(&words);
                let samples = generate_samples(&words, $percent);
                b.bench_function(
                    concat!(
                        "rstrie get - ",
                        stringify!($size),
                        " - ",
                        stringify!($percent),
                        "%"
                    ), |b| {
                    b.iter(|| {
                        samples.iter()
                            .filter_map(|w| trie.get(black_box(w[..].chars())))
                            .count()
                    })
                });
            }
        }


    )+};

    (  _groups [ $( ($size:expr, $percent:expr ), )+ ] ) => {
        paste::paste! {
            criterion_group!(
                get_benches,
                $(
                    [< trie_get_ $size _ $percent >],
                )+
            );
        }
    };
}

macro_rules! cartesian_impl {
    ($out:tt [] $b:tt $init_b:tt) => {
        bench_percents_impl!($out);
        bench_percents_impl!(_groups $out);
    };
    ($out:tt [$a:expr, $($at:tt)*] [] $init_b:tt) => {
        cartesian_impl!($out [$($at)*] $init_b $init_b);
    };
    ([$($out:tt)*] [$a:expr, $($at:tt)*] [$b:expr, $($bt:tt)*] $init_b:tt) => {
        cartesian_impl!([$($out)* ($a, $b),] [$a, $($at)*] [$($bt)*] $init_b);
    };
}

macro_rules! bench_get_percents {
    ([$($size:tt)*], [$($percent:tt)*]) => {
        cartesian_impl!([] [$($size)*,] [$($percent)*,] [$($percent)*,]);
    };
}

bench_get_percents!([big, small], [100, 75, 50, 25, 10, 5, 2, 1]);

criterion_group!(insert_benches, trie_insert_big, trie_insert_small);

criterion_main!(get_benches, insert_benches);
