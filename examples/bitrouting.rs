use core::fmt;
use std::{
    hint::black_box,
    net::Ipv4Addr,
    str::FromStr,
    time::{Duration, Instant},
    u32,
};

use bitvec::{array::BitArray, order::Msb0, vec::BitVec};
use rand::{Rng, distr::Distribution};
use rand_distr::Zipf;
use rstrie::Trie;

/// How many routing table entries we should
/// be generating to compose our test.
pub const ROUTING_TABLE_SIZE: usize = 100_000;

/// How many requests we should run through our
/// routing table.
pub const SIMULATED_SERVED_REQUESTS: usize = 100_000;

/// Translates an [Ipv4Addr] into the bits and
/// returns the composed [Ip].
fn ip_translate(address: Ipv4Addr) -> Ip {
    Ip(BitVec::from(BitArray::from(address.to_bits())))
}

#[derive(PartialEq, Eq)]
pub struct Ip(BitVec<u32, Msb0>);

impl<'a> PartialEq<&'a str> for Ip {
    /// Facilitates comparison against [Ip] strings, making
    /// the code more readable.
    fn eq(&self, other: &&'a str) -> bool {
        let other = ip_translate(Ipv4Addr::from_str(other).unwrap());
        *self == other
    }
}

impl Ip {
    /// Creates an iterator of boolean values.
    pub fn iter(&self) -> impl Iterator<Item = bool> {
        self.0.iter().map(|i| *i)
    }
}

impl<'a> FromIterator<&'a bool> for Ip {
    /// Allows reconstructing [Ip] addresses from an iterator of bits
    /// by first flattening it into a vector and then checking that
    /// it is indeed 32 bits.
    fn from_iter<T: IntoIterator<Item = &'a bool>>(iter: T) -> Self {
        let collected = iter.into_iter().collect::<BitVec<u32, Msb0>>();
        assert_eq!(collected.len(), 32);
        Self(collected)
    }
}

impl fmt::Debug for Ip {
    /// Prints out the [Ip] address.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let vector = self.0.clone().into_vec()[0];
        Ipv4Addr::from_bits(vector).fmt(f)
    }
}

/// Iterates out bits according to the "targets". For instance,
/// if we pass &[192] then we will get 8 bits, &[192, 168] will
/// yield 16 bits, etc.
pub fn iterate_bits(targets: &[u8]) -> impl Iterator<Item = bool> {
    targets
        .into_iter()
        .map(|f| BitArray::<u8, Msb0>::new(*f).into_iter())
        .flatten()
}

/// The routing action to implement.
///
/// These were sort of just determined at random for the sake
/// of example.
#[derive(PartialEq, Eq, Debug)]
pub enum RoutingAction {
    Forward(usize),
    Restrict,
    Modification,
    Dropping,
}

impl Distribution<RoutingAction> for Zipf<f32> {
    /// Samples a routing action from a Zipfian distribution.
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> RoutingAction {
        // We only have four routing actions, so let us restrict the u8
        // to 0-3 inclusive.
        let wow = unchecked_f32_to_u8(self.sample(rng)) & 0x03;
        match wow {
            0 => RoutingAction::Forward(unchecked_f32_to_u8(self.sample(rng)) as usize),
            1 => RoutingAction::Dropping,
            2 => RoutingAction::Restrict,
            3 => RoutingAction::Modification,
            _ => unreachable!(),
        }
    }
}

/// Converts a [f32] to a unique [u8] value.
///
/// Obviously there will be significant collisions, but
/// this maps to a "bigger" space than the naive cast where
/// we drop all but the last few bits.
fn unchecked_f32_to_u8(value: f32) -> u8 {
    let value = value.to_bits();
    let a = (value & 0xFF000000) >> 24;
    let b = (value & 0x00FF0000) >> 16;
    let c = (value & 0x0000FF00) >> 8;
    let d = value & 0x000000FF;

    (a as u8) ^ (b as u8) ^ (c as u8) ^ (d as u8)
}

impl Distribution<Ip> for Zipf<f32> {
    /// Samples a new [Ip] from a Zipfian distribution.
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> Ip {
        let a: u8 = unchecked_f32_to_u8(self.sample(rng));
        let b: u8 = unchecked_f32_to_u8(self.sample(rng));
        let c: u8 = unchecked_f32_to_u8(self.sample(rng));
        let d: u8 = unchecked_f32_to_u8(self.sample(rng));

        let ipv4 = Ipv4Addr::new(a, b, c, d);
        ip_translate(ipv4)
    }
}

/// Search for the longest match given a [Trie] and a prefix which
/// is an array of [u8] types.
pub fn longest_match<'a>(
    trie: &'a Trie<bool, RoutingAction>,
    prefix: &[u8],
) -> Vec<(Ip, &'a RoutingAction)> {
    let finds = trie.completions::<_, &bool, Ip>(iterate_bits(prefix).collect::<Vec<_>>().iter());
    let first = finds.collect::<Vec<_>>();

    println!("Searching for prefix {prefix:?}...");
    println!("Found {} results.\nResults:", first.len());
    for (addy, action) in &first {
        println!("- [{addy:?}] Action: {action:?}")
    }
    println!("");

    first
}

pub fn main() {
    // let us run a quick test to verify the trie is working as expected.
    let john_ip = ip_translate(Ipv4Addr::from_str("192.168.1.253").unwrap());
    let sally_ip = ip_translate(Ipv4Addr::from_str("192.168.1.254").unwrap());
    let alice_ip = ip_translate(Ipv4Addr::from_str("192.169.1.254").unwrap());

    let mut trie = Trie::<bool, RoutingAction>::new();
    trie.insert(john_ip.iter(), RoutingAction::Forward(4));
    trie.insert(sally_ip.iter(), RoutingAction::Dropping);
    trie.insert(alice_ip.iter(), RoutingAction::Restrict);

    // Under the common prefix 192 there should be about three matches.
    let first = longest_match(&trie, &[192]);

    // there should be two entries.
    assert_eq!(first.len(), 3);
    assert_eq!(first[0].0, "192.168.1.253");
    assert_eq!(first[1].0, "192.168.1.254");
    assert_eq!(first[2].0, "192.169.1.254");

    // We will now supply a more specific prefix.
    let second = longest_match(&trie, &[192, 168]);

    // verify the lookup is correct.
    assert_eq!(second.len(), 2);
    assert_eq!(first[0].0, "192.168.1.253");
    assert_eq!(first[1].0, "192.168.1.254");

    // We will now build the routing table.
    let zipfian: Zipf<f32> = Zipf::new(100.0, 0.1).unwrap();

    let mut trie: Trie<_, _> = Trie::new();

    println!(
        "Building a randomized routing table of size {} from Zipfian distribution.",
        ROUTING_TABLE_SIZE
    );
    for _ in 0..ROUTING_TABLE_SIZE {
        let ip: Ip = zipfian.sample(&mut rand::rng());
        let action: RoutingAction = zipfian.sample(&mut rand::rng());
        trie.insert(ip.iter(), action);
    }
    println!("Built the routing table!");

    // Serve a bunch of requests.
    println!("Serving {} requests...", SIMULATED_SERVED_REQUESTS);
    let mut service_time = vec![];
    for _ in 0..SIMULATED_SERVED_REQUESTS {
        let candidate: Ip = zipfian.sample(&mut rand::rng());

        let start = Instant::now();
        black_box(trie.get(candidate.iter()));
        service_time.push(start.elapsed());
    }
    println!("Served requests...");

    let sum =
        (service_time.iter().sum::<Duration>().as_millis() as f64) / (service_time.len() as f64);
    println!("Average Request Service Time: {sum} ms");
}
