use std::borrow::Borrow;

use rstrie::{CompletionIter, Trie};

/// This will be our [Word] object which represents a word
/// from a sentence.
#[derive(Clone, Hash, PartialEq, Eq, Debug, PartialOrd, Ord)]
pub struct Word(String);

impl Word {
    pub fn wordify(text: &str) -> std::vec::IntoIter<Word> {
        text.to_lowercase()
            .chars()
            .filter(|c| c.is_alphanumeric() || c.is_whitespace())
            .collect::<String>()
            .split_ascii_whitespace()
            .map(|inner| Word(inner.to_string()))
            .collect::<Vec<_>>()
            .into_iter()
    }
}

#[allow(dead_code)]
#[derive(Debug)]
pub struct Sentence(String);

impl<Q> FromIterator<Q> for Sentence
where
    Q: Borrow<Word>,
{
    fn from_iter<T: IntoIterator<Item = Q>>(iter: T) -> Self {
        Self(
            iter.into_iter()
                .map(|innie| {
                    let r: &Word = innie.borrow();
                    r.0.clone()
                })
                .collect::<Vec<_>>()
                .join(" "),
        )
    }
}

fn find_completion(trie: &Trie<Word, ()>, search_str: &str) {
    println!("Finding completions for {search_str}...");
    // Find the completions for this phrase.
    let completions: CompletionIter<'_, Word, (), Sentence> =
        trie.completions::<_, Word, Sentence>(Word::wordify(search_str));

    println!("Completions found:");
    for (completion, _) in completions {
        println!("- {}", completion.0)
    }
}

pub fn main() {
    println!("running simple trie...");
    let mut tree = Trie::<Word, ()>::new();
    tree.insert(Word::wordify("I like cats."), ());
    tree.insert(Word::wordify("I like dogs."), ());
    tree.insert(Word::wordify("I like dogs on most days."), ());
    tree.insert(Word::wordify("I like"), ());

    tree.insert(Word::wordify("I like cats sometimes."), ());
    tree.insert(Word::wordify("I like cats somedays."), ());

    find_completion(&tree, "I like");
    find_completion(&tree, "I like cats");
}
