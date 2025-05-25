use rstrie::{CompletionIter, Trie};

/// The text of 1984 from Project Gutenberg.
pub const TEXT: &'static str = include_str!("../data/1984.txt");

/// Turns a string into a cleaned iterator of characters.
pub fn wordify(text: &str) -> std::vec::IntoIter<char> {
    text.trim()
        .to_lowercase()
        .chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace())
        .collect::<Vec<_>>()
        .into_iter()
}

/// Find a completion over a [Trie] for a particular search string.
fn find_completion(trie: &Trie<char, ()>, search_str: &str) {
    println!("Finding completions for {search_str}...");
    // Find the completions for this phrase.
    let completions: CompletionIter<'_, _, (), String> =
        trie.completions::<_, char, String>(wordify(search_str));

    println!("Completions found:");
    for (completion, _) in completions {
        println!("- {}", completion)
    }
}


/// Parses 1984 into a [Trie].
fn parse_1984() -> Trie<char, ()> {
    println!(
        "Beginning parsing of 1984. It has {} total characters.",
        TEXT.len()
    );

    // Perform cleaning steps.
    let splitted = TEXT
        .split("\n")
        .map(|f| f.trim())
        .filter(|f| f.len() > 0)
        .map(|f| f.split("."))
        .flatten()
        .filter(|f| f.len() > 0)
        .map(|f| f.to_string())
        .collect::<Vec<String>>()
        .to_vec();

    // Report on how many sentences there are.
    println!("It has a total of {} sentences.", splitted.len());

    // Report on average sentence length.
    let total_length = splitted.iter().map(|f| f.len()).sum::<usize>();
    println!(
        "Sentences are on average about {:.2} words long.",
        (total_length as f64) / (splitted.len() as f64)
    );

    splitted
        .into_iter()
        .map(|f| wordify(&f).collect::<Vec<_>>())
        // Add the null element in order to collect it.
        .map(|f| (f, ()))
        .collect()
}

pub fn main() {
    let tree = parse_1984();
    find_completion(&tree, "from the for");
    find_completion(&tree, "there");
}
