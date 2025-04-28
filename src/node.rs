
use std::fmt::Debug;

use slotmap::SlotMap;

use crate::{NodeIndex, Slots};

#[derive(Debug, Default, Clone)]
pub(crate) struct Node<K, V> {
    key: Option<K>,
    pub(crate) sub_keys: Vec<NodeIndex>,
    value: Option<V>,
    pub(crate) parent: Option<NodeIndex>
}




impl<K, V> Node<K, V>
{
    pub fn root() -> Self {
        Node {
            key: None,
            sub_keys: Vec::default(),
            value: None,
            parent: None
        }
    }
    pub fn keyed(key: K, parent: NodeIndex) -> Self {
        Self {
            sub_keys: Vec::default(),
            value: None,
            key: Some(key),
            parent: Some(parent)
        }
    }
    
}

impl<K, V> Node<K, V> {
    pub fn key(&self) -> &Option<K> {
        &self.key
    }
    pub fn is_root(&self) -> bool {
        self.key.is_none()
    }
    pub fn parent(&self) -> Option<NodeIndex> {
        self.parent.clone()
    }
    pub fn get(&self, key: &K, buffer: &Slots<K, V>) -> Option<&NodeIndex>
    where 
        K: Ord
    {
        let val = self.sub_keys.iter().find(|k| key == buffer[**k].key().as_ref().unwrap());
        Some(val?)
    }
  
    // pub fn insert(&mut self, key: K, value: NodeIndex) -> Option<NodeIndex>
    // where 
    //     K: Ord
    // {

    //     match self.bin_search(&key) {
    //         Ok(found) => {
             
    //             let (_ , old) = std::mem::replace(&mut self.sub_keys[found], (key, value));
    //             Some(old)
    //         }

    //         Err(found) => {
    //             self.sub_keys.insert(found, (key, value));

    //             None
    //         }
    //     }

    //     // let old = self.remove(&key);

    //     // self.sub_keys.push((key, value));

    //     // old
    // }
    pub fn bin_search(&self, reference: NodeIndex, buffer: &Slots<K, V>) -> Result<usize, usize>
    where 
        K: Ord,
    {
        let key = buffer[reference].key().as_ref().unwrap();
        self.sub_keys.binary_search_by(|node| {
            buffer[*node].key.as_ref().unwrap().cmp(key)
        })
    }
    // pub fn remove(&mut self, key: &K) -> Option<NodeIndex>
    // where 
    //     K: Ord
    // {
    //     Some(self.sub_keys.remove(self.bin_search(key).ok()?).1)
    // } 
    pub fn sub_key_len(&self) -> usize {
        self.sub_keys.len()
    }
    pub fn into_value(self) -> Option<V> {
        self.value
    }
    pub fn value(&self) -> &Option<V> {
        &self.value
    }
    pub fn value_mut(&mut self) -> &mut Option<V> {
        &mut self.value
    }
}



