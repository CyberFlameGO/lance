use std::cmp::max;
use arrow_schema::ArrowError;
use rand::{Rng, SeedableRng};

/// Partition the `arr` for the first `topk` elements are the smallest
pub fn find_min_k(values: &mut [f32], indices: &mut [u64], topk: usize) -> Result<(), ArrowError> {
    let mut rng = rand::rngs::StdRng::from_entropy();
    assert!(topk > 0);
    if values.len() <= topk {
        return Ok(());
    }
    let mut low = 0;  // inclusive
    let mut high = values.len();  // exclusive
    // TODO handle worse case scenarios and fallback to nlog(n) sort
    let mut limit = max(10, values.len());
    loop {
        let pivot: usize = rng.gen_range(low..high);  // choose random pivot
        let (left, right) = partition(values, indices, low, high, pivot)?;
        if limit <= 10 {
            println!("Low {low}; High {high}; Left {left}; Pivot {pivot}");
        }
        if left == topk {
            return Ok(());
        } else if left > topk {
            high = left;
        } else if right > topk {
            return Ok(());
        } else if right <= topk {
            low = right;
        }
        limit -= 1;
        assert!(limit > 0);
    }
}

/// Changes arr to arr_prime, indices to indices_prime, and return a tuple of indices (L, R)
/// s.t. the following must all be true
/// for i in L..R { arr_prime[i] == arr[pivot] }
/// for i in low..L { arr_prime[i] < arr[pivot] }
/// for i in R..high { arr_prime[i] > arr[pivot] }
/// arr_prime and indices_prime maintain alignment
pub(crate) fn partition(arr: &mut [f32], indices: &mut [u64], low: usize, high: usize, pivot: usize) -> Result<(usize, usize), ArrowError> {
    if !(low <= pivot && pivot < high) {
        return Err(ArrowError::ComputeError("Pivot must be between low and high".to_string()));
    }
    arr.swap(high - 1, pivot);
    indices.swap(high - 1, pivot);
    let mut pivot_cursor = high - 1;
    if pivot_cursor == 0 {
        return Ok((0,0));
    }

    // sweep from both ends til the two cursors meet
    let mut left_cursor = low;
    let mut right_cursor = pivot_cursor - 1;

    loop {
        // sweep from the left
        while left_cursor < pivot_cursor {
            if &arr[left_cursor] < &arr[pivot_cursor] {
                left_cursor += 1;
            } else if &arr[left_cursor] == &arr[pivot_cursor] {
                if right_cursor <= left_cursor {
                    break;
                }
                arr.swap(right_cursor, pivot_cursor - 1);
                indices.swap(right_cursor, pivot_cursor - 1);
                arr.swap(left_cursor, right_cursor);
                indices.swap(left_cursor, right_cursor);
                right_cursor -= 1;
                pivot_cursor -= 1;
            } else {
                break;
            }
        }

        // sweep from the right
        while right_cursor > left_cursor {
            if &arr[pivot_cursor] < &arr[right_cursor] {
                right_cursor -= 1;
            } else if &arr[pivot_cursor] == &arr[right_cursor] {
                if right_cursor <= left_cursor {
                    break;
                }
                arr.swap(right_cursor, pivot_cursor - 1);
                indices.swap(right_cursor, pivot_cursor - 1);
                right_cursor -= 1;
                pivot_cursor -= 1;
            } else {
                break;
            }
        }

        if left_cursor >= right_cursor {
            // if they meet then we're done
            break;
        } else {
            // otherwise we swap elements to the right partition and continue
            arr.swap(left_cursor, right_cursor);
            indices.swap(left_cursor, right_cursor);
            left_cursor += 1;
            right_cursor -= 1;
        }
    }
    for i in 0..(high - pivot_cursor) {
        let src = left_cursor + i;
        let dest = pivot_cursor + i;
        arr.swap(src, dest);
        indices.swap(src, dest);
    }
    Ok((left_cursor, left_cursor + (high - pivot_cursor)))
}


#[cfg(test)]
mod tests {
    use std::iter::repeat_with;
    use rand::{Rng, rngs, SeedableRng};
    use crate::index::ann::select::{find_min_k, partition};

    pub fn generate_random_vec(n: usize) -> Vec<f32> {
        let mut rng = rand::thread_rng();
        repeat_with(|| rng.gen::<f32>())
            .take(n)
            .collect::<Vec<f32>>()
    }

    #[test]
    fn test_partition() {
        let mut arr = generate_random_vec(100);
        let mut indices: Vec<u64>= (0..arr.len() as u64).collect();
        let pivot = 3;
        let pivot_val = arr[pivot];
        let len = arr.len();
        let (left, pivot) = partition(&mut arr, &mut indices, 0, len, pivot).unwrap();
        for v in arr[0..left].iter() {
            assert!(*v < pivot_val);
        }
        assert_eq!(pivot_val, arr[left]);
        for v in arr[left + 1..].iter() {
            assert!(*v >= pivot_val);
        }
        let mut indices: Vec<u64> = (0..arr.len() as u64).collect();

        arr = vec![1.0];
        assert_eq!(0, partition(&mut arr, &mut indices, 0, 1, 0).unwrap().0);

        arr = vec![1.0].repeat(10);
        let len = arr.len();
        assert_eq!(0, partition(&mut arr, &mut indices, 0, len, 4).unwrap().0);

        arr = (1..8).map(|int: i16| f32::from(int)).collect();
        let len = arr.len();
        assert_eq!(4, partition(&mut arr, &mut indices, 0, len, 4).unwrap().0);

        arr = (1..8).rev().map(|int: i16| f32::from(int)).collect();
        let len = arr.len();
        assert_eq!(2, partition(&mut arr, &mut indices, 0, len, 4).unwrap().0);

        arr = vec![1.0, 5.0, 10.0, 5.0, 4.0, 5.0, 5.0, 2.0, 9.0, 8.0, 5.0];
        let len = arr.len();
        assert_eq!((3, len-4), partition(&mut arr, &mut indices, 0, len, 1).unwrap());
    }

    #[test]
    fn test_find_min_k() {
        let mut rows: Vec<u64> = (0..100).collect();
        let mut rng = rngs::StdRng::from_entropy();
        let mut scores:[f32; 100] = [0.0; 100];
        rng.fill(&mut scores[..]);

        let mut sorted = scores.clone();
        sorted.sort_unstable_by(|a, b| a.total_cmp(b));

        let topk = 10;
        find_min_k(&mut scores, &mut rows, topk).unwrap();

        for i in 0..topk {
            assert!(scores[i] <= sorted[topk])
        }
    }
}