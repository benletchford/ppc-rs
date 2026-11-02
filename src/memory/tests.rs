use super::*;
use std::panic::{AssertUnwindSafe, catch_unwind};

#[test]
fn bulk_read_follows_visible_regions_across_overlays_and_gaps() {
    let mut mem = PpcSectionMem::new();
    mem.add_region(0x1000, vec![0, 1, 2, 3, 4, 5]);
    mem.add_region(0x1003, vec![30, 31]);

    let mut bytes = [0; 6];
    assert_eq!(mem.read_bytes_into(0x1000, &mut bytes), Some(()));
    assert_eq!(bytes, [0, 1, 2, 30, 31, 5]);

    let mut across_gap = [0xff; 4];
    assert_eq!(mem.read_bytes_into(0x1004, &mut across_gap), None);
    assert_eq!(across_gap, [31, 5, 0xff, 0xff]);

    mem.add_region(0x1006, vec![6, 7]);
    let mut adjacent = [0; 8];
    assert_eq!(mem.read_bytes_into(0x1000, &mut adjacent), Some(()));
    assert_eq!(adjacent, [0, 1, 2, 30, 31, 5, 6, 7]);
}

#[test]
fn bulk_read_preserves_wrapping_at_the_address_space_limit() {
    let mut mem = PpcSectionMem::new();
    mem.add_region(0x1000, vec![1, 2]);
    mem.add_region(0x1001, vec![3]);
    mem.add_region(u32::MAX, vec![0xaa]);
    mem.add_region(0, vec![0xbb]);

    let mut bytes = [0; 2];
    assert_eq!(mem.read_bytes_into(u32::MAX, &mut bytes), Some(()));
    assert_eq!(bytes, [0xaa, 0xbb]);
}

#[test]
fn readable_bytes_admits_only_one_visible_region() {
    let mut mem = PpcSectionMem::new();
    mem.add_readonly_region(0x1000, vec![0, 1, 2, 3]);
    mem.add_region(0x1004, vec![4, 5, 6, 7]);

    assert_eq!(mem.readable_bytes(0x1001, 3), Some(&[1, 2, 3][..]));
    assert_eq!(mem.readable_bytes(0x1004, 4), Some(&[4, 5, 6, 7][..]));
    assert_eq!(mem.readable_bytes(0x1003, 2), None);
    assert_eq!(mem.readable_bytes(0x1008, 1), None);
    assert_eq!(mem.readable_bytes(0x1000, 0), Some(&[][..]));

    mem.add_region(0x1002, vec![20, 21, 22]);
    assert_eq!(mem.readable_bytes(0x1000, 2), Some(&[0, 1][..]));
    assert_eq!(mem.readable_bytes(0x1001, 2), None);
    assert_eq!(mem.readable_bytes(0x1002, 3), Some(&[20, 21, 22][..]));
    assert_eq!(mem.readable_bytes(0x1004, 2), None);
    assert_eq!(mem.readable_bytes(0x1005, 3), Some(&[5, 6, 7][..]));
}

#[test]
fn readable_bytes_declines_wrapping_range_and_uses_new_mapping() {
    let mut mem = PpcSectionMem::new();
    mem.add_region(0x2000, vec![1, 2, 3, 4]);
    assert_eq!(mem.readable_bytes(0x2000, 4), Some(&[1, 2, 3, 4][..]));
    mem.add_readonly_region(0x2000, vec![9, 8, 7, 6]);
    assert_eq!(mem.readable_bytes(0x2000, 4), Some(&[9, 8, 7, 6][..]));

    mem.add_region(u32::MAX, vec![0xaa]);
    assert_eq!(mem.read_u8(u32::MAX), Some(0xaa));
    assert_eq!(mem.readable_bytes(u32::MAX, 1), None);
}

#[test]
fn public_instruction_tokens_share_one_non_reusing_namespace() {
    let first = try_allocate_instruction_cache_token().expect("first token");
    let second = try_allocate_instruction_cache_token().expect("second token");
    let mut mem = PpcSectionMem::new();
    mem.add_readonly_region(0x1000, 0x6000_0000u32.to_be_bytes().to_vec());
    let mapping = mem.instruction_cache_token(0x1000).expect("mapping token");

    assert_ne!(first, second);
    assert_ne!(first, mapping);
    assert_ne!(second, mapping);
}

#[test]
fn exhausted_instruction_token_allocation_cannot_mutate_a_mapping() {
    let mut mem = PpcSectionMem::new();
    let available = AtomicU64::new(42);
    mem.add_mapped_region(
        0x1000,
        0x6000_0000u32.to_be_bytes().to_vec(),
        false,
        &available,
    );
    let exhausted = AtomicU64::new(u64::MAX);

    let result = catch_unwind(AssertUnwindSafe(|| {
        mem.add_mapped_region(
            0x1000,
            0x4E80_0020u32.to_be_bytes().to_vec(),
            false,
            &exhausted,
        );
    }));

    assert!(result.is_err());
    assert_eq!(mem.region_count(), 1);
    assert_eq!(mem.read_instruction_u32_be(0x1000), Some(0x6000_0000));
    assert_eq!(mem.instruction_cache_token(0x1000), Some(42));
    assert_eq!(exhausted.load(Ordering::Relaxed), u64::MAX);
}
