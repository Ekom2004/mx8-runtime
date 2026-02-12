use mx8_core::types::WorkRange;

#[test]
fn work_range_is_half_open() {
    let r = WorkRange {
        start_id: 10,
        end_id: 20,
        epoch: 0,
        seed: 0,
    };
    assert!(r.contains(10));
    assert!(r.contains(19));
    assert!(!r.contains(20));
    assert_eq!(r.len(), 10);
    assert!(!r.is_empty());
}

#[test]
fn empty_work_range() {
    let r = WorkRange {
        start_id: 5,
        end_id: 5,
        epoch: 0,
        seed: 0,
    };
    assert!(r.is_empty());
    assert_eq!(r.len(), 0);
}
