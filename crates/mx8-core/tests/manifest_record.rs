use mx8_core::types::{ManifestRecord, ManifestRecordError};

#[test]
fn manifest_record_requires_location() {
    let r = ManifestRecord {
        sample_id: 0,
        location: "   ".to_string(),
        byte_offset: None,
        byte_length: None,
        decode_hint: None,
    };
    assert_eq!(r.validate(), Err(ManifestRecordError::EmptyLocation));
}

#[test]
fn manifest_record_requires_complete_byte_range() {
    let r = ManifestRecord {
        sample_id: 0,
        location: "s3://bucket/key".to_string(),
        byte_offset: Some(0),
        byte_length: None,
        decode_hint: None,
    };
    assert_eq!(r.validate(), Err(ManifestRecordError::PartialByteRange));
}

#[test]
fn manifest_record_rejects_zero_length_range() {
    let r = ManifestRecord {
        sample_id: 0,
        location: "s3://bucket/key".to_string(),
        byte_offset: Some(0),
        byte_length: Some(0),
        decode_hint: None,
    };
    assert_eq!(
        r.validate(),
        Err(ManifestRecordError::NonPositiveByteLength)
    );
}

#[test]
fn manifest_record_accepts_full_object_reference() {
    let r = ManifestRecord {
        sample_id: 0,
        location: "s3://bucket/key".to_string(),
        byte_offset: None,
        byte_length: None,
        decode_hint: Some("jpeg".to_string()),
    };
    assert_eq!(r.validate(), Ok(()));
}

#[test]
fn manifest_record_accepts_byte_range_reference() {
    let r = ManifestRecord {
        sample_id: 0,
        location: "s3://bucket/key".to_string(),
        byte_offset: Some(123),
        byte_length: Some(456),
        decode_hint: None,
    };
    assert_eq!(r.validate(), Ok(()));
}
