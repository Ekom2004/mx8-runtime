use prost::Message;

use mx8_proto::v0::{GetManifestResponse, NodeCaps, RegisterNodeRequest};

#[test]
fn register_node_request_prost_roundtrip() {
    let msg = RegisterNodeRequest {
        job_id: "job".to_string(),
        node_id: "node".to_string(),
        caps: Some(NodeCaps {
            max_fetch_concurrency: 1,
            max_decode_concurrency: 2,
            max_inflight_bytes: 3,
            max_ram_bytes: 4,
        }),
    };

    let bytes = msg.encode_to_vec();
    let decoded = RegisterNodeRequest::decode(bytes.as_slice()).unwrap();
    assert_eq!(decoded, msg);
}

#[test]
fn get_manifest_response_prost_roundtrip() {
    let msg = GetManifestResponse {
        manifest_bytes: vec![1, 2, 3],
        schema_version: 7,
    };

    let bytes = msg.encode_to_vec();
    let decoded = GetManifestResponse::decode(bytes.as_slice()).unwrap();
    assert_eq!(decoded, msg);
}
