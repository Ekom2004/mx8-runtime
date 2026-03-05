#!/usr/bin/env python3
import os

import mx8
import torch


def ensure(cond: bool, msg: str) -> None:
    if not cond:
        raise SystemExit(msg)


def read_first_batch(
    dataset_link: str,
    manifest_store: str,
    manifest_path: str,
    tokenizer_json: str,
    policy: str,
):
    loader = mx8.text(
        dataset_link,
        store=manifest_store,
        manifest=manifest_path,
        tokenizer=tokenizer_json,
        seq_len=8,
        stride=8,
        batch=16,
        return_mask=True,
        on_decode_error=policy,
        add_bos=False,
        add_eos=False,
        prefetch=2,
        queue=8,
        inflight=64 * 1024 * 1024,
    )
    try:
        batch = next(iter(loader))
        return batch.token_ids, batch.attention_mask, batch.sample_ids
    finally:
        loader.close()


def main() -> None:
    dataset_link = os.environ["MX8_DATASET_LINK"]
    manifest_store = os.environ["MX8_MANIFEST_STORE_ROOT"]
    manifest_path = os.environ["MX8_DEV_MANIFEST_PATH"]
    tokenizer_json = os.environ["MX8_TEXT_TOKENIZER_JSON"]

    # skip-policy run should succeed even with one invalid UTF-8 sample in dataset.
    a_ids, a_mask, a_sids = read_first_batch(
        dataset_link, manifest_store, manifest_path, tokenizer_json, "skip"
    )
    b_ids, b_mask, b_sids = read_first_batch(
        dataset_link, manifest_store, manifest_path, tokenizer_json, "skip"
    )

    ensure(torch.equal(a_ids, b_ids), "expected deterministic token_ids across identical runs")
    ensure(torch.equal(a_mask, b_mask), "expected deterministic attention_mask across identical runs")
    ensure(torch.equal(a_sids, b_sids), "expected deterministic sample_ids across identical runs")

    ensure(a_ids.dtype == torch.int64, f"expected token_ids int64, got {a_ids.dtype}")
    ensure(a_mask.dtype == torch.bool, f"expected attention_mask bool, got {a_mask.dtype}")
    ensure(a_sids.dtype == torch.int64, f"expected sample_ids int64, got {a_sids.dtype}")
    ensure(a_ids.ndim == 2 and a_ids.shape[1] == 8, f"unexpected token_ids shape {tuple(a_ids.shape)}")
    ensure(a_mask.shape == a_ids.shape, "attention_mask shape must match token_ids shape")
    ensure(a_sids.ndim == 1 and a_sids.shape[0] == a_ids.shape[0], "sample_ids shape mismatch")

    # error-policy run must fail on invalid UTF-8 sample.
    loader_err = mx8.text(
        dataset_link,
        store=manifest_store,
        manifest=manifest_path,
        tokenizer=tokenizer_json,
        seq_len=8,
        stride=8,
        batch=16,
        return_mask=True,
        on_decode_error="error",
        add_bos=False,
        add_eos=False,
        prefetch=2,
        queue=8,
        inflight=64 * 1024 * 1024,
    )
    try:
        got_error = False
        try:
            _ = next(iter(loader_err))
        except Exception:
            got_error = True
        ensure(got_error, "decode_error_policy='error' should fail on invalid UTF-8 sample")
    finally:
        loader_err.close()

    print("[mx8] m6_text_gate OK")


if __name__ == "__main__":
    main()
