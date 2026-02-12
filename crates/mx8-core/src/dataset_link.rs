use std::fmt;

use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DatasetLink {
    Plain(String),
    Pinned { base: String, manifest_hash: String },
    Refresh(String),
}

#[derive(Debug, Error)]
pub enum DatasetLinkParseError {
    #[error("empty dataset link")]
    Empty,
    #[error("invalid pinned link; expected '@sha256:<hash>' suffix")]
    InvalidPinned,
}

impl DatasetLink {
    pub fn parse(input: &str) -> Result<Self, DatasetLinkParseError> {
        let input = input.trim();
        if input.is_empty() {
            return Err(DatasetLinkParseError::Empty);
        }

        if let Some((base, suffix)) = input.rsplit_once('@') {
            if suffix == "refresh" {
                return Ok(DatasetLink::Refresh(base.to_string()));
            }

            if let Some(hash) = suffix.strip_prefix("sha256:") {
                if hash.is_empty() {
                    return Err(DatasetLinkParseError::InvalidPinned);
                }
                return Ok(DatasetLink::Pinned {
                    base: base.to_string(),
                    manifest_hash: hash.to_string(),
                });
            }
        }

        Ok(DatasetLink::Plain(input.to_string()))
    }
}

impl fmt::Display for DatasetLink {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DatasetLink::Plain(s) => write!(f, "{s}"),
            DatasetLink::Pinned {
                base,
                manifest_hash,
            } => write!(f, "{base}@sha256:{manifest_hash}"),
            DatasetLink::Refresh(base) => write!(f, "{base}@refresh"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_plain() {
        let link = DatasetLink::parse("s3://bucket/prefix/").unwrap();
        assert_eq!(link, DatasetLink::Plain("s3://bucket/prefix/".to_string()));
    }

    #[test]
    fn parse_refresh() {
        let link = DatasetLink::parse("s3://bucket/prefix/@refresh").unwrap();
        assert_eq!(
            link,
            DatasetLink::Refresh("s3://bucket/prefix/".to_string())
        );
    }

    #[test]
    fn parse_pinned() {
        let link = DatasetLink::parse("s3://bucket/prefix/@sha256:abc").unwrap();
        assert_eq!(
            link,
            DatasetLink::Pinned {
                base: "s3://bucket/prefix/".to_string(),
                manifest_hash: "abc".to_string(),
            }
        );
    }
}
