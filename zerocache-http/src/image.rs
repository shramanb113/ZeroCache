use zerocache_ports::ImageInput;

/// Turns a `data:<mime_type>;base64,<data>` string into an `ImageInput`.
/// This is the wire-shape translation step for the image-embedding routes --
/// the same architectural role `deserialize_input` plays for text in wire.rs
/// -- so a caller-facing parsing failure surfaces as a plain `String` the
/// handler turns into a 400, never reaching AppError (which is reserved for
/// store/provider failures, not malformed request bodies).
pub fn parse_data_uri(uri: &str) -> Result<ImageInput, String> {
    let rest = uri
        .strip_prefix("data:")
        .ok_or_else(|| "expected a data URI starting with 'data:'".to_string())?;
    let (mime_type, data) = rest
        .split_once(";base64,")
        .ok_or_else(|| "expected ';base64,' in data URI".to_string())?;
    if mime_type.is_empty() {
        return Err("data URI is missing a mime type".to_string());
    }
    Ok(ImageInput {
        mime_type: mime_type.to_string(),
        data: data.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_well_formed_data_uri() {
        let image = parse_data_uri("data:image/png;base64,aGVsbG8=").unwrap();
        assert_eq!(image.mime_type, "image/png");
        assert_eq!(image.data, "aGVsbG8=");
    }

    #[test]
    fn rejects_a_uri_missing_the_data_prefix() {
        let result = parse_data_uri("image/png;base64,aGVsbG8=");
        assert!(result.is_err());
    }

    #[test]
    fn rejects_a_uri_missing_the_base64_marker() {
        let result = parse_data_uri("data:image/png,aGVsbG8=");
        assert!(result.is_err());
    }

    #[test]
    fn rejects_an_empty_string() {
        let result = parse_data_uri("");
        assert!(result.is_err());
    }
}
