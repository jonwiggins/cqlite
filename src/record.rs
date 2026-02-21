// SQLite record serialization and deserialization.
// See: https://www.sqlite.org/fileformat2.html#record_format
//
// A record consists of a header followed by a body.
// The header starts with a varint giving the header size (including itself),
// then a varint per column giving the serial type.
// The body contains the column values in the same order.

use crate::error::{Result, RsqliteError};
use crate::format;
use crate::types::Value;
use crate::varint;

/// Decode a record from a byte buffer, returning the column values.
pub fn decode_record(buf: &[u8]) -> Result<Vec<Value>> {
    if buf.is_empty() {
        return Err(RsqliteError::Corrupt("empty record".into()));
    }

    // Read the header size varint.
    let (header_size, mut offset) = varint::read_varint(buf);
    let header_size = header_size as usize;

    if header_size > buf.len() {
        return Err(RsqliteError::Corrupt("record header size exceeds buffer".into()));
    }

    // Read serial types from the header.
    let mut serial_types = Vec::new();
    while offset < header_size {
        let (serial_type, n) = varint::read_varint(&buf[offset..]);
        serial_types.push(serial_type);
        offset += n;
    }

    // Decode values from the body.
    let mut body_offset = header_size;
    let mut values = Vec::with_capacity(serial_types.len());

    for &serial_type in &serial_types {
        let (value, size) = decode_value(&buf[body_offset..], serial_type)?;
        values.push(value);
        body_offset += size;
    }

    Ok(values)
}

/// Encode a record (list of values) into a byte buffer.
pub fn encode_record(values: &[Value]) -> Vec<u8> {
    // First, determine serial types and encode values.
    let mut serial_types = Vec::with_capacity(values.len());
    let mut encoded_values = Vec::with_capacity(values.len());

    for value in values {
        let (serial_type, encoded) = encode_value(value);
        serial_types.push(serial_type);
        encoded_values.push(encoded);
    }

    // Build the header: header_size varint + serial type varints.
    // We need to figure out the header size, which includes the header_size varint itself.
    // This is a chicken-and-egg: the header size varint length depends on the header size.
    let serial_type_bytes: usize = serial_types.iter().map(|&st| varint::varint_len(st)).sum();

    // Try header sizes starting from the smallest varint encoding.
    let header_size = find_header_size(serial_type_bytes);

    let body_size: usize = encoded_values.iter().map(|v| v.len()).sum();
    let mut buf = Vec::with_capacity(header_size + body_size);

    // Write header size varint.
    let mut tmp = [0u8; 9];
    let n = varint::write_varint(&mut tmp, header_size as u64);
    buf.extend_from_slice(&tmp[..n]);

    // Write serial type varints.
    for &st in &serial_types {
        let n = varint::write_varint(&mut tmp, st);
        buf.extend_from_slice(&tmp[..n]);
    }

    debug_assert_eq!(buf.len(), header_size);

    // Write body.
    for encoded in &encoded_values {
        buf.extend_from_slice(encoded);
    }

    buf
}

/// Find the correct header size, accounting for the fact that the header_size
/// varint itself is part of the header.
fn find_header_size(serial_type_bytes: usize) -> usize {
    // Start with assuming 1-byte header_size varint.
    for varint_size in 1..=9 {
        let total = varint_size + serial_type_bytes;
        if varint::varint_len(total as u64) == varint_size {
            return total;
        }
    }
    // Shouldn't happen with valid data.
    1 + serial_type_bytes
}

/// Decode a single value from the body given its serial type.
/// Returns (Value, bytes_consumed).
fn decode_value(buf: &[u8], serial_type: u64) -> Result<(Value, usize)> {
    match serial_type {
        0 => Ok((Value::Null, 0)),
        1 => {
            if buf.is_empty() {
                return Err(RsqliteError::Corrupt("truncated 1-byte integer".into()));
            }
            Ok((Value::Integer(buf[0] as i8 as i64), 1))
        }
        2 => {
            if buf.len() < 2 {
                return Err(RsqliteError::Corrupt("truncated 2-byte integer".into()));
            }
            let v = i16::from_be_bytes([buf[0], buf[1]]);
            Ok((Value::Integer(v as i64), 2))
        }
        3 => {
            if buf.len() < 3 {
                return Err(RsqliteError::Corrupt("truncated 3-byte integer".into()));
            }
            // Sign-extend from 24 bits.
            let v = ((buf[0] as i32) << 16) | ((buf[1] as i32) << 8) | (buf[2] as i32);
            // Sign-extend: if bit 23 is set, extend.
            let v = if v & 0x800000 != 0 {
                v | !0xFFFFFF
            } else {
                v
            };
            Ok((Value::Integer(v as i64), 3))
        }
        4 => {
            if buf.len() < 4 {
                return Err(RsqliteError::Corrupt("truncated 4-byte integer".into()));
            }
            let v = format::read_be_i32(buf, 0);
            Ok((Value::Integer(v as i64), 4))
        }
        5 => {
            if buf.len() < 6 {
                return Err(RsqliteError::Corrupt("truncated 6-byte integer".into()));
            }
            // Sign-extend from 48 bits.
            let mut bytes = [0u8; 8];
            bytes[2..8].copy_from_slice(&buf[..6]);
            let raw = u64::from_be_bytes(bytes);
            // Sign-extend from bit 47.
            let v = if raw & 0x0000_8000_0000_0000 != 0 {
                (raw | 0xFFFF_0000_0000_0000) as i64
            } else {
                raw as i64
            };
            Ok((Value::Integer(v), 6))
        }
        6 => {
            if buf.len() < 8 {
                return Err(RsqliteError::Corrupt("truncated 8-byte integer".into()));
            }
            let v = format::read_be_i64(buf, 0);
            Ok((Value::Integer(v), 8))
        }
        7 => {
            if buf.len() < 8 {
                return Err(RsqliteError::Corrupt("truncated 8-byte float".into()));
            }
            let v = format::read_be_f64(buf, 0);
            Ok((Value::Real(v), 8))
        }
        8 => Ok((Value::Integer(0), 0)),
        9 => Ok((Value::Integer(1), 0)),
        10 | 11 => Err(RsqliteError::Corrupt(format!(
            "reserved serial type {serial_type}"
        ))),
        n if n >= 12 && n % 2 == 0 => {
            // Blob of length (n-12)/2.
            let len = ((n - 12) / 2) as usize;
            if buf.len() < len {
                return Err(RsqliteError::Corrupt("truncated blob value".into()));
            }
            Ok((Value::Blob(buf[..len].to_vec()), len))
        }
        n if n >= 13 && n % 2 == 1 => {
            // Text of length (n-13)/2.
            let len = ((n - 13) / 2) as usize;
            if buf.len() < len {
                return Err(RsqliteError::Corrupt("truncated text value".into()));
            }
            let s = String::from_utf8_lossy(&buf[..len]).into_owned();
            Ok((Value::Text(s), len))
        }
        _ => Err(RsqliteError::Corrupt(format!(
            "invalid serial type {serial_type}"
        ))),
    }
}

/// Encode a single value, returning (serial_type, encoded_bytes).
fn encode_value(value: &Value) -> (u64, Vec<u8>) {
    match value {
        Value::Null => (0, Vec::new()),
        Value::Integer(i) => encode_integer(*i),
        Value::Real(f) => {
            let mut buf = [0u8; 8];
            format::write_be_f64(&mut buf, 0, *f);
            (7, buf.to_vec())
        }
        Value::Text(s) => {
            let bytes = s.as_bytes();
            let serial_type = (bytes.len() as u64) * 2 + 13;
            (serial_type, bytes.to_vec())
        }
        Value::Blob(b) => {
            let serial_type = (b.len() as u64) * 2 + 12;
            (serial_type, b.clone())
        }
    }
}

/// Encode an integer using the smallest representation.
fn encode_integer(i: i64) -> (u64, Vec<u8>) {
    if i == 0 {
        return (8, Vec::new());
    }
    if i == 1 {
        return (9, Vec::new());
    }
    if i >= -128 && i <= 127 {
        return (1, vec![i as u8]);
    }
    if i >= -32768 && i <= 32767 {
        let bytes = (i as i16).to_be_bytes();
        return (2, bytes.to_vec());
    }
    if i >= -8388608 && i <= 8388607 {
        let v = i as i32;
        return (3, vec![(v >> 16) as u8, (v >> 8) as u8, v as u8]);
    }
    if i >= -2147483648 && i <= 2147483647 {
        let bytes = (i as i32).to_be_bytes();
        return (4, bytes.to_vec());
    }
    if i >= -140737488355328 && i <= 140737488355327 {
        // 6 bytes: 48-bit signed integer.
        let bytes = i.to_be_bytes();
        return (5, bytes[2..8].to_vec());
    }
    // Full 8-byte integer.
    let bytes = i.to_be_bytes();
    (6, bytes.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_round_trip_null() {
        let values = vec![Value::Null];
        let encoded = encode_record(&values);
        let decoded = decode_record(&encoded).unwrap();
        assert_eq!(decoded, values);
    }

    #[test]
    fn test_round_trip_integers() {
        let values = vec![
            Value::Integer(0),
            Value::Integer(1),
            Value::Integer(-1),
            Value::Integer(42),
            Value::Integer(1000),
            Value::Integer(100000),
            Value::Integer(i64::MAX),
            Value::Integer(i64::MIN),
        ];
        let encoded = encode_record(&values);
        let decoded = decode_record(&encoded).unwrap();
        assert_eq!(decoded, values);
    }

    #[test]
    fn test_round_trip_real() {
        let values = vec![Value::Real(3.14), Value::Real(0.0), Value::Real(-1.5)];
        let encoded = encode_record(&values);
        let decoded = decode_record(&encoded).unwrap();
        assert_eq!(decoded, values);
    }

    #[test]
    fn test_round_trip_text() {
        let values = vec![
            Value::Text("hello".into()),
            Value::Text("".into()),
            Value::Text("hello world with spaces and unicode: \u{1F600}".into()),
        ];
        let encoded = encode_record(&values);
        let decoded = decode_record(&encoded).unwrap();
        assert_eq!(decoded, values);
    }

    #[test]
    fn test_round_trip_blob() {
        let values = vec![
            Value::Blob(vec![0xDE, 0xAD, 0xBE, 0xEF]),
            Value::Blob(vec![]),
        ];
        let encoded = encode_record(&values);
        let decoded = decode_record(&encoded).unwrap();
        assert_eq!(decoded, values);
    }

    #[test]
    fn test_round_trip_mixed() {
        let values = vec![
            Value::Null,
            Value::Integer(42),
            Value::Real(2.718),
            Value::Text("test".into()),
            Value::Blob(vec![1, 2, 3]),
        ];
        let encoded = encode_record(&values);
        let decoded = decode_record(&encoded).unwrap();
        assert_eq!(decoded, values);
    }

    #[test]
    fn test_integer_encoding_sizes() {
        // 0 and 1 use serial types 8 and 9 (zero bytes).
        let (st, data) = encode_integer(0);
        assert_eq!(st, 8);
        assert!(data.is_empty());

        let (st, data) = encode_integer(1);
        assert_eq!(st, 9);
        assert!(data.is_empty());

        // Small values: 1 byte.
        let (st, data) = encode_integer(42);
        assert_eq!(st, 1);
        assert_eq!(data.len(), 1);

        // Larger: 2 bytes.
        let (st, data) = encode_integer(1000);
        assert_eq!(st, 2);
        assert_eq!(data.len(), 2);

        // 3 bytes.
        let (st, data) = encode_integer(100_000);
        assert_eq!(st, 3);
        assert_eq!(data.len(), 3);
    }

    #[test]
    fn test_decode_empty_record() {
        let err = decode_record(&[]);
        assert!(err.is_err());
    }
}
