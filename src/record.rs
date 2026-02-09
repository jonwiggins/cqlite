use crate::error::{Result, RsqliteError};
use crate::format;
use crate::types::Value;
use crate::varint;

/// Serial type codes for the SQLite record format.
/// Reference: https://www.sqlite.org/fileformat2.html#record_format
#[derive(Debug, Clone, Copy)]
pub enum SerialType {
    Null,    // 0
    Int8,    // 1
    Int16,   // 2
    Int24,   // 3
    Int32,   // 4
    Int48,   // 5
    Int64,   // 6
    Float64, // 7
    Zero,    // 8 (integer value 0)
    One,     // 9 (integer value 1)
    // 10, 11 reserved
    Blob(u64), // N >= 12, even: blob of (N-12)/2 bytes
    Text(u64), // N >= 13, odd: text of (N-13)/2 bytes
}

impl SerialType {
    /// Determine the serial type for a value.
    pub fn for_value(value: &Value) -> u64 {
        match value {
            Value::Null => 0,
            Value::Integer(i) => {
                let i = *i;
                if i == 0 {
                    8
                } else if i == 1 {
                    9
                } else if (-128..=127).contains(&i) {
                    1
                } else if (-32768..=32767).contains(&i) {
                    2
                } else if (-8388608..=8388607).contains(&i) {
                    3
                } else if (-2147483648..=2147483647).contains(&i) {
                    4
                } else if (-140737488355328..=140737488355327).contains(&i) {
                    5
                } else {
                    6
                }
            }
            Value::Real(_) => 7,
            Value::Text(s) => (s.len() as u64) * 2 + 13,
            Value::Blob(b) => (b.len() as u64) * 2 + 12,
        }
    }

    /// Parse a serial type code.
    pub fn from_code(code: u64) -> SerialType {
        match code {
            0 => SerialType::Null,
            1 => SerialType::Int8,
            2 => SerialType::Int16,
            3 => SerialType::Int24,
            4 => SerialType::Int32,
            5 => SerialType::Int48,
            6 => SerialType::Int64,
            7 => SerialType::Float64,
            8 => SerialType::Zero,
            9 => SerialType::One,
            n if n >= 12 && n % 2 == 0 => SerialType::Blob(n),
            n if n >= 13 && n % 2 == 1 => SerialType::Text(n),
            _ => SerialType::Null, // reserved codes treated as NULL
        }
    }

    /// Content size in bytes for this serial type.
    pub fn content_size(code: u64) -> usize {
        match code {
            0 => 0,
            1 => 1,
            2 => 2,
            3 => 3,
            4 => 4,
            5 => 6,
            6 => 8,
            7 => 8,
            8 => 0,
            9 => 0,
            n if n >= 12 && n % 2 == 0 => ((n - 12) / 2) as usize,
            n if n >= 13 && n % 2 == 1 => ((n - 13) / 2) as usize,
            _ => 0,
        }
    }
}

/// Deserialize a record payload into a list of values.
pub fn deserialize_record(payload: &[u8]) -> Result<Vec<Value>> {
    if payload.is_empty() {
        return Ok(vec![]);
    }

    // Read header size varint
    let (header_size, header_size_len) = varint::read_varint(payload);
    let header_size = header_size as usize;

    if header_size > payload.len() {
        return Err(RsqliteError::Corruption(
            "Record header size exceeds payload".into(),
        ));
    }

    // Parse serial types from header
    let mut offset = header_size_len;
    let mut serial_types = Vec::new();
    while offset < header_size {
        let (code, n) = varint::read_varint(&payload[offset..]);
        serial_types.push(code);
        offset += n;
    }

    // Read values from body
    let mut data_offset = header_size;
    let mut values = Vec::with_capacity(serial_types.len());

    for &code in &serial_types {
        let size = SerialType::content_size(code);
        if data_offset + size > payload.len() {
            return Err(RsqliteError::Corruption(
                "Record data exceeds payload".into(),
            ));
        }
        let data = &payload[data_offset..data_offset + size];
        let value = read_value(code, data)?;
        values.push(value);
        data_offset += size;
    }

    Ok(values)
}

/// Read a single value given its serial type code and data bytes.
fn read_value(code: u64, data: &[u8]) -> Result<Value> {
    match SerialType::from_code(code) {
        SerialType::Null => Ok(Value::Null),
        SerialType::Int8 => Ok(Value::Integer(data[0] as i8 as i64)),
        SerialType::Int16 => Ok(Value::Integer(format::read_be_u16(data) as i16 as i64)),
        SerialType::Int24 => {
            let v = ((data[0] as i32) << 16) | ((data[1] as i32) << 8) | (data[2] as i32);
            // Sign-extend from 24 bits
            let v = if v & 0x800000 != 0 { v | !0xFFFFFF } else { v };
            Ok(Value::Integer(v as i64))
        }
        SerialType::Int32 => {
            let v = format::read_be_u32(data) as i32;
            Ok(Value::Integer(v as i64))
        }
        SerialType::Int48 => {
            let v = format::read_be_u48(data);
            // Sign-extend from 48 bits
            let v = if v & 0x800000000000 != 0 {
                (v | 0xFFFF000000000000) as i64
            } else {
                v as i64
            };
            Ok(Value::Integer(v))
        }
        SerialType::Int64 => Ok(Value::Integer(format::read_be_i64(data))),
        SerialType::Float64 => Ok(Value::Real(format::read_be_f64(data))),
        SerialType::Zero => Ok(Value::Integer(0)),
        SerialType::One => Ok(Value::Integer(1)),
        SerialType::Blob(_) => Ok(Value::Blob(data.to_vec())),
        SerialType::Text(_) => {
            let s = String::from_utf8_lossy(data).into_owned();
            Ok(Value::Text(s))
        }
    }
}

/// Serialize a record (list of values) into a payload.
pub fn serialize_record(values: &[Value]) -> Vec<u8> {
    // Build header (serial type codes)
    let mut header = Vec::new();
    let serial_types: Vec<u64> = values.iter().map(SerialType::for_value).collect();

    // We need to compute the header size first (including the header size varint itself)
    let mut type_bytes = Vec::new();
    for &st in &serial_types {
        varint::write_varint_to_vec(&mut type_bytes, st);
    }

    // Header size = size of header_size varint + type bytes
    // This is tricky because the varint size depends on the total
    let total_header_size_estimate =
        varint::varint_len((type_bytes.len() + 1) as u64) + type_bytes.len();
    let header_size_varint_len = varint::varint_len(total_header_size_estimate as u64);
    let total_header_size = header_size_varint_len + type_bytes.len();

    varint::write_varint_to_vec(&mut header, total_header_size as u64);
    header.extend_from_slice(&type_bytes);

    // Build body
    let mut body = Vec::new();
    for (value, &code) in values.iter().zip(&serial_types) {
        write_value(&mut body, value, code);
    }

    let mut result = header;
    result.append(&mut body);
    result
}

/// Write a value's data bytes.
fn write_value(buf: &mut Vec<u8>, value: &Value, _code: u64) {
    match value {
        Value::Null => {}
        Value::Integer(i) => {
            let i = *i;
            if i == 0 || i == 1 {
                // codes 8, 9: zero-length
            } else if (-128..=127).contains(&i) {
                buf.push(i as u8);
            } else if (-32768..=32767).contains(&i) {
                let bytes = (i as i16).to_be_bytes();
                buf.extend_from_slice(&bytes);
            } else if (-8388608..=8388607).contains(&i) {
                buf.push((i >> 16) as u8);
                buf.push((i >> 8) as u8);
                buf.push(i as u8);
            } else if (-2147483648..=2147483647).contains(&i) {
                let bytes = (i as i32).to_be_bytes();
                buf.extend_from_slice(&bytes);
            } else if (-140737488355328..=140737488355327).contains(&i) {
                buf.push((i >> 40) as u8);
                buf.push((i >> 32) as u8);
                buf.push((i >> 24) as u8);
                buf.push((i >> 16) as u8);
                buf.push((i >> 8) as u8);
                buf.push(i as u8);
            } else {
                let bytes = i.to_be_bytes();
                buf.extend_from_slice(&bytes);
            }
        }
        Value::Real(f) => {
            let bytes = f.to_be_bytes();
            buf.extend_from_slice(&bytes);
        }
        Value::Text(s) => {
            buf.extend_from_slice(s.as_bytes());
        }
        Value::Blob(b) => {
            buf.extend_from_slice(b);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_roundtrip() {
        let values = vec![
            Value::Integer(42),
            Value::Text("hello".into()),
            Value::Null,
            Value::Real(3.14),
            Value::Blob(vec![1, 2, 3]),
            Value::Integer(0),
            Value::Integer(1),
        ];
        let payload = serialize_record(&values);
        let decoded = deserialize_record(&payload).unwrap();
        assert_eq!(values, decoded);
    }

    #[test]
    fn test_serial_types() {
        assert_eq!(SerialType::for_value(&Value::Null), 0);
        assert_eq!(SerialType::for_value(&Value::Integer(0)), 8);
        assert_eq!(SerialType::for_value(&Value::Integer(1)), 9);
        assert_eq!(SerialType::for_value(&Value::Integer(42)), 1);
        assert_eq!(SerialType::for_value(&Value::Integer(1000)), 2);
        assert_eq!(SerialType::for_value(&Value::Real(1.0)), 7);
        assert_eq!(SerialType::for_value(&Value::Text("ab".into())), 17); // 2*2+13
        assert_eq!(SerialType::for_value(&Value::Blob(vec![1, 2])), 16); // 2*2+12
    }

    #[test]
    fn test_empty_record() {
        let values: Vec<Value> = vec![];
        let payload = serialize_record(&values);
        let decoded = deserialize_record(&payload).unwrap();
        assert_eq!(decoded.len(), 0);
    }

    #[test]
    fn test_large_integer() {
        let values = vec![Value::Integer(i64::MAX), Value::Integer(i64::MIN)];
        let payload = serialize_record(&values);
        let decoded = deserialize_record(&payload).unwrap();
        assert_eq!(values, decoded);
    }
}
