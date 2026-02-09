//! SQLite variable-length integer encoding/decoding.
//!
//! SQLite uses a variable-length integer format where:
//! - The high bit of each byte indicates if more bytes follow
//! - 1-9 bytes can encode values from 0 to 2^64-1
//! - The 9th byte (if present) uses all 8 bits

/// Read a varint from a byte slice, returning (value, bytes_consumed).
pub fn read_varint(buf: &[u8]) -> (u64, usize) {
    if buf.is_empty() {
        return (0, 0);
    }

    let mut value: u64 = 0;
    for i in 0..8 {
        if i >= buf.len() {
            return (value, i);
        }
        let byte = buf[i];
        value = (value << 7) | (byte & 0x7f) as u64;
        if byte & 0x80 == 0 {
            return (value, i + 1);
        }
    }

    // 9th byte: use all 8 bits
    if buf.len() > 8 {
        value = (value << 8) | buf[8] as u64;
        (value, 9)
    } else {
        (value, buf.len())
    }
}

/// Read a varint as i64.
pub fn read_varint_i64(buf: &[u8]) -> (i64, usize) {
    let (v, n) = read_varint(buf);
    (v as i64, n)
}

/// Write a varint to a buffer, returning the number of bytes written.
pub fn write_varint(buf: &mut [u8], value: u64) -> usize {
    if value <= 0x7f {
        buf[0] = value as u8;
        return 1;
    }
    if value <= 0x3fff {
        buf[0] = ((value >> 7) as u8) | 0x80;
        buf[1] = (value & 0x7f) as u8;
        return 2;
    }
    if value <= 0x1fffff {
        buf[0] = ((value >> 14) as u8) | 0x80;
        buf[1] = ((value >> 7) as u8) | 0x80;
        buf[2] = (value & 0x7f) as u8;
        return 3;
    }
    if value <= 0x0fffffff {
        buf[0] = ((value >> 21) as u8) | 0x80;
        buf[1] = ((value >> 14) as u8) | 0x80;
        buf[2] = ((value >> 7) as u8) | 0x80;
        buf[3] = (value & 0x7f) as u8;
        return 4;
    }
    if value <= 0x07ffffffff {
        buf[0] = ((value >> 28) as u8) | 0x80;
        buf[1] = ((value >> 21) as u8) | 0x80;
        buf[2] = ((value >> 14) as u8) | 0x80;
        buf[3] = ((value >> 7) as u8) | 0x80;
        buf[4] = (value & 0x7f) as u8;
        return 5;
    }
    if value <= 0x03ffffffffff {
        buf[0] = ((value >> 35) as u8) | 0x80;
        buf[1] = ((value >> 28) as u8) | 0x80;
        buf[2] = ((value >> 21) as u8) | 0x80;
        buf[3] = ((value >> 14) as u8) | 0x80;
        buf[4] = ((value >> 7) as u8) | 0x80;
        buf[5] = (value & 0x7f) as u8;
        return 6;
    }
    if value <= 0x01ffffffffffff {
        buf[0] = ((value >> 42) as u8) | 0x80;
        buf[1] = ((value >> 35) as u8) | 0x80;
        buf[2] = ((value >> 28) as u8) | 0x80;
        buf[3] = ((value >> 21) as u8) | 0x80;
        buf[4] = ((value >> 14) as u8) | 0x80;
        buf[5] = ((value >> 7) as u8) | 0x80;
        buf[6] = (value & 0x7f) as u8;
        return 7;
    }
    if value <= 0x00ffffffffffffff {
        buf[0] = ((value >> 49) as u8) | 0x80;
        buf[1] = ((value >> 42) as u8) | 0x80;
        buf[2] = ((value >> 35) as u8) | 0x80;
        buf[3] = ((value >> 28) as u8) | 0x80;
        buf[4] = ((value >> 21) as u8) | 0x80;
        buf[5] = ((value >> 14) as u8) | 0x80;
        buf[6] = ((value >> 7) as u8) | 0x80;
        buf[7] = (value & 0x7f) as u8;
        return 8;
    }

    // 9 bytes
    buf[0] = ((value >> 56) as u8) | 0x80;
    buf[1] = ((value >> 49) as u8) | 0x80;
    buf[2] = ((value >> 42) as u8) | 0x80;
    buf[3] = ((value >> 35) as u8) | 0x80;
    buf[4] = ((value >> 28) as u8) | 0x80;
    buf[5] = ((value >> 21) as u8) | 0x80;
    buf[6] = ((value >> 14) as u8) | 0x80;
    buf[7] = ((value >> 7) as u8) | 0x80;
    buf[8] = (value & 0xff) as u8;
    9
}

/// Calculate the number of bytes needed to encode a varint.
pub fn varint_len(value: u64) -> usize {
    if value <= 0x7f {
        1
    } else if value <= 0x3fff {
        2
    } else if value <= 0x1fffff {
        3
    } else if value <= 0x0fffffff {
        4
    } else if value <= 0x07ffffffff {
        5
    } else if value <= 0x03ffffffffff {
        6
    } else if value <= 0x01ffffffffffff {
        7
    } else if value <= 0x00ffffffffffffff {
        8
    } else {
        9
    }
}

/// Write a varint to a Vec.
pub fn write_varint_to_vec(vec: &mut Vec<u8>, value: u64) {
    let mut buf = [0u8; 9];
    let n = write_varint(&mut buf, value);
    vec.extend_from_slice(&buf[..n]);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_varint_roundtrip() {
        let test_values: Vec<u64> = vec![
            0,
            1,
            127,
            128,
            255,
            256,
            16383,
            16384,
            2097151,
            2097152,
            268435455,
            268435456,
            34359738367,
            34359738368,
            u32::MAX as u64,
            u64::MAX,
        ];
        for val in test_values {
            let mut buf = [0u8; 9];
            let written = write_varint(&mut buf, val);
            let (read_val, read_len) = read_varint(&buf[..written]);
            assert_eq!(val, read_val, "roundtrip failed for {}", val);
            assert_eq!(written, read_len, "length mismatch for {}", val);
        }
    }

    #[test]
    fn test_varint_len() {
        assert_eq!(varint_len(0), 1);
        assert_eq!(varint_len(127), 1);
        assert_eq!(varint_len(128), 2);
        assert_eq!(varint_len(16383), 2);
        assert_eq!(varint_len(16384), 3);
    }

    #[test]
    fn test_varint_single_byte() {
        let mut buf = [0u8; 9];
        assert_eq!(write_varint(&mut buf, 0), 1);
        assert_eq!(buf[0], 0);

        assert_eq!(write_varint(&mut buf, 127), 1);
        assert_eq!(buf[0], 127);
    }

    #[test]
    fn test_varint_two_bytes() {
        let mut buf = [0u8; 9];
        let n = write_varint(&mut buf, 128);
        assert_eq!(n, 2);
        let (v, len) = read_varint(&buf[..n]);
        assert_eq!(v, 128);
        assert_eq!(len, 2);
    }
}
