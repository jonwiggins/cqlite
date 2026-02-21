// SQLite variable-length integer encoding/decoding.
// See: https://www.sqlite.org/fileformat2.html#varint
//
// A varint is 1-9 bytes. For bytes 1-8, the high bit indicates whether
// more bytes follow. The 9th byte (if present) uses all 8 bits.

/// Read a varint from a byte slice. Returns (value, bytes_consumed).
///
/// # Panics
/// Panics if the slice is empty or truncated mid-varint.
pub fn read_varint(buf: &[u8]) -> (u64, usize) {
    // The 9th byte (if reached) uses all 8 bits.
    // Bytes 1-8 use 7 bits each (high bit = continuation flag).
    let mut value: u64 = 0;
    for (i, &byte) in buf.iter().enumerate().take(8) {
        value = (value << 7) | (byte & 0x7F) as u64;
        if byte & 0x80 == 0 {
            return (value, i + 1);
        }
    }
    // 9th byte: all 8 bits are data
    let byte = buf[8];
    value = (value << 8) | byte as u64;
    (value, 9)
}

/// Read a varint, returning None if the buffer is too short.
pub fn try_read_varint(buf: &[u8]) -> Option<(u64, usize)> {
    if buf.is_empty() {
        return None;
    }
    for i in 0..8 {
        if i >= buf.len() {
            return None;
        }
        let byte = buf[i];
        if byte & 0x80 == 0 {
            return Some(read_varint(buf));
        }
    }
    if buf.len() < 9 {
        return None;
    }
    Some(read_varint(buf))
}

/// Write a varint into a buffer. Returns the number of bytes written.
///
/// The buffer must be at least 9 bytes long.
pub fn write_varint(buf: &mut [u8], value: u64) -> usize {
    if value <= 0x7F {
        buf[0] = value as u8;
        return 1;
    }

    // Determine the number of bytes needed.
    let nbytes = varint_len(value);

    if nbytes == 9 {
        // 9-byte encoding: first 8 bytes carry 7 bits each, 9th carries 8 bits.
        let mut v = value;
        buf[8] = v as u8;
        v >>= 8;
        for i in (0..8).rev() {
            buf[i] = 0x80 | (v & 0x7F) as u8;
            v >>= 7;
        }
    } else {
        // 1-8 byte encoding: high bit = continuation, low 7 bits = data.
        let mut v = value;
        // Last byte has no continuation bit.
        buf[nbytes - 1] = (v & 0x7F) as u8;
        v >>= 7;
        for i in (0..nbytes - 1).rev() {
            buf[i] = 0x80 | (v & 0x7F) as u8;
            v >>= 7;
        }
    }

    nbytes
}

/// Return how many bytes a varint encoding of `value` requires.
pub fn varint_len(value: u64) -> usize {
    if value <= 0x7F {
        1
    } else if value <= 0x3FFF {
        2
    } else if value <= 0x1F_FFFF {
        3
    } else if value <= 0x0FFF_FFFF {
        4
    } else if value <= 0x07_FFFF_FFFF {
        5
    } else if value <= 0x03FF_FFFF_FFFF {
        6
    } else if value <= 0x01_FFFF_FFFF_FFFF {
        7
    } else if value <= 0x00FF_FFFF_FFFF_FFFF {
        8
    } else {
        9
    }
}

/// Convenience: read a varint and interpret it as a signed i64 (two's complement).
pub fn read_varint_i64(buf: &[u8]) -> (i64, usize) {
    let (v, n) = read_varint(buf);
    (v as i64, n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_byte() {
        // Values 0-127 encode in 1 byte.
        let mut buf = [0u8; 9];
        for v in 0..=127u64 {
            let n = write_varint(&mut buf, v);
            assert_eq!(n, 1, "value {v} should encode in 1 byte");
            assert_eq!(buf[0], v as u8);
            let (decoded, dn) = read_varint(&buf);
            assert_eq!(decoded, v);
            assert_eq!(dn, 1);
        }
    }

    #[test]
    fn test_two_bytes() {
        let mut buf = [0u8; 9];
        // 128 = 0x80 -> 2 bytes
        let n = write_varint(&mut buf, 128);
        assert_eq!(n, 2);
        let (v, dn) = read_varint(&buf);
        assert_eq!(v, 128);
        assert_eq!(dn, 2);

        // Max 2-byte value: 0x3FFF = 16383
        let n = write_varint(&mut buf, 0x3FFF);
        assert_eq!(n, 2);
        let (v, dn) = read_varint(&buf);
        assert_eq!(v, 0x3FFF);
        assert_eq!(dn, 2);
    }

    #[test]
    fn test_boundary_values() {
        let mut buf = [0u8; 9];
        // Test boundaries between each varint size.
        let boundaries: &[(u64, usize)] = &[
            (0x7F, 1),             // max 1-byte
            (0x80, 2),             // min 2-byte
            (0x3FFF, 2),           // max 2-byte
            (0x4000, 3),           // min 3-byte
            (0x1F_FFFF, 3),        // max 3-byte
            (0x20_0000, 4),        // min 4-byte
            (0x0FFF_FFFF, 4),      // max 4-byte
            (0x1000_0000, 5),      // min 5-byte
            (0x07_FFFF_FFFF, 5),   // max 5-byte
            (0x08_0000_0000, 6),   // min 6-byte
            (0x03FF_FFFF_FFFF, 6), // max 6-byte
        ];
        for &(val, expected_len) in boundaries {
            let n = write_varint(&mut buf, val);
            assert_eq!(n, expected_len, "varint_len mismatch for {val:#x}");
            let (decoded, dn) = read_varint(&buf);
            assert_eq!(decoded, val, "round-trip failed for {val:#x}");
            assert_eq!(dn, expected_len);
        }
    }

    #[test]
    fn test_nine_byte_varint() {
        let mut buf = [0u8; 9];
        // u64::MAX requires 9 bytes.
        let n = write_varint(&mut buf, u64::MAX);
        assert_eq!(n, 9);
        let (v, dn) = read_varint(&buf);
        assert_eq!(v, u64::MAX);
        assert_eq!(dn, 9);
    }

    #[test]
    fn test_round_trip_various() {
        let mut buf = [0u8; 9];
        let values = [
            0,
            1,
            127,
            128,
            255,
            256,
            16383,
            16384,
            1_000_000,
            u64::MAX / 2,
            u64::MAX,
        ];
        for &val in &values {
            let n = write_varint(&mut buf, val);
            let (decoded, dn) = read_varint(&buf);
            assert_eq!(decoded, val, "round-trip failed for {val}");
            assert_eq!(n, dn);
        }
    }

    #[test]
    fn test_varint_len() {
        assert_eq!(varint_len(0), 1);
        assert_eq!(varint_len(127), 1);
        assert_eq!(varint_len(128), 2);
        assert_eq!(varint_len(16383), 2);
        assert_eq!(varint_len(16384), 3);
        assert_eq!(varint_len(u64::MAX), 9);
    }

    #[test]
    fn test_try_read_varint_empty() {
        assert_eq!(try_read_varint(&[]), None);
    }

    #[test]
    fn test_try_read_varint_truncated() {
        // A byte with continuation bit set but no following byte.
        assert_eq!(try_read_varint(&[0x80]), None);
        // Two continuation bytes but no terminator.
        assert_eq!(try_read_varint(&[0x80, 0x80]), None);
    }

    #[test]
    fn test_try_read_varint_valid() {
        let result = try_read_varint(&[0x05]);
        assert_eq!(result, Some((5, 1)));

        let result = try_read_varint(&[0x81, 0x00]);
        assert_eq!(result, Some((128, 2)));
    }

    #[test]
    fn test_signed_round_trip() {
        let mut buf = [0u8; 9];
        // Negative numbers via two's complement reinterpretation.
        let val: i64 = -1;
        let n = write_varint(&mut buf, val as u64);
        let (decoded, dn) = read_varint_i64(&buf);
        assert_eq!(decoded, val);
        assert_eq!(n, dn);
    }
}
