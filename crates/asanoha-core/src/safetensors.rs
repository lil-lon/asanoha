//! Reader for the safetensors format: an 8-byte little-endian header length,
//! a JSON header, then the little-endian tensor data.
//!
//! The header maps each tensor name to `{"dtype", "shape", "data_offsets"}`,
//! with offsets relative to the start of the data, plus an optional
//! `__metadata__` map of strings.

use std::collections::BTreeMap;

use crate::json::{self, Value};
use crate::{Error, Real};

/// Maximum header length in bytes. Parsing allocates a JSON tree several
/// times larger than the header, so this bounds memory use.
pub const MAX_HEADER_LEN: u64 = 16 << 20;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Dtype {
    F32,
    F64,
    I32,
    I64,
}

impl Dtype {
    fn parse(s: &str) -> Result<Self, Error> {
        Ok(match s {
            "F32" => Dtype::F32,
            "F64" => Dtype::F64,
            "I32" => Dtype::I32,
            "I64" => Dtype::I64,
            _ => return Err(Error::UnsupportedDtype(s.to_owned())),
        })
    }

    fn name(self) -> &'static str {
        match self {
            Dtype::F32 => "F32",
            Dtype::F64 => "F64",
            Dtype::I32 => "I32",
            Dtype::I64 => "I64",
        }
    }

    fn size(self) -> usize {
        match self {
            Dtype::F32 | Dtype::I32 => 4,
            Dtype::F64 | Dtype::I64 => 8,
        }
    }
}

#[derive(Debug)]
struct Info {
    dtype: Dtype,
    shape: Vec<usize>,
    start: usize,
    end: usize,
}

/// A parsed safetensors buffer. Tensor data is borrowed, not copied.
#[derive(Debug)]
pub struct SafeTensors<'a> {
    tensors: BTreeMap<String, Info>,
    metadata: BTreeMap<String, String>,
    data: &'a [u8],
}

impl<'a> SafeTensors<'a> {
    /// Parses and validates a whole file. The header must be at most
    /// [`MAX_HEADER_LEN`] bytes, shapes and offsets must be written as
    /// integers, every tensor must have dtype `F32`, `F64`, `I32` or `I64`
    /// and a byte size matching its shape, and the
    /// tensors must cover the data exactly, with no gaps, overlaps or
    /// trailing bytes.
    pub fn parse(bytes: &'a [u8]) -> Result<Self, Error> {
        let err = |msg: &str| Error::Format(msg.to_owned());
        let (len, rest) = bytes
            .split_first_chunk::<8>()
            .ok_or_else(|| err("shorter than the 8-byte header length"))?;
        let len = u64::from_le_bytes(*len);
        if len > MAX_HEADER_LEN {
            return Err(err("header length exceeds MAX_HEADER_LEN"));
        }
        if len > rest.len() as u64 {
            return Err(err("header length exceeds the file size"));
        }
        let (header, data) = rest.split_at(len as usize);
        let header = std::str::from_utf8(header).map_err(|_| err("header is not UTF-8"))?;
        let Value::Object(entries) = json::parse(header)? else {
            return Err(err("header is not a JSON object"));
        };

        let mut tensors = BTreeMap::new();
        let mut metadata = BTreeMap::new();
        for (name, value) in entries {
            if name == "__metadata__" {
                metadata = parse_metadata(&value)?;
            } else {
                let info = parse_info(&name, &value, data.len())?;
                tensors.insert(name.into_owned(), info);
            }
        }

        let mut spans: Vec<_> = tensors.values().map(|t| (t.start, t.end)).collect();
        spans.sort_unstable();
        let mut pos = 0;
        for (start, end) in spans {
            if start != pos {
                return Err(err("tensor data has gaps or overlaps"));
            }
            pos = end;
        }
        if pos != data.len() {
            return Err(err("trailing bytes after the tensor data"));
        }

        Ok(Self {
            tensors,
            metadata,
            data,
        })
    }

    /// Tensor names in sorted order.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }

    /// The tensor called `name`.
    pub fn tensor(&self, name: &str) -> Result<TensorView<'_>, Error> {
        let t = self
            .tensors
            .get(name)
            .ok_or_else(|| Error::MissingTensor(name.to_owned()))?;
        Ok(TensorView {
            dtype: t.dtype,
            shape: &t.shape,
            data: &self.data[t.start..t.end],
        })
    }

    /// The `__metadata__` map, empty if the header has none.
    pub fn metadata(&self) -> &BTreeMap<String, String> {
        &self.metadata
    }
}

fn parse_metadata(value: &Value<'_>) -> Result<BTreeMap<String, String>, Error> {
    value
        .as_object()
        .and_then(|m| {
            m.iter()
                .map(|(k, v)| Some((k.to_string(), v.as_str()?.to_owned())))
                .collect()
        })
        .ok_or_else(|| Error::Format("__metadata__ must map strings to strings".into()))
}

fn parse_info(name: &str, value: &Value<'_>, data_len: usize) -> Result<Info, Error> {
    let err = |msg: &str| Error::Format(format!("tensor {name:?}: {msg}"));
    let entry = value
        .as_object()
        .filter(|e| e.len() == 3)
        .ok_or_else(|| err("expected exactly dtype, shape and data_offsets"))?;
    let dtype = entry
        .get("dtype")
        .and_then(Value::as_str)
        .ok_or_else(|| err("invalid dtype"))?;
    let dtype = Dtype::parse(dtype)?;
    let shape = usizes(entry.get("shape")).ok_or_else(|| err("invalid shape"))?;
    let offsets = usizes(entry.get("data_offsets")).ok_or_else(|| err("invalid data_offsets"))?;
    let [start, end] = offsets[..] else {
        return Err(err("data_offsets must have two entries"));
    };
    if start > end || end > data_len {
        return Err(err("data_offsets out of range"));
    }
    let size = shape
        .iter()
        .try_fold(dtype.size(), |acc, &d| acc.checked_mul(d));
    if size != Some(end - start) {
        return Err(err("byte size does not match shape and dtype"));
    }
    Ok(Info {
        dtype,
        shape,
        start,
        end,
    })
}

/// An array of non-negative integers.
fn usizes(value: Option<&Value<'_>>) -> Option<Vec<usize>> {
    value?
        .as_array()?
        .iter()
        .map(|x| usize::try_from(x.as_i64()?).ok())
        .collect()
}

/// One tensor of a [`SafeTensors`].
#[derive(Debug, Clone, Copy)]
pub struct TensorView<'s> {
    dtype: Dtype,
    shape: &'s [usize],
    data: &'s [u8],
}

impl TensorView<'_> {
    /// The shape; empty for a scalar.
    pub fn shape(&self) -> &[usize] {
        self.shape
    }

    /// Elements of an `F32` or `F64` tensor in row-major order, converted to
    /// `T` (rounding to nearest when `T` is narrower).
    pub fn to_vec<T: Real>(&self) -> Result<Vec<T>, Error> {
        match self.dtype {
            Dtype::F32 => Ok(self
                .data
                .as_chunks()
                .0
                .iter()
                .map(|&b| T::from_f64(f32::from_le_bytes(b) as f64))
                .collect()),
            Dtype::F64 => Ok(self
                .data
                .as_chunks()
                .0
                .iter()
                .map(|&b| T::from_f64(f64::from_le_bytes(b)))
                .collect()),
            d => Err(Error::UnsupportedDtype(format!(
                "{} (expected F32 or F64)",
                d.name()
            ))),
        }
    }

    /// Elements of an `I32` or `I64` tensor in row-major order.
    pub fn to_vec_i64(&self) -> Result<Vec<i64>, Error> {
        match self.dtype {
            Dtype::I32 => Ok(self
                .data
                .as_chunks()
                .0
                .iter()
                .map(|&b| i32::from_le_bytes(b) as i64)
                .collect()),
            Dtype::I64 => Ok(self
                .data
                .as_chunks()
                .0
                .iter()
                .map(|&b| i64::from_le_bytes(b))
                .collect()),
            d => Err(Error::UnsupportedDtype(format!(
                "{} (expected I32 or I64)",
                d.name()
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::XorShift64;

    /// Header length, header and data concatenated.
    fn file(header: &str, data: &[u8]) -> Vec<u8> {
        let mut out = (header.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(data);
        out
    }

    /// A file with the tensors' data laid out in the given order.
    fn write(tensors: &[(&str, &str, &[usize], Vec<u8>)], metadata: Option<&str>) -> Vec<u8> {
        let mut entries = Vec::new();
        let mut data = Vec::new();
        if let Some(m) = metadata {
            entries.push(format!("\"__metadata__\":{m}"));
        }
        for (name, dtype, shape, bytes) in tensors {
            let shape: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
            entries.push(format!(
                "\"{name}\":{{\"dtype\":\"{dtype}\",\"shape\":[{}],\"data_offsets\":[{},{}]}}",
                shape.join(","),
                data.len(),
                data.len() + bytes.len()
            ));
            data.extend_from_slice(bytes);
        }
        file(&format!("{{{}}}", entries.join(",")), &data)
    }

    fn le<const N: usize, X: Copy>(x: &[X], to_bytes: fn(X) -> [u8; N]) -> Vec<u8> {
        x.iter().flat_map(|&v| to_bytes(v)).collect()
    }

    fn reals(rng: &mut XorShift64) -> Vec<f64> {
        let mut x = vec![
            0.0,
            -0.0,
            1.0,
            -2.5,
            f64::MAX,
            f64::MIN_POSITIVE,
            5e-324,
            f32::MAX as f64,
            f32::MIN_POSITIVE as f64,
            1e-45,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ];
        x.extend((0..20).map(|_| (rng.next_f64() - 0.5) * 1e3));
        x
    }

    fn bits<T: Real>(x: &[T]) -> Vec<u64> {
        x.iter().map(|v| v.to_f64().to_bits()).collect()
    }

    #[test]
    fn mixed_tensors_and_metadata() {
        let x64 = reals(&mut XorShift64::new(3));
        let x32: Vec<f32> = reals(&mut XorShift64::new(5))
            .iter()
            .map(|&v| v as f32)
            .collect();
        let i32s = [0, 1, -1, i32::MIN, i32::MAX, 123_456];
        let i64s = [0, -7, i64::MIN, i64::MAX, 1 << 53];
        // Data order differs from name order.
        let bytes = write(
            &[
                ("z.f64", "F64", &[4, 8], le(&x64, f64::to_le_bytes)),
                ("b.i32", "I32", &[2, 3], le(&i32s, i32::to_le_bytes)),
                ("empty", "F32", &[0, 3], vec![]),
                ("a.f32", "F32", &[32], le(&x32, f32::to_le_bytes)),
                ("scalar", "F64", &[], le(&[0.5], f64::to_le_bytes)),
                ("c.i64", "I64", &[5], le(&i64s, i64::to_le_bytes)),
            ],
            Some(r#"{"asanoha.format_version": "1", "note": "é"}"#),
        );
        let st = SafeTensors::parse(&bytes).unwrap();

        let names: Vec<_> = st.names().collect();
        assert_eq!(
            names,
            ["a.f32", "b.i32", "c.i64", "empty", "scalar", "z.f64"]
        );
        assert_eq!(st.metadata()["asanoha.format_version"], "1");
        assert_eq!(st.metadata()["note"], "é");

        let t = st.tensor("z.f64").unwrap();
        assert_eq!(t.shape(), [4, 8]);
        assert_eq!(bits(&t.to_vec::<f64>().unwrap()), bits(&x64));
        let narrowed: Vec<f32> = x64.iter().map(|&v| v as f32).collect();
        assert_eq!(bits(&t.to_vec::<f32>().unwrap()), bits(&narrowed));

        let t = st.tensor("a.f32").unwrap();
        assert_eq!(bits(&t.to_vec::<f32>().unwrap()), bits(&x32));
        assert_eq!(bits(&t.to_vec::<f64>().unwrap()), bits(&x32));

        let expected: Vec<i64> = i32s.iter().map(|&v| v as i64).collect();
        assert_eq!(st.tensor("b.i32").unwrap().to_vec_i64().unwrap(), expected);
        assert_eq!(st.tensor("c.i64").unwrap().to_vec_i64().unwrap(), i64s);

        let t = st.tensor("empty").unwrap();
        assert_eq!(t.shape(), [0, 3]);
        assert!(t.to_vec::<f64>().unwrap().is_empty());
        let t = st.tensor("scalar").unwrap();
        assert!(t.shape().is_empty());
        assert_eq!(t.to_vec::<f64>().unwrap(), [0.5]);
    }

    // Writers pad the header with spaces so the data starts at a multiple of
    // 8 bytes. The bytes are spelled out by hand, without the helpers above.
    #[test]
    fn padded_headers() {
        let bytes = [
            b"\x40\0\0\0\0\0\0\0".as_slice(),
            br#"{"bias":{"dtype":"F64","shape":[2,1],"data_offsets":[0,16]}}    "#,
            &[0, 0, 0, 0, 0, 0, 0xd0, 0x3f, 0, 0, 0, 0, 0, 0, 0x10, 0xc0],
        ]
        .concat();
        let st = SafeTensors::parse(&bytes).unwrap();
        let t = st.tensor("bias").unwrap();
        assert_eq!(t.shape(), [2, 1]);
        assert_eq!(t.to_vec::<f64>().unwrap(), [0.25, -4.0]);

        let bytes = [
            b"\x20\0\0\0\0\0\0\0".as_slice(),
            br#"{"__metadata__":{"k":"v"}}      "#,
        ]
        .concat();
        let st = SafeTensors::parse(&bytes).unwrap();
        assert_eq!(st.names().count(), 0);
        assert_eq!(st.metadata()["k"], "v");

        let st = SafeTensors::parse(b"\x08\0\0\0\0\0\0\0 \r{}\n\t  ").unwrap();
        assert_eq!(st.names().count(), 0);
    }

    #[test]
    fn missing_tensor_and_wrong_conversion() {
        let bytes = write(
            &[
                ("i", "I64", &[1], vec![0; 8]),
                ("f", "F32", &[1], vec![0; 4]),
            ],
            None,
        );
        let st = SafeTensors::parse(&bytes).unwrap();
        assert_eq!(
            st.tensor("x").unwrap_err(),
            Error::MissingTensor("x".into())
        );
        assert_eq!(
            st.tensor("i").unwrap().to_vec::<f64>().unwrap_err(),
            Error::UnsupportedDtype("I64 (expected F32 or F64)".into())
        );
        assert_eq!(
            st.tensor("f").unwrap().to_vec_i64().unwrap_err(),
            Error::UnsupportedDtype("F32 (expected I32 or I64)".into())
        );
    }

    #[test]
    fn rejects_malformed_framing() {
        assert!(matches!(SafeTensors::parse(&[]), Err(Error::Format(_))));
        assert!(matches!(
            SafeTensors::parse(&[2, 0, 0, 0, 0, 0, 0]),
            Err(Error::Format(_))
        ));

        let mut bytes = file("{}", &[]);
        bytes[0] = 3;
        assert!(matches!(SafeTensors::parse(&bytes), Err(Error::Format(_))));

        let mut bytes = file("{}", &[]);
        bytes[..8].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(matches!(SafeTensors::parse(&bytes), Err(Error::Format(_))));

        let mut bytes = file("{\"\u{e9}\":1}", &[]);
        bytes[11] = 0xff;
        assert!(matches!(SafeTensors::parse(&bytes), Err(Error::Format(_))));

        assert!(matches!(
            SafeTensors::parse(&file("{", &[])),
            Err(Error::Json(_))
        ));
        assert!(matches!(
            SafeTensors::parse(&file("[]", &[])),
            Err(Error::Format(_))
        ));
    }

    #[test]
    fn header_length_limit() {
        let max = MAX_HEADER_LEN as usize;
        let header = format!("{{}}{}", " ".repeat(max - 2));
        assert!(SafeTensors::parse(&file(&header, &[])).is_ok());
        let header = format!("{{}}{}", " ".repeat(max - 1));
        assert_eq!(
            SafeTensors::parse(&file(&header, &[])).unwrap_err(),
            Error::Format("header length exceeds MAX_HEADER_LEN".into())
        );
    }

    #[test]
    fn rejects_invalid_headers() {
        let t = |entry: &str| format!(r#"{{"t":{entry}}}"#);
        let ok = r#"{"dtype":"F32","shape":[2],"data_offsets":[0,8]}"#;
        assert!(SafeTensors::parse(&file(&t(ok), &[0; 8])).is_ok());

        for (header, data_len) in [
            (t(r#"{"dtype":"F32","shape":[2]}"#), 8),
            (
                t(r#"{"dtype":"F32","shape":[2],"data_offsets":[0,8],"x":1}"#),
                8,
            ),
            (t(r#"{"dtype":"F32","shape":[2],"offsets":[0,8]}"#), 8),
            (t(r#"{"dtype":1,"shape":[2],"data_offsets":[0,8]}"#), 8),
            (t(r#"{"dtype":"F32","shape":2,"data_offsets":[0,8]}"#), 8),
            (t(r#"{"dtype":"F32","shape":[-2],"data_offsets":[0,8]}"#), 8),
            (
                t(r#"{"dtype":"F32","shape":[2.5],"data_offsets":[0,8]}"#),
                8,
            ),
            // Integers must be written as integer literals, so values that
            // round to valid integers are still rejected.
            (
                t(r#"{"dtype":"F32","shape":[2.0],"data_offsets":[0,8]}"#),
                8,
            ),
            (
                t(r#"{"dtype":"F32","shape":[1.00000000000000001],"data_offsets":[0,4]}"#),
                4,
            ),
            (
                t(r#"{"dtype":"F32","shape":[-1e-400],"data_offsets":[0,0]}"#),
                0,
            ),
            (
                t(r#"{"dtype":"F32","shape":[2],"data_offsets":[0,8e0]}"#),
                8,
            ),
            (t(r#"{"dtype":"F32","shape":[2],"data_offsets":[0]}"#), 8),
            (
                t(r#"{"dtype":"F32","shape":[2],"data_offsets":[0,8,8]}"#),
                8,
            ),
            (t(r#"{"dtype":"F32","shape":[2],"data_offsets":[8,0]}"#), 8),
            (t(r#"{"dtype":"F32","shape":[2],"data_offsets":[0,16]}"#), 8),
            (t(r#"{"dtype":"F32","shape":[3],"data_offsets":[0,8]}"#), 8),
            (t(r#"{"dtype":"F64","shape":[2],"data_offsets":[0,8]}"#), 8),
            (
                t(
                    r#"{"dtype":"F32","shape":[4503599627370495,4503599627370495],"data_offsets":[0,8]}"#,
                ),
                8,
            ),
            (t(ok), 12),
            (
                t(r#"{"dtype":"F32","shape":[2],"data_offsets":[4,12]}"#),
                12,
            ),
            (r#"{"__metadata__":{"a":1}}"#.to_owned(), 0),
            (r#"{"__metadata__":["a"]}"#.to_owned(), 0),
            (
                format!(r#"{{"a":{ok},"b":{{"dtype":"F32","shape":[2],"data_offsets":[4,12]}}}}"#),
                12,
            ),
            (
                format!(r#"{{"a":{ok},"b":{{"dtype":"F32","shape":[1],"data_offsets":[12,16]}}}}"#),
                16,
            ),
        ] {
            let bytes = file(&header, &vec![0; data_len]);
            let result = SafeTensors::parse(&bytes);
            assert!(
                matches!(result, Err(Error::Format(_))),
                "{header} {result:?}"
            );
        }

        let header = t(r#"{"dtype":"F16","shape":[2],"data_offsets":[0,4]}"#);
        assert_eq!(
            SafeTensors::parse(&file(&header, &[0; 4])).unwrap_err(),
            Error::UnsupportedDtype("F16".into())
        );
    }
}
