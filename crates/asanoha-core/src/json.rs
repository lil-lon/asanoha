//! Strict JSON reader for safetensors headers and model descriptors.
//!
//! Accepts exactly RFC 8259 JSON. Duplicate object keys, trailing characters,
//! unpaired surrogates, numbers outside the `f64` range and nesting deeper than
//! [`MAX_DEPTH`] are rejected.

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::collections::btree_map::Entry;

use crate::Error;

/// Maximum number of nested arrays and objects. The parser is recursive, so
/// this bounds its stack use.
pub const MAX_DEPTH: usize = 128;

/// A parsed JSON value. Strings borrow from the input unless they contain
/// escapes.
#[derive(Debug, Clone, PartialEq)]
pub enum Value<'a> {
    /// `null`
    Null,
    /// `true` or `false`
    Bool(bool),
    /// A number written without fraction or exponent that fits in `i64`.
    Integer(i64),
    /// Any other number, rounded to the nearest `f64`.
    Number(f64),
    /// A string with escapes resolved.
    String(Cow<'a, str>),
    /// An array.
    Array(Vec<Value<'a>>),
    /// An object; keys are unique.
    Object(BTreeMap<Cow<'a, str>, Value<'a>>),
}

impl<'a> Value<'a> {
    /// Member `key` of an object.
    pub fn get(&self, key: &str) -> Option<&Value<'a>> {
        self.as_object()?.get(key)
    }

    /// The string, if this is a string.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }

    /// The number, rounded to the nearest `f64`, if this is a number.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::Integer(i) => Some(*i as f64),
            Value::Number(x) => Some(*x),
            _ => None,
        }
    }

    /// The integer, if this is a [`Value::Integer`].
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Value::Integer(i) => Some(*i),
            _ => None,
        }
    }

    /// The elements, if this is an array.
    pub fn as_array(&self) -> Option<&[Value<'a>]> {
        match self {
            Value::Array(v) => Some(v),
            _ => None,
        }
    }

    /// The members, if this is an object.
    pub fn as_object(&self) -> Option<&BTreeMap<Cow<'a, str>, Value<'a>>> {
        match self {
            Value::Object(m) => Some(m),
            _ => None,
        }
    }
}

/// Parses one JSON document, optionally surrounded by whitespace.
pub fn parse(s: &str) -> Result<Value<'_>, Error> {
    let mut p = Parser {
        s,
        pos: 0,
        depth: 0,
    };
    let v = p.value()?;
    p.skip_ws();
    if p.pos != s.len() {
        return Err(p.err("trailing characters"));
    }
    Ok(v)
}

struct Parser<'a> {
    s: &'a str,
    pos: usize,
    depth: usize,
}

impl<'a> Parser<'a> {
    fn err(&self, msg: &str) -> Error {
        Error::Json(format!("{msg} at byte {}", self.pos))
    }

    fn peek(&self) -> Option<u8> {
        self.s.as_bytes().get(self.pos).copied()
    }

    fn skip_ws(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.pos += 1;
        }
    }

    fn expect(&mut self, b: u8) -> Result<(), Error> {
        if self.peek() != Some(b) {
            return Err(self.err(&format!("expected {:?}", b as char)));
        }
        self.pos += 1;
        Ok(())
    }

    fn value(&mut self) -> Result<Value<'a>, Error> {
        self.skip_ws();
        match self.peek() {
            Some(c @ (b'{' | b'[')) => {
                if self.depth == MAX_DEPTH {
                    return Err(self.err("nesting too deep"));
                }
                self.depth += 1;
                let v = if c == b'{' {
                    self.object()
                } else {
                    self.array()
                };
                self.depth -= 1;
                v
            }
            Some(b'"') => Ok(Value::String(self.string()?)),
            Some(b't') => self.literal("true", Value::Bool(true)),
            Some(b'f') => self.literal("false", Value::Bool(false)),
            Some(b'n') => self.literal("null", Value::Null),
            Some(b'-' | b'0'..=b'9') => self.number(),
            Some(_) => Err(self.err("unexpected character")),
            None => Err(self.err("unexpected end of input")),
        }
    }

    fn literal(&mut self, word: &str, v: Value<'a>) -> Result<Value<'a>, Error> {
        if !self.s[self.pos..].starts_with(word) {
            return Err(self.err("invalid literal"));
        }
        self.pos += word.len();
        Ok(v)
    }

    fn object(&mut self) -> Result<Value<'a>, Error> {
        self.expect(b'{')?;
        let mut map = BTreeMap::new();
        self.skip_ws();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(Value::Object(map));
        }
        loop {
            self.skip_ws();
            let at = self.pos;
            let key = self.string()?;
            self.skip_ws();
            self.expect(b':')?;
            let v = self.value()?;
            match map.entry(key) {
                Entry::Occupied(e) => {
                    return Err(Error::Json(format!(
                        "duplicate key {:?} at byte {at}",
                        e.key()
                    )));
                }
                Entry::Vacant(e) => {
                    e.insert(v);
                }
            }
            self.skip_ws();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b'}') => {
                    self.pos += 1;
                    return Ok(Value::Object(map));
                }
                _ => return Err(self.err("expected ',' or '}'")),
            }
        }
    }

    fn array(&mut self) -> Result<Value<'a>, Error> {
        self.expect(b'[')?;
        let mut items = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(Value::Array(items));
        }
        loop {
            items.push(self.value()?);
            self.skip_ws();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b']') => {
                    self.pos += 1;
                    return Ok(Value::Array(items));
                }
                _ => return Err(self.err("expected ',' or ']'")),
            }
        }
    }

    fn string(&mut self) -> Result<Cow<'a, str>, Error> {
        self.expect(b'"')?;
        // Text since the last escape; slices start and end at ASCII bytes, so
        // they are always on character boundaries.
        let mut chunk = self.pos;
        let mut owned: Option<String> = None;
        loop {
            match self.peek() {
                Some(b'"') => {
                    let tail = &self.s[chunk..self.pos];
                    self.pos += 1;
                    return Ok(match owned {
                        Some(mut out) => {
                            out.push_str(tail);
                            Cow::Owned(out)
                        }
                        None => Cow::Borrowed(tail),
                    });
                }
                Some(b'\\') => {
                    let mut out = owned.take().unwrap_or_default();
                    out.push_str(&self.s[chunk..self.pos]);
                    self.pos += 1;
                    out.push(self.escape()?);
                    owned = Some(out);
                    chunk = self.pos;
                }
                Some(0..=0x1f) => return Err(self.err("control character in string")),
                Some(_) => self.pos += 1,
                None => return Err(self.err("unterminated string")),
            }
        }
    }

    /// Decodes the escape after a backslash.
    fn escape(&mut self) -> Result<char, Error> {
        let b = self.peek().ok_or_else(|| self.err("unterminated string"))?;
        self.pos += 1;
        Ok(match b {
            b'"' => '"',
            b'\\' => '\\',
            b'/' => '/',
            b'b' => '\u{8}',
            b'f' => '\u{c}',
            b'n' => '\n',
            b'r' => '\r',
            b't' => '\t',
            b'u' => {
                let hi = self.hex4()?;
                let code = if (0xd800..0xdc00).contains(&hi) {
                    if !self.s[self.pos..].starts_with("\\u") {
                        return Err(self.err("unpaired surrogate"));
                    }
                    self.pos += 2;
                    let lo = self.hex4()?;
                    if !(0xdc00..0xe000).contains(&lo) {
                        return Err(self.err("unpaired surrogate"));
                    }
                    0x10000 + ((hi - 0xd800) << 10) + (lo - 0xdc00)
                } else {
                    hi
                };
                // Only a lone low surrogate is left that is not a `char`.
                char::from_u32(code).ok_or_else(|| self.err("unpaired surrogate"))?
            }
            _ => return Err(self.err("invalid escape")),
        })
    }

    fn hex4(&mut self) -> Result<u32, Error> {
        let code = self
            .s
            .get(self.pos..self.pos + 4)
            .and_then(|h| {
                h.chars()
                    .try_fold(0, |acc, c| Some(acc * 16 + c.to_digit(16)?))
            })
            .ok_or_else(|| self.err("invalid \\u escape"))?;
        self.pos += 4;
        Ok(code)
    }

    fn number(&mut self) -> Result<Value<'a>, Error> {
        let start = self.pos;
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        match self.peek() {
            Some(b'0') => self.pos += 1,
            Some(b'1'..=b'9') => {
                self.digits();
            }
            _ => return Err(self.err("invalid number")),
        }
        let mut integer = true;
        if self.peek() == Some(b'.') {
            integer = false;
            self.pos += 1;
            if self.digits() == 0 {
                return Err(self.err("invalid number"));
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            integer = false;
            self.pos += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.pos += 1;
            }
            if self.digits() == 0 {
                return Err(self.err("invalid number"));
            }
        }
        let text = &self.s[start..self.pos];
        if integer && let Ok(i) = text.parse() {
            return Ok(Value::Integer(i));
        }
        // The JSON number grammar is a subset of what `f64::from_str` accepts.
        let x: f64 = text.parse().map_err(|_| self.err("invalid number"))?;
        if !x.is_finite() {
            return Err(Error::Json(format!("number out of range at byte {start}")));
        }
        Ok(Value::Number(x))
    }

    /// Skips ASCII digits and returns how many there were.
    fn digits(&mut self) -> usize {
        let start = self.pos;
        while matches!(self.peek(), Some(b'0'..=b'9')) {
            self.pos += 1;
        }
        self.pos - start
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::XorShift64;
    use std::fmt::Write;

    fn num(x: f64) -> Value<'static> {
        Value::Number(x)
    }

    fn str_(s: &str) -> Value<'static> {
        Value::String(Cow::Owned(s.to_owned()))
    }

    #[test]
    fn literals_and_whitespace() {
        assert_eq!(parse("null").unwrap(), Value::Null);
        assert_eq!(parse(" \t\r\ntrue\n").unwrap(), Value::Bool(true));
        assert_eq!(parse("false").unwrap(), Value::Bool(false));
        assert_eq!(parse("[ ]").unwrap(), Value::Array(vec![]));
        assert_eq!(parse("{ }").unwrap(), Value::Object(BTreeMap::new()));
    }

    #[test]
    fn numbers() {
        assert_eq!(parse("12").unwrap(), Value::Integer(12));
        assert_eq!(parse("12.0").unwrap(), num(12.0));
        assert_eq!(parse("1e1").unwrap(), num(10.0));
        for (text, x) in [
            ("0", 0.0),
            // An integer literal, so the sign of zero is not kept.
            ("-0", 0.0),
            ("-0.0", -0.0),
            ("12", 12.0),
            ("9223372036854775807", 9_223_372_036_854_775_807.0),
            ("9223372036854775808", 9_223_372_036_854_775_808.0),
            ("-3.25", -3.25),
            ("1e3", 1000.0),
            ("1E-2", 0.01),
            ("2.5e+1", 25.0),
            ("0.1", 0.1),
            ("1e-400", 0.0),
            ("1.7976931348623157e308", f64::MAX),
        ] {
            let v = parse(text).unwrap().as_f64().unwrap();
            assert_eq!(v.to_bits(), x.to_bits(), "{text}");
        }
    }

    #[test]
    fn as_i64_only_for_integer_literals() {
        let i = |s| parse(s).unwrap().as_i64();
        assert_eq!(i("0"), Some(0));
        assert_eq!(i("-17"), Some(-17));
        assert_eq!(i("9007199254740993"), Some(9_007_199_254_740_993));
        assert_eq!(i("9223372036854775807"), Some(i64::MAX));
        assert_eq!(i("-9223372036854775808"), Some(i64::MIN));
        assert_eq!(i("9223372036854775808"), None);
        assert_eq!(i("2.0"), None);
        assert_eq!(i("1e2"), None);
        assert_eq!(i("1.00000000000000001"), None);
        assert_eq!(i("-1e-400"), None);
        assert_eq!(i("\"1\""), None);
    }

    #[test]
    fn strings_and_escapes() {
        let v = parse(r#""plain é €""#).unwrap();
        assert!(matches!(v, Value::String(Cow::Borrowed("plain é €"))));

        let v = parse(r#""a\"b\\c\/d\b\f\n\r\te""#).unwrap();
        assert_eq!(v, str_("a\"b\\c/d\u{8}\u{c}\n\r\te"));
        assert!(matches!(v, Value::String(Cow::Owned(_))));

        assert_eq!(parse(r#""é€""#).unwrap(), str_("é€"));
        assert_eq!(parse(r#""x😀y""#).unwrap(), str_("x😀y"));
        assert_eq!(parse(r#""\u0000""#).unwrap(), str_("\0"));
        assert_eq!(parse(r#""""#).unwrap(), str_(""));
    }

    #[test]
    fn nested_document_and_accessors() {
        let text = r#"{
            "__metadata__": {"asanoha.format_version": "1"},
            "w": {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, 24]},
            "flags": [true, null, -1.5e-3]
        }"#;
        let v = parse(text).unwrap();
        assert_eq!(v.as_object().unwrap().len(), 3);
        assert_eq!(
            v.get("__metadata__")
                .and_then(|m| m.get("asanoha.format_version"))
                .and_then(Value::as_str),
            Some("1")
        );
        let w = v.get("w").unwrap();
        assert_eq!(w.get("dtype").and_then(Value::as_str), Some("F32"));
        let shape: Vec<i64> = w
            .get("shape")
            .and_then(Value::as_array)
            .unwrap()
            .iter()
            .map(|x| x.as_i64().unwrap())
            .collect();
        assert_eq!(shape, vec![2, 3]);
        assert_eq!(
            v.get("flags").unwrap().as_array().unwrap(),
            &[Value::Bool(true), Value::Null, num(-1.5e-3)]
        );
        assert_eq!(v.get("missing"), None);
        assert_eq!(w.get("dtype").unwrap().get("x"), None);
    }

    #[test]
    fn rejects_invalid() {
        for bad in [
            "",
            "  ",
            "nul",
            "tru",
            "nullx",
            "01",
            "-01",
            "1.",
            ".5",
            "-",
            "+1",
            "1e",
            "1e+",
            "0x10",
            "NaN",
            "Infinity",
            "1e400",
            "-1e400",
            "[1,]",
            "[1 2]",
            "[",
            "{",
            r#"{"a":1,}"#,
            "{a:1}",
            r#"{"a" 1}"#,
            r#"{"a":1 "b":2}"#,
            r#"{"a":1,"a":2}"#,
            r#"{"a":{},"b":1,"a":[]}"#,
            "{1:2}",
            r#""abc"#,
            r#""\x""#,
            r#""\u12""#,
            r#""\u12G4""#,
            r#""\u+123""#,
            r#""\ud800""#,
            r#""\ud800x""#,
            r#""\udc00""#,
            r#""\ud800A""#,
            r#""\ud800\ud800""#,
            "\"a\tb\"",
            "\"a\nb\"",
            "\"\u{1f}\"",
            "'a'",
            "[1] x",
            "{} {}",
            "\u{feff}1",
        ] {
            assert!(parse(bad).is_err(), "{bad:?} should be rejected");
        }
    }

    #[test]
    fn error_names_duplicate_key() {
        let err = parse(r#"{"k": 1, "k": 2}"#).unwrap_err();
        assert_eq!(err, Error::Json(r#"duplicate key "k" at byte 9"#.into()));
    }

    #[test]
    fn nesting_limit() {
        let nested = |n| "[".repeat(n) + &"]".repeat(n);
        assert!(parse(&nested(MAX_DEPTH)).is_ok());
        let err = parse(&nested(MAX_DEPTH + 1)).unwrap_err();
        assert_eq!(
            err,
            Error::Json(format!("nesting too deep at byte {MAX_DEPTH}"))
        );
        assert!(parse(&"[{\"a\":".repeat(10_000)).is_err());
    }

    #[test]
    fn random_documents_round_trip() {
        let mut rng = XorShift64::new(0x5eed);
        for _ in 0..2000 {
            let mut text = String::new();
            let expected = random_value(&mut rng, 4, &mut text);
            assert_eq!(parse(&text).unwrap(), expected, "{text}");
        }
    }

    fn coin(rng: &mut XorShift64) -> bool {
        rng.next_u64().is_multiple_of(2)
    }

    fn below(rng: &mut XorShift64, n: u64) -> u64 {
        rng.next_u64() % n
    }

    fn random_ws(rng: &mut XorShift64, text: &mut String) {
        for _ in 0..below(rng, 3) {
            text.push([' ', '\t', '\n', '\r'][below(rng, 4) as usize]);
        }
    }

    /// Appends the JSON text of a random value with random formatting and
    /// returns the value.
    fn random_value(rng: &mut XorShift64, depth: u32, text: &mut String) -> Value<'static> {
        random_ws(rng, text);
        let v = match below(rng, if depth == 0 { 4 } else { 6 }) {
            0 => {
                text.push_str("null");
                Value::Null
            }
            1 => {
                let b = coin(rng);
                text.push_str(if b { "true" } else { "false" });
                Value::Bool(b)
            }
            2 => random_number(rng, text),
            3 => Value::String(Cow::Owned(random_string(rng, text))),
            4 => {
                text.push('[');
                let mut items = Vec::new();
                for k in 0..below(rng, 4) {
                    if k > 0 {
                        text.push(',');
                    }
                    items.push(random_value(rng, depth - 1, text));
                }
                random_ws(rng, text);
                text.push(']');
                Value::Array(items)
            }
            _ => {
                text.push('{');
                let mut map = BTreeMap::new();
                for _ in 0..below(rng, 4) {
                    let mut member = String::new();
                    random_ws(rng, &mut member);
                    let key = random_string(rng, &mut member);
                    random_ws(rng, &mut member);
                    member.push(':');
                    let v = random_value(rng, depth - 1, &mut member);
                    if map.contains_key(key.as_str()) {
                        continue;
                    }
                    if !map.is_empty() {
                        text.push(',');
                    }
                    text.push_str(&member);
                    map.insert(Cow::Owned(key), v);
                }
                random_ws(rng, text);
                text.push('}');
                Value::Object(map)
            }
        };
        random_ws(rng, text);
        v
    }

    fn random_number(rng: &mut XorShift64, text: &mut String) -> Value<'static> {
        if coin(rng) {
            let i = rng.next_u64() as i64;
            write!(text, "{i}").unwrap();
            return Value::Integer(i);
        }
        let x = (rng.next_f64() - 0.5) * 10f64.powi(below(rng, 601) as i32 - 300);
        let start = text.len();
        // Both formats print the shortest digits that round-trip.
        if coin(rng) {
            write!(text, "{x}").unwrap();
        } else {
            write!(text, "{x:e}").unwrap();
        }
        // `{x}` prints integral values without a fraction, so they read back
        // as integers when they fit in `i64`.
        match text[start..].parse() {
            Ok(i) => Value::Integer(i),
            Err(_) => Value::Number(x),
        }
    }

    fn random_string(rng: &mut XorShift64, text: &mut String) -> String {
        const POOL: [char; 15] = [
            'a', 'Z', '0', ' ', '"', '\\', '/', '\n', '\t', '\u{1}', '\u{7f}', 'é', '€', '😀',
            '\u{ffff}',
        ];
        let s: String = (0..below(rng, 6))
            .map(|_| POOL[below(rng, POOL.len() as u64) as usize])
            .collect();
        text.push('"');
        for c in s.chars() {
            if !(c == '"' || c == '\\' || c < ' ' || below(rng, 4) == 0) {
                text.push(c);
                continue;
            }
            let short = match c {
                '"' => Some("\\\""),
                '\\' => Some("\\\\"),
                '/' => Some("\\/"),
                '\n' => Some("\\n"),
                '\t' => Some("\\t"),
                _ => None,
            };
            match short {
                Some(e) if coin(rng) => text.push_str(e),
                _ => {
                    for unit in c.encode_utf16(&mut [0; 2]) {
                        if coin(rng) {
                            write!(text, "\\u{unit:04x}").unwrap();
                        } else {
                            write!(text, "\\u{unit:04X}").unwrap();
                        }
                    }
                }
            }
        }
        text.push('"');
        s
    }
}
