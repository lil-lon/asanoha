use core::fmt;
use core::ops::Range;
use core::str::FromStr;

use crate::Error;

/// Parity of an irrep of O(3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i8)]
pub enum Parity {
    /// `e`, parity `+1`.
    Even = 1,
    /// `o`, parity `-1`.
    Odd = -1,
}

/// Irreducible representation `l p` of O(3), e.g. `1o`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Irrep {
    /// Degree.
    pub l: u8,
    /// Parity.
    pub p: Parity,
}

impl Irrep {
    /// Creates `l p`.
    pub fn new(l: u8, p: Parity) -> Self {
        Self { l, p }
    }

    /// `2l + 1`.
    pub fn dim(self) -> usize {
        2 * self.l as usize + 1
    }
}

impl fmt::Display for Irrep {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let p = match self.p {
            Parity::Even => 'e',
            Parity::Odd => 'o',
        };
        write!(f, "{}{p}", self.l)
    }
}

impl FromStr for Irrep {
    type Err = Error;

    /// Parses `"1o"`, `"2e"`, or `"3y"` where `y` means parity `(-1)^l`, as
    /// e3nn-jax's `Irrep(str)` does. `l` must be ASCII decimal digits; the
    /// extra forms Python's `int()` accepts (`"+1e"`, `"1 e"`, `"1_0e"`) are
    /// rejected.
    fn from_str(s: &str) -> Result<Self, Error> {
        let s = s.trim();
        let err = || Error::Irreps(format!("invalid irrep {s:?}"));
        let (l, p) = s
            .split_at_checked(s.len().wrapping_sub(1))
            .ok_or_else(err)?;
        if l.is_empty() || !l.bytes().all(|b| b.is_ascii_digit()) {
            return Err(err());
        }
        let l: u8 = l.parse().map_err(|_| err())?;
        let p = match p {
            "e" => Parity::Even,
            "o" => Parity::Odd,
            "y" if l.is_multiple_of(2) => Parity::Even,
            "y" => Parity::Odd,
            _ => return Err(err()),
        };
        Ok(Self { l, p })
    }
}

/// One block of an [`Irreps`]: `mul` copies (channels) of `ir`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MulIrrep {
    /// Number of channels.
    pub mul: u32,
    /// The irrep.
    pub ir: Irrep,
}

impl MulIrrep {
    /// `mul · (2l + 1)`.
    pub fn dim(self) -> usize {
        self.mul as usize * self.ir.dim()
    }
}

impl fmt::Display for MulIrrep {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}x{}", self.mul, self.ir)
    }
}

/// Ordered list of `(mul, irrep)` blocks, written in e3nn syntax such as
/// `128x0e+64x1o+32x2e+32x3o`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct Irreps(Vec<MulIrrep>);

impl Irreps {
    /// Creates irreps from blocks, kept in the given order.
    pub fn new(blocks: Vec<MulIrrep>) -> Self {
        Self(blocks)
    }

    /// `Σ mul · (2l + 1)`.
    pub fn dim(&self) -> usize {
        self.0.iter().map(|b| b.dim()).sum()
    }

    /// `Σ mul`, the number of channels.
    pub fn num_irreps(&self) -> usize {
        self.0.iter().map(|b| b.mul as usize).sum()
    }

    /// The blocks in order.
    pub fn blocks(&self) -> &[MulIrrep] {
        &self.0
    }

    /// Range of each block in a feature vector of length [`Irreps::dim`].
    /// The ranges are the same in both layouts; only the order inside a block
    /// differs.
    pub fn slices(&self) -> Vec<Range<usize>> {
        let mut start = 0;
        self.0
            .iter()
            .map(|b| {
                let r = start..start + b.dim();
                start = r.end;
                r
            })
            .collect()
    }

    /// Offset of each block in channel space (length [`Irreps::num_irreps`]),
    /// as used for per-channel weights.
    pub fn channel_offsets(&self) -> Vec<usize> {
        let mut start = 0;
        self.0
            .iter()
            .map(|b| {
                let o = start;
                start += b.mul as usize;
                o
            })
            .collect()
    }
}

impl fmt::Display for Irreps {
    /// Same as e3nn-jax's `repr`, except that empty irreps print as `""`
    /// (e3nn-jax prints `"Irreps()"`, which it cannot parse back).
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (k, b) in self.0.iter().enumerate() {
            if k > 0 {
                f.write_str("+")?;
            }
            write!(f, "{b}")?;
        }
        Ok(())
    }
}

impl FromStr for Irreps {
    type Err = Error;

    /// Parses `"128x0e + 64x1o"` as e3nn-jax's `Irreps(str)` does: a block
    /// without `x` has `mul = 1`, `mul = 0` is kept, and a blank string gives
    /// empty irreps.
    fn from_str(s: &str) -> Result<Self, Error> {
        if s.trim().is_empty() {
            return Ok(Self::default());
        }
        let blocks = s
            .split('+')
            .map(|term| {
                let term = term.trim();
                let (mul, ir) = match term.split_once('x') {
                    Some((mul, ir)) => {
                        let mul = mul.trim();
                        if mul.is_empty() || !mul.bytes().all(|b| b.is_ascii_digit()) {
                            return Err(Error::Irreps(format!("invalid multiplicity in {term:?}")));
                        }
                        let mul = mul.parse().map_err(|_| {
                            Error::Irreps(format!("invalid multiplicity in {term:?}"))
                        })?;
                        (mul, ir)
                    }
                    None => (1, term),
                };
                Ok(MulIrrep {
                    mul,
                    ir: ir.parse()?,
                })
            })
            .collect::<Result<_, _>>()?;
        Ok(Self(blocks))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ir(l: u8, p: Parity) -> Irrep {
        Irrep::new(l, p)
    }

    #[test]
    fn irrep_parse_and_display() {
        assert_eq!("0e".parse::<Irrep>().unwrap(), ir(0, Parity::Even));
        assert_eq!("3o".parse::<Irrep>().unwrap(), ir(3, Parity::Odd));
        assert_eq!(" 12e ".parse::<Irrep>().unwrap(), ir(12, Parity::Even));
        assert_eq!(ir(2, Parity::Odd).to_string(), "2o");
        assert_eq!(ir(3, Parity::Odd).dim(), 7);
        assert_eq!("1y".parse::<Irrep>().unwrap(), ir(1, Parity::Odd));
        assert_eq!("2y".parse::<Irrep>().unwrap(), ir(2, Parity::Even));
    }

    #[test]
    fn irrep_rejects_what_e3nn_rejects() {
        for bad in ["", "e", "1", "-1e", "1.0e", "1x", "é"] {
            assert!(bad.parse::<Irrep>().is_err(), "{bad:?} should be rejected");
        }
    }

    #[test]
    fn irrep_rejects_python_int_quirks() {
        for bad in ["+1e", "1 e", "1_0e", "256e"] {
            assert!(bad.parse::<Irrep>().is_err(), "{bad:?} should be rejected");
        }
    }

    // Examples from the e3nn-jax `Irreps` docstrings (`_src/irreps.py`).
    #[test]
    fn e3nn_docstring_examples() {
        let x: Irreps = "100x0e + 50x1e".parse().unwrap();
        assert_eq!(x.to_string(), "100x0e+50x1e");
        assert_eq!(x.dim(), 250);

        let x: Irreps = "100x0e + 50x1e + 0x2e".parse().unwrap();
        assert_eq!(x.to_string(), "100x0e+50x1e+0x2e");

        let x: Irreps = "2x0e + 1e".parse().unwrap();
        assert_eq!(x.slices(), vec![0..2, 2..5]);

        let x: Irreps = "3x0e + 2x1e".parse().unwrap();
        assert_eq!(x.dim(), 9);
        assert_eq!(x.num_irreps(), 5);

        let empty: Irreps = "".parse().unwrap();
        assert_eq!(empty, Irreps::default());
    }

    #[test]
    fn nequix_hidden_irreps() {
        let h: Irreps = "128x0e + 64x1o + 32x2e + 32x3o".parse().unwrap();
        assert_eq!(h.to_string(), "128x0e+64x1o+32x2e+32x3o");
        assert_eq!(h.dim(), 704);
        assert_eq!(h.num_irreps(), 256);
        assert_eq!(h.slices(), vec![0..128, 128..320, 320..480, 480..704]);
        assert_eq!(h.channel_offsets(), vec![0, 128, 192, 224]);
        assert_eq!(h, h.to_string().parse().unwrap());
    }

    #[test]
    fn gated_irreps_dims() {
        let hg: Irreps = "256x0e+64x1o+32x2e+32x3o".parse().unwrap();
        assert_eq!(hg.dim(), 832);
        assert_eq!(hg.num_irreps(), 384);
    }

    #[test]
    fn implicit_multiplicity_and_empty() {
        let s: Irreps = "0e+1o+2e+3o".parse().unwrap();
        assert_eq!(s.to_string(), "1x0e+1x1o+1x2e+1x3o");
        assert_eq!(s.dim(), 16);
        assert_eq!(s.num_irreps(), 4);

        let empty: Irreps = "  ".parse().unwrap();
        assert_eq!(empty.dim(), 0);
        assert!(empty.blocks().is_empty());
        assert!(empty.slices().is_empty());
        assert_eq!(empty.to_string(), "");
    }

    #[test]
    fn rejects_malformed() {
        for bad in [
            "128x",
            "x0e",
            "128x0e+",
            "+0e",
            "128x0e++1o",
            "-1x0e",
            "1.5x0e",
            "2x2x0e",
        ] {
            assert!(bad.parse::<Irreps>().is_err(), "{bad:?} should be rejected");
        }
    }
}
