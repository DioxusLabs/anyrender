// Copyright 2018 the Kurbo Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! A fork of Kurbo's SVG path parser which adds support for NaN and Infinity values

use core::error::Error;
use core::fmt::{self, Display, Formatter};

use peniko::kurbo::{Arc, BezPath, Point, SvgArc};

/// Try to parse a bezier path from an SVG path element.
///
/// This is implemented on a best-effort basis, intended for cases where the
/// user controls the source of paths, and is not intended as a replacement
/// for a general, robust SVG parser.
pub(crate) fn parse_svg_path(data: &str) -> Result<BezPath, SvgParseError> {
    let mut lexer = SvgLexer::new(data);
    let mut path = BezPath::new();
    let mut last_cmd = 0;
    let mut last_ctrl = None;
    let mut first_pt = Point::ORIGIN;
    let mut implicit_moveto = None;
    while let Some(c) = lexer.get_cmd(last_cmd) {
        if c != b'm' && c != b'M' {
            if path.elements().is_empty() {
                return Err(SvgParseError::UninitializedPath);
            }

            if let Some(pt) = implicit_moveto.take() {
                path.move_to(pt);
            }
        }
        match c {
            b'm' | b'M' => {
                implicit_moveto = None;
                let pt = lexer.get_maybe_relative(c)?;
                path.move_to(pt);
                lexer.last_pt = pt;
                first_pt = pt;
                last_ctrl = Some(pt);
                last_cmd = c - (b'M' - b'L');
            }
            b'l' | b'L' => {
                let pt = lexer.get_maybe_relative(c)?;
                path.line_to(pt);
                lexer.last_pt = pt;
                last_ctrl = Some(pt);
                last_cmd = c;
            }
            b'h' | b'H' => {
                let mut x = lexer.get_number()?;
                lexer.opt_comma();
                if c == b'h' {
                    x += lexer.last_pt.x;
                }
                let pt = Point::new(x, lexer.last_pt.y);
                path.line_to(pt);
                lexer.last_pt = pt;
                last_ctrl = Some(pt);
                last_cmd = c;
            }
            b'v' | b'V' => {
                let mut y = lexer.get_number()?;
                lexer.opt_comma();
                if c == b'v' {
                    y += lexer.last_pt.y;
                }
                let pt = Point::new(lexer.last_pt.x, y);
                path.line_to(pt);
                lexer.last_pt = pt;
                last_ctrl = Some(pt);
                last_cmd = c;
            }
            b'q' | b'Q' => {
                let p1 = lexer.get_maybe_relative(c)?;
                let p2 = lexer.get_maybe_relative(c)?;
                path.quad_to(p1, p2);
                last_ctrl = Some(p1);
                lexer.last_pt = p2;
                last_cmd = c;
            }
            b't' | b'T' => {
                let p1 = match last_ctrl {
                    Some(ctrl) => (2.0 * lexer.last_pt.to_vec2() - ctrl.to_vec2()).to_point(),
                    None => lexer.last_pt,
                };
                let p2 = lexer.get_maybe_relative(c)?;
                path.quad_to(p1, p2);
                last_ctrl = Some(p1);
                lexer.last_pt = p2;
                last_cmd = c;
            }
            b'c' | b'C' => {
                let p1 = lexer.get_maybe_relative(c)?;
                let p2 = lexer.get_maybe_relative(c)?;
                let p3 = lexer.get_maybe_relative(c)?;
                path.curve_to(p1, p2, p3);
                last_ctrl = Some(p2);
                lexer.last_pt = p3;
                last_cmd = c;
            }
            b's' | b'S' => {
                let p1 = match last_ctrl {
                    Some(ctrl) => (2.0 * lexer.last_pt.to_vec2() - ctrl.to_vec2()).to_point(),
                    None => lexer.last_pt,
                };
                let p2 = lexer.get_maybe_relative(c)?;
                let p3 = lexer.get_maybe_relative(c)?;
                path.curve_to(p1, p2, p3);
                last_ctrl = Some(p2);
                lexer.last_pt = p3;
                last_cmd = c;
            }
            b'a' | b'A' => {
                let radii = lexer.get_number_pair()?;
                let x_rotation = lexer.get_number()?.to_radians();
                lexer.opt_comma();
                let large_arc = lexer.get_flag()?;
                lexer.opt_comma();
                let sweep = lexer.get_flag()?;
                lexer.opt_comma();
                let p = lexer.get_maybe_relative(c)?;
                let svg_arc = SvgArc {
                    from: lexer.last_pt,
                    to: p,
                    radii: radii.to_vec2(),
                    x_rotation,
                    large_arc,
                    sweep,
                };

                match Arc::from_svg_arc(&svg_arc) {
                    Some(arc) => {
                        // TODO: consider making tolerance configurable
                        arc.to_cubic_beziers(0.1, |p1, p2, p3| {
                            path.curve_to(p1, p2, p3);
                        });
                    }
                    None => {
                        path.line_to(p);
                    }
                }

                last_ctrl = Some(p);
                lexer.last_pt = p;
                last_cmd = c;
            }
            b'z' | b'Z' => {
                path.close_path();
                lexer.last_pt = first_pt;
                implicit_moveto = Some(first_pt);
            }
            _ => return Err(SvgParseError::UnknownCommand(c as char)),
        }
    }
    Ok(path)
}

/// An error which can be returned when parsing an SVG.
#[derive(Debug)]
#[non_exhaustive]
pub(crate) enum SvgParseError {
    /// A number was expected.
    Wrong,
    /// The input string ended while still expecting input.
    UnexpectedEof,
    /// Encountered an unknown command letter.
    UnknownCommand(char),
    /// Encountered a command that precedes expected 'moveto' command.
    UninitializedPath,
}

impl Display for SvgParseError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            SvgParseError::Wrong => write!(f, "Unable to parse a number"),
            SvgParseError::UnexpectedEof => write!(f, "Unexpected EOF"),
            SvgParseError::UnknownCommand(letter) => write!(f, "Unknown command, \"{letter}\""),
            SvgParseError::UninitializedPath => {
                write!(f, "Uninitialized path (missing moveto command)")
            }
        }
    }
}

impl Error for SvgParseError {}

struct SvgLexer<'a> {
    data: &'a str,
    ix: usize,
    pub last_pt: Point,
}

impl SvgLexer<'_> {
    fn new(data: &str) -> SvgLexer<'_> {
        SvgLexer {
            data,
            ix: 0,
            last_pt: Point::ORIGIN,
        }
    }

    fn skip_ws(&mut self) {
        while let Some(&c) = self.data.as_bytes().get(self.ix) {
            if !(c == b' ' || c == 9 || c == 10 || c == 12 || c == 13) {
                break;
            }
            self.ix += 1;
        }
    }

    fn get_cmd(&mut self, last_cmd: u8) -> Option<u8> {
        self.skip_ws();
        if let Some(c) = self.get_byte() {
            if c.is_ascii_lowercase() || c.is_ascii_uppercase() {
                return Some(c);
            } else if last_cmd != 0 && (c == b'-' || c == b'.' || c.is_ascii_digit()) {
                // Plausible number start
                self.unget();
                return Some(last_cmd);
            } else {
                self.unget();
            }
        }
        None
    }

    fn get_byte(&mut self) -> Option<u8> {
        self.data.as_bytes().get(self.ix).map(|&c| {
            self.ix += 1;
            c
        })
    }

    fn unget(&mut self) {
        self.ix -= 1;
    }

    fn get_str(&mut self, s: &[u8]) -> Result<(), SvgParseError> {
        for &expected_c in s.iter() {
            let c = self.get_byte().ok_or(SvgParseError::UnexpectedEof)?;
            if c != expected_c {
                return Err(SvgParseError::Wrong);
            }
        }

        Ok(())
    }

    fn get_number(&mut self) -> Result<f64, SvgParseError> {
        self.skip_ws();
        let start = self.ix;
        let mut is_negative = false;
        let mut c = self.get_byte().ok_or(SvgParseError::UnexpectedEof)?;

        // Handle NaN
        if c == b'n' || c == b'N' {
            self.unget();
            self.get_str(b"NaN")?;
            return Ok(f64::NAN);
        }

        // If first byte is + or - then read the next byte
        if c == b'-' || c == b'+' {
            is_negative = c == b'-';
            c = self.get_byte().ok_or(SvgParseError::UnexpectedEof)?;
        }

        // Handle Infinity, +Infinity, and -Infinity
        if c == b'i' || c == b'I' {
            self.get_str(b"Infinity")?;
            if is_negative {
                return Ok(-f64::INFINITY);
            } else {
                return Ok(f64::INFINITY);
            };
        }

        // Reset back by 1 byte after checking for NaN and Infinity
        self.unget();

        let mut digit_count = 0;
        let mut seen_period = false;
        while let Some(c) = self.get_byte() {
            if c.is_ascii_digit() {
                digit_count += 1;
            } else if c == b'.' && !seen_period {
                seen_period = true;
            } else {
                self.unget();
                break;
            }
        }
        if let Some(c) = self.get_byte() {
            if c == b'e' || c == b'E' {
                let mut c = self.get_byte().ok_or(SvgParseError::Wrong)?;
                if c == b'-' || c == b'+' {
                    c = self.get_byte().ok_or(SvgParseError::Wrong)?;
                }
                if !c.is_ascii_digit() {
                    return Err(SvgParseError::Wrong);
                }
                while let Some(c) = self.get_byte() {
                    if !c.is_ascii_digit() {
                        self.unget();
                        break;
                    }
                }
            } else {
                self.unget();
            }
        }
        if digit_count > 0 {
            self.data[start..self.ix]
                .parse()
                .map_err(|_| SvgParseError::Wrong)
        } else {
            Err(SvgParseError::Wrong)
        }
    }

    fn get_flag(&mut self) -> Result<bool, SvgParseError> {
        self.skip_ws();
        match self.get_byte().ok_or(SvgParseError::UnexpectedEof)? {
            b'0' => Ok(false),
            b'1' => Ok(true),
            _ => Err(SvgParseError::Wrong),
        }
    }

    fn get_number_pair(&mut self) -> Result<Point, SvgParseError> {
        let x = self.get_number()?;
        self.opt_comma();
        let y = self.get_number()?;
        self.opt_comma();
        Ok(Point::new(x, y))
    }

    fn get_maybe_relative(&mut self, cmd: u8) -> Result<Point, SvgParseError> {
        let pt = self.get_number_pair()?;
        if cmd.is_ascii_lowercase() {
            Ok(self.last_pt + pt.to_vec2())
        } else {
            Ok(pt)
        }
    }

    fn opt_comma(&mut self) {
        self.skip_ws();
        if let Some(c) = self.get_byte() {
            if c != b',' {
                self.unget();
            }
        }
    }
}
