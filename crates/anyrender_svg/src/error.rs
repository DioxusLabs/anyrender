// Copyright 2023 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

/// Triggered when there is an issue parsing user input.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    Svg(usvg::Error),
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Svg(error) => write!(f, "Error parsing svg: {error}"),
        }
    }
}

impl core::error::Error for Error {}

impl From<usvg::Error> for Error {
    fn from(value: usvg::Error) -> Self {
        Self::Svg(value)
    }
}
