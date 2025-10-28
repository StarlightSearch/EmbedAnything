use crate::pdf::tesseract::error::TessResult;
use crate::pdf::tesseract::input::{Args, Image};
use crate::pdf::tesseract::parse_line_util::{parse_next, FromLine};
use core::fmt;

#[derive(Debug, PartialEq)]
pub struct BoxOutput {
    pub output: String,
    pub boxes: Vec<Box>,
}

impl fmt::Display for BoxOutput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.output)
    }
}

#[derive(Debug, PartialEq)]
pub struct Box {
    pub symbol: String,
    pub left: i32,
    pub bottom: i32,
    pub right: i32,
    pub top: i32,
    pub page: i32,
}

impl fmt::Display for Box {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} {} {} {} {} {}",
            self.symbol, self.left, self.bottom, self.right, self.top, self.page
        )
    }
}

impl FromLine for Box {
    fn from_line(line: &str) -> Option<Self> {
        let mut x = line.split_whitespace();

        Some(Box {
            symbol: x.next()?.to_string(),
            left: parse_next(&mut x)?,
            bottom: parse_next(&mut x)?,
            right: parse_next(&mut x)?,
            top: parse_next(&mut x)?,
            page: parse_next(&mut x)?,
        })
    }
}

pub fn image_to_boxes(image: &Image, args: &Args) -> TessResult<BoxOutput> {
    let mut command = crate::pdf::tesseract::command::create_tesseract_command(image, args)?;
    command.arg("makebox");

    let output = crate::pdf::tesseract::command::run_tesseract_command(&mut command)?;
    let boxes = string_to_boxes(&output)?;
    Ok(BoxOutput { output, boxes })
}

fn string_to_boxes(output: &str) -> TessResult<Vec<Box>> {
    output.lines().map(Box::parse).collect::<_>()
}

#[cfg(test)]
mod tests {
    use crate::pdf::tesseract::{
        error::TessError,
        output_boxes::{string_to_boxes, Box},
    };

    #[test]
    fn test_string_to_boxes() {
        let result = string_to_boxes("L 18 26 36 59 0");
        assert_eq!(
            *result.unwrap().first().unwrap(),
            Box {
                symbol: String::from("L"),
                left: 18,
                bottom: 26,
                right: 36,
                top: 59,
                page: 0
            }
        )
    }

    #[test]
    fn test_string_to_boxes_parse_error() {
        let result = string_to_boxes("L 18 X 36 59 0");
        assert_eq!(
            result,
            Err(TessError::ParseError(
                "invalid line 'L 18 X 36 59 0'".into()
            ))
        )
    }
}
