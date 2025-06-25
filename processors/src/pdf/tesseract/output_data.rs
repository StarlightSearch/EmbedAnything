use input::{Args, Image};
use parse_line_util::{parse_next, FromLine};

use super::*;
use core::fmt;

#[derive(Debug, PartialEq)]
pub struct DataOutput {
    pub output: String,
    pub data: Vec<Data>,
}

impl fmt::Display for DataOutput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.output)
    }
}

#[derive(Debug, PartialEq)]
pub struct Data {
    pub level: i32,
    pub page_num: i32,
    pub block_num: i32,
    pub par_num: i32,
    pub line_num: i32,
    pub word_num: i32,
    pub left: i32,
    pub top: i32,
    pub width: i32,
    pub height: i32,
    pub conf: f32,
    pub text: String,
}

impl fmt::Display for Data {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} {} {} {} {} {} {} {} {} {} {} {}",
            self.level,
            self.page_num,
            self.block_num,
            self.par_num,
            self.line_num,
            self.word_num,
            self.left,
            self.top,
            self.width,
            self.height,
            self.conf,
            self.text,
        )
    }
}

impl FromLine for Data {
    fn from_line(line: &str) -> Option<Self> {
        let mut x = line.split_whitespace();
        Some(Data {
            level: parse_next(&mut x)?,
            page_num: parse_next(&mut x)?,
            block_num: parse_next(&mut x)?,
            par_num: parse_next(&mut x)?,
            line_num: parse_next(&mut x)?,
            word_num: parse_next(&mut x)?,
            left: parse_next(&mut x)?,
            top: parse_next(&mut x)?,
            width: parse_next(&mut x)?,
            height: parse_next(&mut x)?,
            conf: parse_next(&mut x)?,
            text: x.next().unwrap_or("").to_string(),
        })
    }
}

pub fn image_to_data(image: &Image, args: &Args) -> error::TessResult<DataOutput> {
    let mut command = command::create_tesseract_command(image, args)?;
    command.arg("tsv");

    let output = command::run_tesseract_command(&mut command)?;

    let data = string_to_data(&output)?;

    Ok(DataOutput { output, data })
}

fn string_to_data(output: &str) -> error::TessResult<Vec<Data>> {
    output.lines().skip(1).map(Data::parse).collect::<_>()
}

#[cfg(test)]
mod tests {
    use crate::pdf::tesseract::output_data::{string_to_data, Data};

    #[test]
    fn test_string_to_data() {
        let result = string_to_data("level   page_num        block_num       par_num line_num        word_num        left    top     width   height  conf    text
        5       1       1       1       1       1       65      41      46      20      96.063751       The");
        assert_eq!(
            *result.unwrap().first().unwrap(),
            Data {
                level: 5,
                page_num: 1,
                block_num: 1,
                par_num: 1,
                line_num: 1,
                word_num: 1,
                left: 65,
                top: 41,
                width: 46,
                height: 20,
                conf: 96.063_75,
                text: String::from("The"),
            }
        )
    }

    #[test]
    fn test_string_to_data_parse_error() {
        let result = string_to_data("level   page_num        block_num       par_num line_num        word_num        left    top     width   height  conf    text\n\
        Test");
        assert_eq!(
            result,
            Err(crate::pdf::tesseract::error::TessError::ParseError(
                "invalid line 'Test'".into()
            ))
        )
    }
}
