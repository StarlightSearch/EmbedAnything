use parse_line_util::FromLine;

use super::*;
use core::fmt;

#[derive(Debug, PartialEq)]
pub struct ConfigParameterOutput {
    pub output: String,
    pub config_parameters: Vec<ConfigParameter>,
}

impl fmt::Display for ConfigParameterOutput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.output)
    }
}

#[derive(Debug, PartialEq)]
pub struct ConfigParameter {
    pub name: String,
    pub default_value: String,
    pub description: String,
}

impl fmt::Display for ConfigParameter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} {} {}",
            self.name, self.default_value, self.description,
        )
    }
}

impl FromLine for ConfigParameter {
    fn from_line(line: &str) -> Option<Self> {
        let (name, x) = line.split_once("\t")?;
        let (default_value, description) = x.split_once("\t")?;

        Some(ConfigParameter {
            name: name.into(),
            default_value: default_value.into(),
            description: description.into(),
        })
    }
}

pub fn get_tesseract_config_parameters() -> error::TessResult<ConfigParameterOutput> {
    let mut command = command::get_tesseract_command(None);
    command.arg("--print-parameters");

    let output = command::run_tesseract_command(&mut command)?;

    let config_parameters = string_to_config_parameter_output(&output)?;

    Ok(ConfigParameterOutput {
        output,
        config_parameters,
    })
}

fn string_to_config_parameter_output(
    output: &str,
) -> crate::pdf::tesseract::error::TessResult<Vec<ConfigParameter>> {
    output
        .lines()
        .skip(1)
        .map(ConfigParameter::parse)
        .collect::<_>()
}

#[cfg(test)]
mod tests {
    use crate::pdf::tesseract::output_config_parameters::{
        string_to_config_parameter_output, ConfigParameter,
    };

    #[test]
    fn test_string_to_config_parameter_output() {
        let result = string_to_config_parameter_output(
            "Tesseract parameters:\n\
        log_level\t2147483647\tLogging level\n\
        textord_dotmatrix_gap\t3\t pixel gap for broken pixed pitch\n\
        textord_debug_block\t0\tBlock to do debug on\n\
        textord_pitch_range\t2\tMax range test on pitch",
        )
        .unwrap();

        let expected = ConfigParameter {
            name: "log_level".into(),
            default_value: "2147483647".into(),
            description: "Logging level".into(),
        };

        assert_eq!(result.first().unwrap(), &expected);
    }

    #[test]
    fn test_get_tesseract_config_parameters() {
        let result =
            crate::pdf::tesseract::output_config_parameters::get_tesseract_config_parameters()
                .unwrap();
        let x = result
            .config_parameters
            .iter()
            .find(|&x| x.name == "tessedit_char_whitelist")
            .unwrap();

        let expected = ConfigParameter {
            name: "tessedit_char_whitelist".into(),
            default_value: "".into(),
            description: "Whitelist of chars to recognize".into(),
        };

        assert_eq!(*x, expected);
    }

    #[test]
    fn test_string_to_config_parameter_output_parse_error() {
        let result = string_to_config_parameter_output(
            "Tesseract parameters:\n\
        log_level\t2147483647\tLogging level\n\
        Test\n\
        textord_debug_block\t0\tBlock to do debug on\n\
        textord_pitch_range\t2\tMax range test on pitch",
        );
        assert_eq!(
            result,
            Err(crate::pdf::tesseract::error::TessError::ParseError(
                "invalid line 'Test'".into()
            ))
        )
    }
}
