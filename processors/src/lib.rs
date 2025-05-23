/// This library contains the traits and structs used to build a processor for arbitrary contents.
pub mod processor;

/// This module contains the file processor for different file types.
pub mod pdf;

/// This module contains the file processor for markdown files.
pub mod markdown_processor;

/// This module contains the file processor for text files.
pub mod txt_processor;

/// This module contains the file processor for HTML files.
pub mod html_processor;

/// This module contains the file processor for DOCX files.
pub mod docx_processor;
