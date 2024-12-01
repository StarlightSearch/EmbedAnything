# Use PDFs that need OCR

Embed Anything can be used to embed scanned documents using OCR. This is useful for tasks such as document search and retrieval. You can set `use_ocr=True` in the `TextEmbedConfig` to enable OCR. But this requires `tesseract` and `poppler` to be installed.

You can install `tesseract` and `poppler` using the following commands:

## Install Tesseract and Poppler

### Windows

For Tesseract, download the installer from [here](https://github.com/UB-Mannheim/tesseract/wiki) and install it.

For Poppler, download the installer from [here](https://github.com/oschwartz10612/poppler-windows?tab=readme-ov-file) and install it.

### MacOS

For Tesseract, you can install it using Homebrew.

``` bash
brew install tesseract
```

For Poppler, you can install it using Homebrew.

``` bash
brew install poppler
```

### Linux

For Tesseract, you can install it using the package manager for your Linux distribution. For example, on Ubuntu, you can install it using:

``` bash
sudo apt install tesseract-ocr
sudo apt install libtesseract-dev

```

For Poppler, you can install it using the package manager for your Linux distribution. For example, on Ubuntu, you can install it using:

``` bash
sudo apt install poppler-utils
```

For more information, refer to the [Tesseract installation guide](https://tesseract-ocr.github.io/tessdoc/Installation.html).

## Example Usage

``` python
--8<-- "examples/text_ocr.py"
```
