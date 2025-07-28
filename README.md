# Figma, PNG, JPEG, PDF to Canvas.js Converter

A web-based tool that converts Figma exports, PNG, JPEG, or PDF files into Canvas.js objects.

## Features

- Upload and convert various Figma (or) image formats (PNG, JPEG) and PDF files
- Preview the converted image on a canvas
- Generate Canvas.js code that reproduces the image
- Copy the generated code with a single click
- Responsive design for desktop and mobile devices

## Demo

[Watch the demo video on YouTube](#) (Link to be added after creating the demo)

## How to Use

1. Open the application in a web browser
2. Click on "Choose File" to upload an image (PNG, JPEG) or PDF file
3. The uploaded file will be displayed in the preview area
4. The Canvas.js code will be generated automatically in the code section
5. Click "Copy Code" to copy the code to your clipboard
6. Paste the code into your project to use the Canvas.js object

## Setup Instructions

### Local Setup

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/image-to-canvasjs-converter.git
   ```

2. Navigate to the project directory:
   ```
   cd image-to-canvasjs-converter
   ```

3. Open `index.html` in your web browser

### Using a Local Server

For better performance, especially when working with PDFs, it's recommended to use a local server:

1. If you have Node.js installed, you can use http-server:
   ```
   npm install -g http-server
   http-server
   ```

2. If you have Python installed:
   ```
   # Python 3
   python -m http.server
   
   # Python 2
   python -m SimpleHTTPServer
   ```

3. Open the provided URL in your browser https://image-to-canvas-js-converter.vercel.app/

## Technologies Used

- HTML5 Canvas for rendering images
- JavaScript for file processing and code generation
- PDF.js for PDF file handling
- Canvas Confetti for visual feedback

## Limitations

- For very large images or complex PDFs, processing may take longer
- The generated Canvas.js code uses a data URL approach for simplicity, which may result in larger code size for complex images
- Only the first page of multi-page PDFs is processed

## Future Enhancements

- Support for SVG files
- Option to optimize the generated code for size
- Advanced image processing options (resize, crop, filters)
- Support for multi-page PDF processing
- Direct integration with Figma API

## License

MIT

## Author

Venkat Ashwin Kumar

---

Feel free to contribute to this project by submitting issues or pull requests!
