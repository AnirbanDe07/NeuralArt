# STYLEFORGE AI

A Flask-based AI style transfer app that applies the artistic style of one image to the content of another using an AdaIN-inspired neural network pipeline.

## Features

- Upload a content image and a style image
- Adjust transfer strength with an alpha slider
- Generate a stylized output image in the browser
- Built-in example gallery for quick testing
- Uses a pretrained VGG encoder and decoder model

## Project Structure

- `app.py` - Flask web application
- `train.py` - training script for the model pipeline
- `utils/models.py` - model definitions
- `utils/utils.py` - utility functions including AdaIN logic
- `static/uploads/` - uploaded and generated images
- `templates/` - HTML templates
- `examples/` - sample content/style outputs
- `experiment/` - saved model checkpoints and experiment configs
- `vgg_normalised.pth` - pretrained encoder weights

## Requirements

Install the dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Run the App

From the project root, run:

```powershell
Set-Location "E:\STYLE IMAGE"; .\.venv\Scripts\python.exe app.py
```

Or, if using a normal terminal environment:

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000/
```

## Usage

1. Open the app in your browser.
2. Choose a content image and a style image.
3. Adjust the style strength slider if needed.
4. Click "Transfer Style".
5. View the generated result in the browser.

## Notes

- The app expects uploaded files to be `.png`, `.jpg`, or `.jpeg` images.
- Model files should remain in the project directory for inference to work properly.
- The output is saved under `static/uploads/`.

## License

This project is for educational and personal use unless otherwise specified by the repository owner.
