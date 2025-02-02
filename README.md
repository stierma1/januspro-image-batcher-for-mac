# Text-to-Image Generator Using Deepseek JanusPro-7B Model

A Python-based tool that generates high-quality images from textual descriptions using the Deepseek JanusPro-7B model. The tool reads configuration files containing prompts and generation parameters, making it easy to batch process multiple image requests.

![image info](./sample-output/coffee-store-robot-1.jpg)
![image info](./sample-output/suited-robot-1.jpg)

## Features

- Generate images based on text prompts
- Process multiple prompts from a single JSON configuration file
- Customizable output filenames using templates
- Support for different seeds, temperatures, and guidance scales
- Automatic saving of generated images to specified directory

## Installation

```bash 
git clone https://github.com/your-repo/text-to-image-generator.git
pip install -r requirement.txt
```

## Usage

1. Prepare a configuration file (`config.json`) containing your prompts and generation parameters:

```json

   [
    {
        "prompt": "Robot casually coding in a coffee store",
        "fileNameTemplate": "coffee-store-robot-{#}.jpg",
        "temperature": 1,
        "seed": 12345,
        "guidance": 8,
        "number_of_samples": 3
    },
    {
        "prompt": "Robot wearing a suit",
        "fileNameTemplate": "suited-robot-{#}.jpg",
        "temperature": 1,
        "seed": 12345,
        "guidance": 8,
        "number_of_samples": 3
    }
]
```

2. Run the generator script:
```python
python main.py --config ./path/to/config.json --output ./output/directory
```
## Performance

The tool has been tested on a Macbook Pro M4 Max (2024) with the following specifications:
- Apple M4 Max chip (16-core CPU, 40-core GPU)
- 48GB unified memory
- macOS sequoia 15.0+

On this hardware, generating 6 images (2 prompts × 3 candidates each) takes 100 seconds assuming the model is already downloaded

## Contributing

Contributions are welcome! Feel free to submit pull requests or report issues on the [issue tracker](https://github.com/your-repo/text-to-image-generator/issues).

## License

MIT

## Acknowledgments

- This project leverages the Deepseek JanusPro-7B model for text-to-image generation.
- Special thanks to the open-source community for maintaining and improving these powerful tools.

## Note

This is an experimental implementation. While the generated images are generally of high quality, results may vary based on the complexity and clarity of your prompts.