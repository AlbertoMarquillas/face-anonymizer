# Data Folder

This folder is reserved for test images and videos used with the face anonymizer.

## Structure

```
data/
├─ README.md
├─ .gitkeep
└─ your local files here
```

## Usage

* Place your own test images/videos inside this folder.
* Use the CLI to anonymize them, for example:

  ```powershell
  python .\src\main.py --input .\data\your_image.jpg --method blur --output .\build\out.jpg
  ```

## Notes

* Do **not** commit datasets or media files into this repository.
* Only `README.md` and `.gitkeep` are versioned.
* Large or sensitive files should remain local.
