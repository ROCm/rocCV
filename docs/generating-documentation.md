# Generating Documentation

rocCV uses Doxygen for API documentation generation. To generate documentation locally, refer to the following instructions:

1. Install Doxygen:
```bash
sudo apt install doxygen
```

2. In the root folder of the project where the `Doxyfile` configuration file is located, run the following command:
```bash
doxygen Doxyfile
```

3. The generated documentation files will be located in `docs/doxygen`.