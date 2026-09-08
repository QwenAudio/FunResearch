# JAEC

We publicly release the CorrTrack-NStep inference runtime, comprising the neural TDE and LP modules, to facilitate comparisons with other adaptive filtering algorithms.

## Usage

```bash
python jaec.py -i test_sample/test.wav -o output.wav
```

The output has a fixed 22 ms algorithmic delay relative to the microphone input.

## Runtime Performance

Each RTF is the median of 10 runs of the command shown above.

| Operating system | Architecture | CPU | Main clock | RTF |
| --- | --- | --- | ---: | ---: |
| macOS | Arm64 | Apple M4 | 4.46 GHz | 0.0020 |
| Linux | x86-64 | Intel Xeon | 2.90 GHz | 0.0039 |
| Windows | x86-64 | Intel Core Ultra 9 185H | 2.30 GHz | 0.0029 |

## License

The source code, pretrained weights, and precompiled runtime libraries in this
repository are licensed under the [Apache License 2.0](LICENSE) (Apache-2.0).
Third-party components retain their original licenses; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
