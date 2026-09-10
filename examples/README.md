# Examples

The commands in the main README use user-prepared tables under `data/` and write generated artifacts under `outputs/`.

For an offline installation check that does not require research data or pretrained-model downloads, run:

```bash
python scripts/run_smoke_test.py --config configs/default.yaml
```

The smoke test uses synthetic strings and small molecular graphs only. It validates tensor shapes, a forward pass, binary loss, gradients, and metric plumbing; it is not a scientific benchmark.

