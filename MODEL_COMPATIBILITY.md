# Model Compatibility Notes

## Working Models
The following models work correctly with the current setup:
- `aasist` - ✅ Compatible 
- `rawnet_2` - ✅ Compatible
- `tcm_add` - ✅ Compatible  
- `rawgat_st` - ✅ Compatible
- `nes2net_x` - ✅ Compatible
- `wav2vec2_aasist` - ✅ Compatible
- `hubert_ecapa` - ✅ Compatible (uses .ckpt files)
- `wav2vec2_ecapa` - ✅ Compatible (uses .ckpt files)
- `wavlm_ecapa` - ✅ Compatible (uses .ckpt files)

## Known Issues

### xlsr_sls Model
**Status:** ❌ Incompatible with current fairseq version

**Issue:** The `xlsr_sls` model depends on the XLSR-300M checkpoint (`xlsr2_300m.pt`) which was created with an older fairseq version. The current fairseq version (0.12.2) has incompatible configuration schemas.

**Error:** Configuration keys like `multiple_train_files`, `eval_wer`, `eval_wer_config` are not recognized by the newer fairseq version.

**Workarounds:**
1. **Use different models:** Choose from the working models listed above
2. **Downgrade fairseq:** Install `fairseq==0.10.2` (may have build issues)
3. **Update checkpoint:** Recreate the XLSR-300M checkpoint with current fairseq version

**Recommended:** Use `aasist`, `rawnet_2`, or other compatible models instead.

## Usage Examples

### Working model:
```bash
./run_evaluation.sh --protocol_files fake_or_real --batch_size 32 --fix_length --device cpu --models aasist
```

### Multiple models:
```bash
./run_evaluation.sh --protocol_files fake_or_real --batch_size 32 --fix_length --device cpu --models aasist rawnet_2 tcm_add
```

### All compatible models:
```bash
./run_evaluation.sh --protocol_files fake_or_real --batch_size 32 --fix_length --device cpu --models all
```
(Note: This will include xlsr_sls and fail - specify individual models instead)