# SPD-MEF (Implementation of "Robust Multi-Exposure Image Fusion: A Structural Patch Decomposition Approach")
Python implementation of the algorithm in the paper "[Robust Multi-Exposure Image Fusion: A Structural Patch Decomposition Approach](https://ieeexplore.ieee.org/abstract/document/7859418/)"

---

## Usage
**a. Go to the project directory.**

```shell
cd SPD_MEF/
```

**b. Generate the fused enhanced image.**

```shell
python spd_mef.py --input_path ${INPUT_PATH} (Optional: --p ${P} --gsig ${GSIG} --lsig ${LSIG} --patch_size ${PATCH_SIZE} --step_size ${STEP_SIZE} --exp_thres ${EXP_THRES} --cons_thres ${CONS_THRES} --strt_thres ${STRT_THRES})
```

Please replace `${INPUT_PATH}` and the optional parameters in the shell of the command line with the correct value. You can use the following command to check the meaning of these optional parameters.
```shell
python spd_mef.py -h
```

