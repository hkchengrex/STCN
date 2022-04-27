# STCN Multi-scale testing

This is a scratch implementation of multi-scale testing of STCN. Following Section E in the Appendix, we take the average output of {480p, 600p} * {flip, no-flip} versions of the video. One should be able to get *85.2 J&F* in the YouTubeVOS 2019 validation set with this setting.

The idea is that we compute each option separately, save the probability maps on disk, then merge the results to obtain the final mask.

**Run these commands in the root folder of STCN, not in the `ms` folder.**

Computing step:
```
    python -m ms.eval_youtube --output ../output/yv19-ms/nf-480-m5 --res 480
    python -m ms.eval_youtube --output ../output/yv19-ms/f-480-m5 --res 480 --flip
    python -m ms.eval_youtube --output ../output/yv19-ms/nf-600-m5 --res 600 --use_km 
    python -m ms.eval_youtube --output ../output/yv19-ms/f-600-m5 --res 600 --use_km --flip
```
(You can change the output path to whatever you like. If you have multiple GPUs, you can run them in parallel.)

Merging step:
```
    python -m ms.merge_ensemble --list ../output/yv19-ms/f-480-m5 ../output/yv19-ms/f-600-m5 ../output/yv19-ms/nf-480-m5 ../output/yv19-ms/nf-600-m5  --output ../output/yv19-ms/merged
```

Then `zip` the `Annotations` folder in `../output/yv19-ms/merged`. You can submit the zipped result to the [evaluation server](https://competitions.codalab.org/competitions/20127#participate).

Precomputed results: https://drive.google.com/file/d/1m3x-EWCTy70Lwco1pQl_3D73buK2Vvbo/view?usp=sharing