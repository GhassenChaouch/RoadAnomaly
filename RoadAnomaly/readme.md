to run in cmd:
pip install -r requirements.txt   




# Road Anomaly dataset

![](frames/animals06_sheep_roads_lambs.webp)

This dataset contains images of unusual dangers which can be encountered by a vehicle on the road - animals, rocks, traffic cones and other obstacles.
Its purpose is testing autonomous driving perception algorithms in rare but safety-critical circumstances.

The dataset contains images with associated per-pixel labels.
The labeling has been performed with our [LabelGrab](github.com/cvlab-epfl/LabelGrab) tool. Most frames retain the editor's files and can be further edited; some are missing because the labeling was done at a different resolution and rescaled. 
If you need the editor files and instance labels for all frames, please [contact us](mailto:krzysztof.lis@epfl.ch).

For any additional information, please contact [Krzysztof Lis](mailto:krzysztof.lis@epfl.ch).

The images are provided for research purposes.
We do not own the images; the authors and sources are listed in `credits.txt`.

### Directory structure

`RoadAnomaly`
- `readme.md`
- `frame_list.json` - list of all image file names
- `credits.txt` - sources of the images
- `frames/`
  - `frame_name.webp` - image
  - `frame_name.labels/labels_semantic.png` - per pixel label, background =  `0` and anomaly = `2`
  - `frame_name.labels/labels_instance.png` - instance id of objects, some are missing at this time.

### References
If you find this dataset useful, please cite:

```
@inproceedings{DetectingTheUnexpected2019,
	title = {Detecting the Unexpected via Image Resynthesis},
	author = {Krzysztof Lis and Krishna Nakka and Pascal Fua and Mathieu Salzmann},
	booktitle = {International Conference on Computer Vision},
	year = 2019,
	url = {https://infoscience.epfl.ch/record/269093?ln=en}
}
```
