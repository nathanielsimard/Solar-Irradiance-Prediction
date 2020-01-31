# IFT6759 - Project 1: Solar Irradiance Prediction

See the project presentation [here](https://docs.google.com/presentation/d/1jrozj0cLCsrJCBtXJKte6pHPNrkXh4_FM0KO05TIvGM/edit?usp=sharing).

## Documentation pages

- [Introduction](#GHI-Nowcasting-Introduction)
- [Imagery sources](#Data-Sources)
- [Metadata catalog](#Metadata-Catalog-Dataframe)
- [Evaluation](#Model-Evaluation-Guidelines)
- [Utilities](#Utility-modules)

# GHI Nowcasting Introduction

Note: this is a recap of some of the presentation slides available [here](solar_irradiance_presentation.pdf).

## Goal

Use satellite imagery ([GOES-13](https://en.wikipedia.org/wiki/GOES_13)) and any other relevant
metadata to predict [Global Horizontal Irradiance (GHI)](https://en.wikipedia.org/wiki/Solar_irradiance)
values as measured by seven [SURFRAD](https://www.esrl.noaa.gov/gmd/grad/surfrad/) stations in
the continental US.

The GHI at a specific geolocation depends on multiple factors including elevation, cloud cover,
surrounding land cover, turbidity, and air particles. The efficiency of solar panels can be tied
to the GHI. It can also be used in climate modeling.

Multiple GHI prediction models exist, but not all of them rely on satellite imagery. So-called
"clear sky" models rely on physical and mathematical models and can be used to estimate the
approximate GHI upper bound for any position on earth. These can serve as an auxiliary input
data source for imagery-based models.

The prediction horizon of interest for the project is ``[T_0, T_0 + 1h, T_0 + 3h, T_0 + 6h]``.
In other words, given a timestamp ``T_0``, we ask you to provide GHI values for that timestamp
as well as 1, 3, and 6 hours in the future. Due to the short-term nature of these predictions,
this is a called a "[nowcasting](<https://en.wikipedia.org/wiki/Nowcasting_(meteorology)>)" task.
To respect the nature of this problem, predictions for a given ``T_0`` timestep should **never**
rely on "future" imagery, that is imagery captured after ``T_0``. Models can however rely
on ``T_0`` as well as any timestep before that.

The idea behind using satellite imagery for GHI nowcasting is that some atmospheric phenomena
(e.g. clouds, particles) might be easier to track that way.

For this project, we are only evaluating our models at specific points on the map (i.e. at SURFRAD
stations). Ultimately, however, we are interested in models that can generalize to predict GHIs at
any point on the map. As such, you can only use data that would be available for the entirety of
the map. This means that your model cannot rely on past GHI values measured by stations on the
ground, since these would not be available at every point of the map.

## GOES-13, SURFRAD, and other metadata

We will provide all teams with preprocessed [GOES-13](https://en.wikipedia.org/wiki/GOES_13)
imagery. This imagery possesses five usable channels spanning the electromagnetic spectrum from
visible light to Mid-Infrared (MIR). GOES-13 imagery is geostationary and projected to a fixed
coordinate space. This means that image registration is not necessary between different
acquisition timestamps. This data is described in more detail [here](datasources.md).

A visualization of the GOES-13 imagery along with the measured GHI of the SURFRAD stations is
available [here](https://drive.google.com/file/d/12myylJZ_pDEORjvMpoHv-10O4HZIwW2y).

GOES-13 imagery is available from April 2010 to December 2016, inclusively, at 15-minute intervals.
The availability of this data is not perfect, and some files may be missing. The imagery channels
themselves may also contain pixels tagged as "missing" or "unavailable". It will be up to you to
handle these edge cases (filling in missing values, dealing with NaNs, etc.). You will have access
to the data ranging from April 2010 to December 2015, inclusively. The 2016 data is reserved for our
final (blind) test set. While this is technically publicly available data, we ask you
**not to use 2016 data to train your models**.

The SURFRAD stations provide GHI measurements at 1-minute intervals. To simplify the registration
of GOES-13 and SURFRAD data sources and to ensure uniformity among data, we provide a
[metadata catalog](dataframe.md) in the form of a Pandas dataframe. In that dataframe, the SURFRAD
measurements are smoothed to 15-minute intervals using a moving average. The SURFRAD measurements
may also contain missing values in the form of NaNs. Once again, it is up to the students to handle
these issues.

Finally, the provided metadata catalog includes daytime flags, clear sky GHI estimates and
cloudiness flags for all 15-minute entries. More information on these is provided [here](dataframe.md).

## Evaluation

Refer to the [evaluation page](evaluation.md).



# Data Sources

The raw NetCDF imagery of the Geostationary Operational Environmental Satellite (GOES)-13 provided by the
NOAA over the period of interest is quite voluminous (**over 800 GBs**). It is split into 15-minute chunks,
and the imagery itself covers all of the continental United States with several 16-bit channels. For this
project's goals, you might only need to focus on small areas of interest (i.e. patches) around the GHI
measurement (SURFRAD) stations. Furthermore, many time periods are not very useful for GHI prediction (for
example, night time values are not considered in the evaluation of the models). It is thus recommended to
preprocess this data in order to extract only the useful bits and speed up training. Remember: faster data
loading means more efficient training, and more efficient training means more time to try models and tune
hyperparameters.

As mentionned in the [disk usage documentation](../../disk-usage.md), the data for this project is available
in a shared read-only directory:
```
/project/cq-training-1/project1/data
```

We provide three versions of the GOES imagery in order to help you start the project:
 - The original, 16-bit, GZip-compressed (lossless) NetCDF (.nc) files, in 15-minute chunks.
 - Repackaged, 16-bit, JPEG2000-compressed (lossy) HDF5 (.h5) archives, in 1-day chunks.
 - Repackaged, 8-bit, JPEG-compressed (lossy) HDF5 (.h5) archives, in 1-day chunks.

We provide only a little bit of documentation for the content of these data sources below. For the NetCDF
data, you might be able to find more GOES-13 documentation online, but manual inspection using Python should be
sufficient to identify what to extract. For the HDF5 archives, opening them manually and using the utility
functions described [here](utilities.md) should be enough to get you started. Remember, real-world data is
rarely well-documented: you will almost always have to dig in and try to understand the structure on your own.
This is no different, and it should provide you a good opportunity to learn how to use debugging/inspection tools.

Finally, note that the easiest way to associate a UTC timestamp with the provided imagery sources is to rely on
the provided [dataframe](dataframe.md).

## Original NetCDF files

You might not be able to properly store, copy or duplicate the original NetCDF data due to storage limitations
on Helios. Furthermore, multiple users simultaneously accessing small files on the cluster can drastically
reduce I/O throughput. However, since it is untouched and in its original format, you can use the metadata of
the attributes and channels as you please.

Understanding the NetCDF file contents might be necessary in order to properly extract the channel arrays
without introducing (or carrying over) invalid values. We recommend that you look at the conventions for
the packaging of [Climate and Forecast Metadata](http://cfconventions.org/Data/cf-conventions/cf-conventions-1.7/build/ch08.html),
especially the sections regarding the use of scaling factors and missing/filled values. Those working with the
HDF5 data will have less options in terms of how they wish to handle missing/corrupted array values.

Besides, note that the channel data is compressed inside the NetCDF files themselves (GZip), and it will be
automatically decompressed when the file is opened by the Python package you are using. If you only want to
crop a small region around a station, loading and decompressing the data for the entire continental US in each
training minibatch might create a dramatic ingestion overhead. This is where cropping becomes useful...

You can use [``netCDF4``](https://unidata.github.io/netcdf4-python/netCDF4/index.html) or
[``h5netcdf``](https://github.com/shoyer/h5netcdf) to read NetCDF files in Python. The latter is less documented
but will it allow multithreaded read access to .nc files "out-of-the-box" with fewer issues.

## Decompressing HDF5 data

Utility functions for unpacking the compressed data arrays in the HDF5 files are provided [here](utilities.md).
These functions are based on [``OpenCV``](https://opencv.org/), but you could easily modify them to use any other
image reading library. HDF5 files can be easily opened using [``h5py``](https://www.h5py.org/).

The HDF5 archives contain the primary channels of interest extracted from the NetCDF files as well as the latitude
and longitude maps used to associate array values to geographic locations. Each of these is encoded into a dataset
object that supports seeking in order to reload one array at a time (i.e. at one timestamp). You should look at the
``fetch_hdf5_sample`` function of the utility module in order to reload an array. If you plan on repackaging your
own HDF5 files, feel free to use the same archive structure (and utility function), or devise your own.

For an overview of the HDF5 file format & specification, see [this introduction](https://support.hdfgroup.org/HDF5/Tutor/HDF5Intro.pdf).

Curious students are also encouraged to dig in and see exactly what information is lost in the compression process,
as what you learn might help you decide which data source to rely on...

### 16-bit HDF5 archives

The 16-bit, JPEG2000-compressed HDF5 archives require roughly 50% less storage space than the original
NetCDF files, and incur a maximum loss of less than ~1% over the original channel's value range. HDF5 archives
also allow small chunks of data to be read and decompressed individually. This combination provides a good tradeoff
between storage cost and data loss. However, JPEG2000 decompression is much slower than GZip or JPEG
decompression. It will be up to you to determine the impact of the various tradeoffs.

### 8-bit HDF5 archives

The 8-bit, JPEG-compressed HDF5 archives require nearly 85% less storage space than the original GZipped
NetCDF files. However, this pre-quantified JPEG compression is much more lossy than the other two alternatives,
as encoded values can fluctuate by up to 60% of their original range in the worst cases (i.e. in dark
regions). However, interestingly, this is not easily visible to the naked eye, and the imagery still looks
intact when visualized. You will have to determine whether these small perturbations can negatively affect the
behavior of your model, and whether the convenience of the extreme compression counterbalances this potential impact.

## Building an efficient data loading pipeline

See [this link](https://www.tensorflow.org/guide/data_performance) for a fairly in-depth tutorial on the development
and optimization of ``tf.data`` pipelines. We recommend focusing on efficient data loaders at an early point in
the project as it can become a serious bottleneck later on.

For optimal cluster I/O performance, it is recommended to store data in files that are at least 100MB, and inside
SLURM's temporary directory (``$SLURM_TMPDIR``).

### Pipeline formatting

We expect your data loading pipeline to be a ``tf.data.Dataset`` object (or have a compatible object interface).
Each iteration over this object should produce a tuple. Each tuple's last-position element should be a tensor
of target GHI values. All other tuple elements should be provided as inputs for your model. A simple pipeline
implementation will return two-element tuples, i.e. an input tensor for your model and the target GHI value tensor
it should predict. An unpacking strategy for the tuples produced by a pipeline is illustrated below:
```
data_loader = some.module.create_data_loader(...)
model = some.other.module.create_model(...)
...
# code below runs a single epoch
for (input_a, input_b, ..., target_output) in data_loader:
    with tf.GradientTape() as tape:
        predictions = model(input_a, input_b, ...)
        loss = criterion(y_true=target_output, y_pred=predictions)
    grads = tape.gradient(...)
    ...
...
```

Respecting this expected format will be important in order to make your data loading pipeline compatible with
our [evaluation script](evaluation.md). In that case, you will have to reinstantiate your pipeline to load the
(withheld) test data. This test data will **NOT** include the ground truth GHI values, meaning you will have
to either return zero-filled target output tensors, or simply return ``None`` as the last element of the
generated tuples. While this approach may seem strange to some, we argue that it is better than writing two
versions of your data loading pipeline and introducing new potential disparities.

# Metadata Catalog (Dataframe)

We provide a metadata catalog as a [Pandas dataframe](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html)
to simplify the indexing of raw GHI values and imagery data for training and evaluation purposes. Your
data loading pipeline is expected to scan this dataframe in order to know which data it should load.
A similar dataframe (minus several columns containing ground-truth values) will be used for the final
evaluation of your submitted model's performance ([see this page](evaluation.md) for more information).

The pickle file used to reinstantiate the Pandas dataframe is located in the shared (read-only) directory
mentionned in the [disk usage documentation](../../disk-usage.md), that is:
```
/project/cq-training-1/project1/data/catalog.helios.public.20100101-20160101.pkl
```

The dataframe is indexed using timestamps compatible with Python's ``datetime`` package ([more
info here](https://docs.python.org/3/library/datetime.html)). These timestamps are in Coordinated
Universal Time (or "UTC"), so do not be worried if sunrise/sunset times seem off. Just keep
this in mind if you plan on using local (station) times in your final submission. The dataframe possesses
an entry for every possible 15 minute interval over the entire 2010-2015 period. If data is missing in an
interval, the dataframe will still have an indexed entry, but some of its attributes may be "NaNs".

## Dataframe columns

Each column in the dataframe specifies an attribute for every 15-minute entry. Details on these attributes
are provided below:

 - ``ncdf_path``: The absolute path (on Helios) to the NetCDF file containing the raw (16-bit) imagery data
   for the specified 15-minute interval. Can be "nan" if unavailable. Reminder: the raw NetCDF files are
   only ~4MB chunks, and simultaneous I/O operations on these might become very slow on the cluster.
 - ``hdf5_<x>bit_path``: The absolute path (on Helios) to an HDF5 archive containing a compressed version
   (8-bit or 16-bit) of the imagery data for the specified 15-minute interval. Since this archive will likely
   be an aggregate of many 15-minute entries, the offset below must be used to locate the proper entry.
 - ``hdf5_<x>bit_offset``: An offset (non-negative integer) value used to point to the correct 15-minute
   data slice in an HDF5 archive that corresponds to this row's timestamp.
 - ``<station_code>_DAYTIME``: a binary flag (0/1) indicating whether it is day time at the station or not.
 - ``<station_code>_CLOUDINESS``: a categorical flag (night/cloudy/slightly cloudy/clear/variable) indicating
   the weather conditions at the station. This is a rough estimate based on a heuristic, and may be useful
   to analyze the performance of your model in various conditions. This value will not be available in the
   dataframe used for the final test.
 - ``<station_code>_CLEARSKY_GHI``: the GHI estimation at the station obtained using the "clear sky" model.
   See the section below for more information on the model used.
 - ``<station_code>_GHI``: the real (measured) GHI at the station. This is the "ground truth" that your
   model should predict, and it will obviously not be available in the dataframe used for the final test.
   Remember: this value is unavailable ("NaN") at at regular intervals (roughly every three hours starting
   at mightnight every day), but it might also be missing at other random timestamps.

For more information on the NetCDF and HDF5 files, see [this page](datasources.md).

## SURFRAD Stations

We are targeting seven stations of interest located in continental United States. Their coordinates
(latitude, longitude, elevation) are provided below. The station acronyms can be used to lookup the
columns related to that particular station in the catalog.

 - Bondville, IL ("BND") @ 40.05192, -88.37309, 230m;
 - Table Mountain, CO ("TBL") @ 40.12498, -105.23680, 1689m;
 - Desert Rock, NV ("DRA") @ 36.62373, -116.01947, 1007m;
 - Fort Peck, MT ("FPK") @ 48.30783, -105.10170, 634m;
 - Goodwin Creek, MS ("GWN") @ 34.25470, -89.87290, 98m;
 - Penn. State University, PA ("PSU") @ 40.72012, -77.93085, 376m;
 - Sioux Falls, SD ("SXF") @ 43.73403, -96.62328, 473m.

You will need the latitude/longitude coordinates above to locate the stations within the pixelized
arrays of the NetCDF/HDF5 files. These files contain the 1D arrays required to map these coordinate to
pixel offset values directly.

Finally, note that the final evaluation will use these same stations. Your report should nonetheless contain
a brief discussion of how well you would expect your model would perform if it was applied to other regions.

## Cloudiness flag

The cloudiness flag given in the dataframe is inspired by the work of Tina et al. (2012), "Analysis of
forecast errors for irradiance on the horizontal plane" (doi:10.1016/j.enconman.2012.05.031). The PDF of this
paper is available [in the repository](tina2012.pdf). The cloudiness flag may help you determine whether
your model truly outperforms the "clear sky" estimate by allowing you to target only cloudy days for
evaluation and comparison. You could also use it to compute general statistics on the input data, or even to
create a classification sub-task inside your model.

In the dataframe, this flag can only take five different values: "night", "cloudy", "slightly cloudy", "clear",
and "variable".

## Clearsky GHI estimations

The clearsky GHI estimates provided in the dataframe are based on the [the pvlib package](https://pvlib-python.readthedocs.io/en/stable/clearsky.html).
The model used is the one described by Ineichen and Perez (2002) in "A new airmass independent formulation for
the Linke turbidity coefficient" (doi:10.1016/S0038-092X(02)00045-2). Month-wise Linke turbidity factors are
queried for each station via latitude/longitude.

## Invalid data

As mentioned before, some entries of the dataframe may contain a "NaN" attribute instead of a GHI value or
an imagery file path. For increased safety, you should make sure to validate all imagery files regardless once
they are copied inside your team's directory (or once they are repackaged in your own intermediary format).

Remember also that the imagery itself might need to be padded if some pixels are missing for a specific timestamp.


# Model Evaluation Guidelines

We provide a script ([``evaluator.py``](evaluator.py)) in this repository that must be slightly modified in
order to prepare your own data loading pipeline and model for evaluation. The predictions of your model will
be exported and compared to the (withheld) groundtruth over the test period. Your prediction errors will then
be aggregated and compared to those of other teams to create a performance ranking.

## Evaluation script

The [``evaluator.py``](evaluator.py) script has two important functions: ``prepare_dataloader`` and
``prepare_model``. These will be imported into another nearly identical script for final analysis and execution.
On your side, ``evaluator.py`` contains the evaluation loop and everything else you need to produce and save
your model's predictions. If you modify the two functions and correctly manage to save predictions into text
files, you should be OK for the final test on our side.

Note that any modification to the ``evaluator.py`` script outside of the two target functions will be ignored.
If these functions were not implemented or if something breaks during evaluation, your model will not be ranked,
and you will be penalized.

You can import 3rd-party packages you require inside the target functions directly. However, these packages
should be available on PyPI, and these should be listed in the ``requirements.txt`` file submitted alongside
the evaluation script. Remember: the evaluation script will be running from within your team's submission
folder. This means you can easily import your own Python modules that are also placed within this folder.

### Data loader preparation (``prepare_dataloader``)

Our goal is to evaluate the performance of your model, but your model's behavior is tied to the way it receives
its input data. As such, you will need to "set the table" yourself and connect your own data pipeline to your
own model.

If your data pipeline does not only rely on the original data sources (NetCDF/HDF5 files), you will have to
generate the intermediary representations you need in the ``prepare_dataloader`` function. In any case, that
function must return a ``tf.data.Dataset`` object that is ready to generate input tensors for your model.
These tensors should be provided in a tuple, as documented [here](datasources.md#pipeline-formatting).

As mentioned in the project presentation, your data pipeline will have access to all the imagery during
evaluation. However, your model's predictions **cannot** rely on "future" imagery. This means that given
the list of timestamps to generate predictions for, you can only ever use imagery that comes **before
(or exactly at)** each of the timestamps. We will be heavily penalizing teams that do not respect this
rule in their final submission, and we already have scripts in place to detect this. If you are unsure about
this rule, you can ask a TA for clarification.

A configuration dictionary can optionally be used to provide (and keep track of) external hyperparameters
for your pipeline's constructor. If required, your final submission should include a JSON file named
``eval_user_cfg.json`` in your team's ``code`` folder (see [this page](../../disk-usage.md) for more info
on the submission directory structure). For more information on the ``prepare_dataloader`` function, refer
to its [docstring](evaluator.py).

### Model preparation (``prepare_model``)

The model preparation function itself can be fairly minimalistic depending on your model's architecture.
For example, users that built ``tf.keras``-compatible models will only need to fill this function with:
```
path = "/project/cq-training-1/project1/submissions/teamXX/model/best_model.pth"
model = tf.keras.models.load_model(path)
```
During the final evaluation, your submitted checkpoint should be located in the ``model`` directory alongside
your code. This means that you will be able to open it directly using its absolute path (as shown above). For
more information on the model submission process, refer to [this guide](../../howto-submit.md).

A configuration dictionary can optionally be used to provide (and keep track of) external hyperparameters
for your model's constructor. If required, your final submission should include a JSON file named
``eval_user_cfg.json`` in your team's ``code`` folder. For more information on the ``prepare_model``
function, refer to its [docstring](evaluator.py).

### Testing your modified evaluation script

A dummy dataframe with the same columns as the final test dataframe is provided [here](dummy_test_catalog.pkl)
for pre-testing purposes, and a compatible admin test file is provided [here](dummy_test_cfg.json).
These only rely on the data already at your disposal, but the real test set will rely on withheld data.

To test your modified evaluation script, you should run it from your team's submission code directory as such:
```
cd /project/cq-training-1/project1/submissions/teamXX/code
python evaluator.py output.txt dummy_test_cfg.json
```
If you plan on also submitting a ``eval_user_cfg.json`` file, you can start the script via:
```
python evaluator.py output.txt dummy_test_cfg.json -u="eval_user_cfg.json"
```
We will automatically be detecting the user config file and providing if needed it in our own batch
evaluation script.

As a particularity of the evaluation script, note that we will be providing you with a dictionary of
the stations of interest for which to prepare your data loader/model. We do this in order to control the
ordering of the predictions to make sure stations are not arbitrarily mixed. In practice, this means that,
during the final test, your ``prepare_dataloader`` and ``prepare_model`` functions may be called up to seven
times each (once per station).

Regarding the "datetimes" that will be used for the final test: as mentioned in the project presentation, we
will be focused on daytime GHI predictions. Also, for a single "target datetime" (``T_0``), remember that we
expect your model to produce four GHI values, i.e. one for each of ``[T_0, T_0 + 1h, T_0 + 3h, T_0 + 6h]``.
Due to the wide "horizon" that a single target datetime covers, it is possible that predictions for ``T_0`` may
fall in nighttime (e.g. if ``T_0`` is 4PM).  In order to still properly cover all real use cases of a GHI
prediction model, we will still ask for prediction sequences that partly fall after sunset or before sunrise.
In those cases, only the GHI predictions that correspond to timestamps in the day will be kept for analysis.

Finally, note that your model should in no circumstances produce Not-A-Number (NaN) values as output. The
evaluation will throw an error if it does, and you will be penalized. If the groundtruth contains a NaN value
for a target GHI, we will ignore its impact on our end. We will also only focus on daytime sequences and thus
ignore nighttime predictions.

The TAs will be offering evaluation "simulations" during the last class before the submission deadline. This
will allow you to confirm that:

  - Your team's submission has the expected directory structure;
  - The dependencies inside your ``requirements.txt`` file can be installed in our job environment;
  - Your ``evaluator.py`` script's modified functions are compatible with our version of the script;
  - Your model's checkpoint is indeed accessible and reloadable on our GPUs;
  - The preparation of your data loading pipeline and model do not take too long or hang; and
  - Your model's performance is not completely abysmal.

The last point means that TAs might choose to warn you if your model seems to be behaving oddly, but they
will not give you a direct indication of how well your model performs. We expect the final test set to contain
roughly 800-1000 different timestamps for which to generate GHI predictions. The time required to prepare your
data loader/model and infer these GHI values should not exceed (roughly) 30 minutes.


# Utility modules

You will have to write many functions during this project that may contain common and reusable
code snippets. Such snippets (once properly tested!) are always nice to keep around in some
kind of 'utility' module. We decided to provide a handful of such useful functions that
may help you develop visualization and debugging tools faster. Some of them might also be 
necessary should you decide to create your own data packaging strategy based on HDF5 or
compressed images. You can find them [here](utils.py).

Some of the noteworthy utility functions are detailed below. You should always consider these
functions as "potentially buggy" unless you understand them and have tested them yourself.
Feel free to modify and add functions to the ``utils.py`` module as you wish. Finally, make
sure you install the proper dependencies (e.g. OpenCV, matplotlib) before using them.

## Compressing and decompressing numpy arrays

Your data loading strategy might require you to extract and repackage useful arrays from the
NetCDF or HDF5 files we provide in order to save disk space and/or speed up processing. If
you use our HDF5 files, you will have to manually decompress the information stored in them
first. If you want to create new HDF5 files, you might have to compress your own data as well.

For compression, we provide a simple utility function (``compress_array``) that converts
numpy arrays into byte strings that can be stored in dynamic-length HDF5 datasets. For
decompression, we provide the opposite operation (``decompress_array``) which will convert
the byte string back into a numpy array. In both cases, depending on the encoding used,
you might have to import various 3rd-party dependencies into your code (e.g. OpenCV, LZ4).

Example usage:
```
>>> array = np.random.randint(255, size=(300, 300)).astype(np.uint8)
>>> array  # since this is a 2D 8-bit array, default auto compression will use JPEG
array([[231,   5,  46, ...,  55,  92,  71],
       ...,
       [217, 140, 204, ...,  19, 184, 151]], dtype=uint8)
>>> code = utils.compress_array(array, compr_type="auto")
>>> code[0:30]  # this will show the beginning of the JPEG header...
b'uint8+jpg\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x01,\x01,\x00\x00\xff'
>>> len(code)  # should be smaller than 300x300 = 90000, i.e. the original size
77420
>>> decoded_array = utils.decompress_array(code, dtype=np.uint8, shape=(300, 300))
>>> decoded_array  # since it used JPEG, there might be some information loss
array([[231,   4,  46, ...,  54,  91,  71],
       ...,
       [218, 142, 202, ...,  21, 183, 146]], dtype=uint8)
```

## Reading numpy arrays from HDF5 files

The HDF5 archives we provide contain compressed numpy arrays as detailed [here](datasources.md).
To properly reload these arrays, we provide a utility function (``fetch_hdf5_sample``)
that can be used on an already-opened HDF5 archive. This function will call the
``decompress_array`` utility detailed above as necessary. The "sample index" argument
given to ``fetch_hdf5_sample`` corresponds to the offset encoded into the metadata dataframes
described [here](dataframe.md).

Example usage:
```
>>> hdf5_path = "/project/cq-training-1/project1/data/hdf5v7_8bit/2010.06.01.0800.h5"
>>> hdf5_offset = 32  # this would correspond to: 2010.06.01.0800 + (32)*15min = 2010.06.01.1600
>>> with h5py.File(hdf5_path, "r") as h5_data:
>>>   ch1_data = utils.fetch_hdf5_sample("ch1", h5_data, hdf5_offset)
>>> ch1_data.shape  # channel data is saved as 2D array (HxW)
(650, 1500)
>>> ch1_data  # the utility function will automatically reconvert the data to float32
array([[-1.0784250e-04, -1.0784250e-04, -1.0784250e-04, ...,
         3.7579414e-01,  3.4611768e-01,  3.6590198e-01],
       ...,
       [ 4.9450004e-01,  4.7471571e-01,  4.9450004e-01, ...,
         7.3191184e-01,  7.8137261e-01,  8.2094121e-01]], dtype=float32)
```

## Visualizing the content of an HDF5 file

Both 8-bit and 16-bit HDF5 archives can be visualized using the provided ``viz_hdf5_imagery``
function. This may help you understand the role of each available imagery channel and the
impact of cloud cover over a measurement station.

The animation shown in the project presentation is obtained using this function with 8-bit
imagery for June 21st, 2010, and for the following channels: ``["ch1", "ch2", "ch3", "ch4", "ch6"]``.

Remember that you might not be able to forward display windows from Helios nodes to your
local display manager. This might force you to run these visualization completely offline.

Example usage:
```
>>> hdf5_path = "/some/local/path/project1/data/hdf5v7_8bit2010.06.21.0800.h5"
>>> target_channels = ["ch1", "ch2", "ch3", "ch4", "ch6"]
>>> dataframe_path = "/some/local/path/project1/data/some.local.catalog.pkl"
>>> stations = {"BND": (lat, lon, elev), "TBL": (lat, lon, elev), ...}
>>> viz_hdf5_imagery(hdf5_path, target_channels, dataframe_path, stations)
# the above line will block until the visualization is stopped...
```

## Visualizing GHI predictions

Finally, we provide a function to display GHI plots (measured values, clearsky estimates,
and predicted values) for a set of stations over different time horizons: ``viz_predictions``.
In this case, the function expects to receive the text file generated by the evaluation
script ([described here](evaluation.md)) which contains your model's raw predictions for a
set of timestamps. These timestamps must also be provided through the test configuration file,
for which you also have an example [here](dummy_test_cfg.json).

Feel free to ignore this function and instead only rely on its subfunctions (``draw_daily_ghi``
and ``plot_ghi_curves``) if you want to incorporate it into your own code, or if you wish
to customize it in any way.

Remember that you might not be able to forward display windows from Helios nodes to your
local display manager. This might force you to run these visualization completely offline.




