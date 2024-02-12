# Dash Emulator Universal

This version of the DASH Headless Player can play dash videos over several different configurations.

## Supported Features

- 360 degree videos
- Live Streaming
- QUIC and TCP supported
- Streaming from local file system
- Bandwidth throttled streaming
- 3 different ABR algorithms - Bandwidth-based, Buffer-based and Hybrid
- Downloaded file saver
- Statistics Collector
- 2 different BW Estimation methods - Segment-based and instantaneous


## Building and running

### Setup

This project provides a nix shell in order to make development easier. Nix is a
package manager that provides fully isolated builds and development environments. 
This is especially useful since the project requires Python >= 3.10.

Set up devshell (create virtualenv and install correct python version):
``` bash
$ nix develop
```

Build project for the first time:
``` bash
$ pip install wheel
$ pip install .
```

When you make changes, you will need to rebuild the project with `pip install .`.
This will reexport the iplay command. Note that iplay is not an executable, it is
an alias defined in `setup.py`.

### Basic usage

Test and run iplay:
```bash
iplay -i <MPD_FILE_PATH>
```

## MPD formatting

MPEG-DASH uses a XML file called a .mpd (the "DASH manifest") to describe streams. 
Since these files are very complex, iStream Player only supports a subset of possible 
manifest formats. To ensure that your manifest works correctly with iStream Player,
you need to generate it using something like:

```bash
$ MP4Box -dash $SEGMENT_TIME -frag $SEGMENT_TIME -rap -bs-switching no -profile dashavc264:live -url-template output_${bitrate1}k.mp4:id="0" output_${bitrate2}k.mp4:id="1" output_${bitrate3}k.mp4:id="2" output_${bitrate4}k.mp4:id="3" output_${bitrate5}k.mp4:id="4" output_${bitrate6}k.mp4:id="5" -segment-name '$RepresentationID$/segments_m4s/segment_$Number$' -out multi_resolution.mpd
```

Where `SEGMENT_TIME` is the desired segment time in ms and `bitrateX` are the desired bitrates in kbps.
