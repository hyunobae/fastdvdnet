"""Convert image sequences in dirs found under `datab_root` into high quality mp4 video files.
"""
import argparse
import os
import subprocess

default_codec = "h264"
default_crf = "18"
default_keyint = "4"


def convert_scenes(datab_root, out_root, codec, crf, keyint):
    # If not provided, set values by default
    if not codec:
        codec = default_codec
    assert codec in ['h264', 'hevc'], '--codec must be one of h264 or hevc'

    if not crf:
        crf = default_crf

    if not keyint:
        keyint = default_keyint

    # Codec options
    codec_args = ["-preset", "slow"]
    if codec == 'h264':
        codec_args = ["-c:v", "libx264", "-g", keyint,
                      "-profile:v", "high"]
    elif codec == 'hevc' or codec == 'h265':
        codec_args = ["-c:v", "libx265", "-x265-params",
                      "keyint=%s:no-open-gop=1" % (keyint)]
    else:
        raise ValueError("Unknown codec")

    # Quiet mode
    # if quiet:
    #     cmdout = subprocess.DEVNULL
    # else:
    #     cmdout = None

    # Output dir
    if not os.path.isdir(out_root):
        os.makedirs(out_root)
    print('Writing sequences to {}'.format(out_root))

    def convert(in_path, out_path):
        cmd = ["ffmpeg", "-y", "-i", in_path]
        cmd += codec_args
        cmd += ["-crf", crf, "-an", out_path]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd)

    # Convert sequences in subdirs under datab_root
    for subdir in os.listdir(datab_root):
        in_path = os.path.join(datab_root, subdir, '%2d.png')
        out_path = os.path.join(out_root, subdir + '.mp4')
        convert(in_path, out_path)

ind = "D:/testset/set8"
otd = "D:/testset/set8"
convert_scenes(ind, ind, default_codec, default_crf, default_keyint)