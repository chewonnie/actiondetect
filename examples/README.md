# Example Input Videos

This folder contains small URFD RGB-only and ETRI RGB videos for dashboard input.
All example video files are grouped in `examples/videos/`.
Use them in `1b. 원본 동영상 입력 분석`.

## Files

| file | label | dashboard input type |
|---|---|---|
| `videos/urfd_fall_01_rgb.mp4` | fall | `일반 RGB 영상` |
| `videos/urfd_fall_02_rgb.mp4` | fall | `일반 RGB 영상` |
| `videos/urfd_adl_01_rgb.mp4` | ADL | `일반 RGB 영상` |
| `videos/urfd_adl_02_rgb.mp4` | ADL | `일반 RGB 영상` |
| `videos/etri_p10_a001_eating.mp4` | eating | `일반 RGB 영상` |
| `videos/etri_p10_a003_medicine.mp4` | medicine | `일반 RGB 영상` |
| `videos/etri_p10_a020_shoes.mp4` | hygiene_grooming | `일반 RGB 영상` |
| `videos/etri_p10_a053_pickup_litter.mp4` | mobility | `일반 RGB 영상` |

## Source / License

Source: UR Fall Detection Dataset

- Page: http://fenix.ur.edu.pl/~mkepski/ds/uf.html
- License: CC BY-NC-SA 4.0
- Required citation: Bogdan Kwolek, Michal Kepski. Human fall detection on embedded platform using depth maps and wireless accelerometer. Computer Methods and Programs in Biomedicine, Vol. 117, Issue 3, Dec 2014, pp. 489-501.

These files were copied from `datasets/fall/urfd/**` and cropped from the
original `cam0` layout to keep only the right-side RGB region. The left-side
depth map is not included in `examples/`.

Source: ETRI Activity3D / EPreTX RGB sample videos

- Local source: `/Users/chaewon/Downloads/etri/RGB/P10/20210322PM_S02_H120_P10/`
- Subject: `P10`
- Dashboard input type: `일반 RGB 영상`

Check the dataset redistribution terms before publishing these ETRI samples to
a public repository.
