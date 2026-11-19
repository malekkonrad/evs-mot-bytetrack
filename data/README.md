# Data

The dataset is not included in this repository. Place it here so the layout is:

```
data/evs_mot_public_dataset/
├── evs_mot-train/MOT_02 … MOT_05/{det,gt,img1,seqinfo.ini}
└── evs_mot-test/MOT_01, MOT_06, MOT_07/{det,img1,seqinfo.ini}
```

A symlink works too: `ln -s /path/to/evs_mot_public_dataset data/`.
